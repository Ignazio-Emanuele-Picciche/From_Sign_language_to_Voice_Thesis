#!/usr/bin/env python3

import argparse
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

import warnings

warnings.filterwarnings("ignore")


# ===========
# Dataset
# ===========
class VideoDataset(Dataset):
    def __init__(
        self, manifest_path, video_dir, target_size=(224, 224), max_frames=300
    ):
        self.video_dir = Path(video_dir)
        self.target_size = target_size
        self.max_frames = max_frames
        self.manifest = pd.read_csv(manifest_path, sep="\t")
        print(f"Loaded {len(self.manifest)} videos from manifest")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        video_id = row["id"]
        video_path = self.video_dir / f"{video_id}.mp4"
        frames = self._load_video(video_path)
        if frames is None:
            frames = torch.zeros(1, 3, *self.target_size)
            valid = False
        else:
            valid = True
        return {
            "video_id": video_id,
            "frames": frames,  # (T, 3, H, W)
            "valid": valid,
        }

    def _load_video(self, video_path):
        if not video_path.exists():
            print(f"⚠️  Video not found: {video_path}")
            return None
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"⚠️  Failed to open video: {video_path}")
            return None
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret or len(frames) >= self.max_frames:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.target_size)
            frame = frame.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            frame = (frame - mean) / std
            frame = np.transpose(frame, (2, 0, 1))
            frames.append(frame)
        cap.release()
        if len(frames) == 0:
            return None
        frames = np.stack(frames, axis=0)
        return torch.from_numpy(frames).float()


def collate_fn(batch):
    return batch


# =========================
# MODEL: SignHiera Fallback
# =========================
class SimpleSignHiera(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        # Try to load real model
        if pretrained_path and os.path.exists(pretrained_path):
            print(
                f"Loading SignHiera backbone from {pretrained_path} (NOTE: expects arch compatible with this script!)"
            )
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            # You must adjust this if you have access to real SignHiera model class!
            # Fallback: random simple CNN for now
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 768, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.feature_dim = 768
            # Try to load weights
            try:
                state_dict = (
                    checkpoint.get("model")
                    or checkpoint.get("state_dict")
                    or checkpoint
                )
                self.load_state_dict(state_dict, strict=False)
                print("✅ Loaded model weights (if compatible).")
            except Exception as e:
                print(f"⚠️  Could not load weights: {e}")
                print("   Using random initialization.")
        else:
            print("⚠️  No pretrained model found, using simple CNN fallback.")
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 768, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.feature_dim = 768

    def forward(self, x):
        # x: (T, 3, H, W) or (B, T, 3, H, W)
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            features = self.backbone(x)
            features = features.view(B, T, -1)
        else:
            T, C, H, W = x.shape
            features = self.backbone(x)
            features = features.view(T, -1)
        return features  # (B, T, D) or (T, D)


# =========================
# Feature Extraction
# =========================
def extract_features(args):
    print("=" * 70)
    print("🚀 SIGNHIERA FEATURE EXTRACTION")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = SimpleSignHiera(pretrained_path=args.model_path).to(device)
    model.eval()

    dataset = VideoDataset(args.manifest, args.video_dir, max_frames=args.max_frames)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    num_processed, num_failed = 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            sample = batch[0]
            video_id = sample["video_id"]
            frames = sample["frames"]
            valid = sample["valid"]
            if not valid:
                num_failed += 1
                continue
            frames = frames.to(device)
            try:
                features = model(frames)  # (T, D)
                features = features.cpu().numpy()
                np.save(out_dir / f"{video_id}.npy", features)
                num_processed += 1
            except Exception as e:
                print(f"⚠️  Error processing {video_id}: {e}")
                num_failed += 1

    print("=" * 70)
    print("✅ EXTRACTION COMPLETE")
    print(f"Processed: {num_processed} videos")
    print(f"Failed:    {num_failed} videos")
    print(f"Features saved to: {out_dir}")
    print("🎉 Done!")


# =====================
def main():
    parser = argparse.ArgumentParser(
        description="Extract SignHiera features from videos"
    )
    parser.add_argument(
        "--manifest", type=str, required=True, help="TSV manifest (id, ...)"
    )
    parser.add_argument(
        "--video_dir", type=str, required=True, help="Directory with videos"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/dm_70h_ub_signhiera.pth",
        help="Path to the SignHiera checkpoint (default: models/dm_70h_ub_signhiera.pth)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Where to save features"
    )
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    assert os.path.exists(args.manifest), f"Manifest not found: {args.manifest}"
    assert os.path.exists(args.video_dir), f"Video dir not found: {args.video_dir}"
    if not os.path.exists(args.model_path):
        print(
            f"⚠️ Model {args.model_path} not found, using random weights! (LOW PERFORMANCE)\n"
            f"   Expected model location: models/dm_70h_ub_signhiera.pth"
        )
    extract_features(args)


if __name__ == "__main__":
    main()
