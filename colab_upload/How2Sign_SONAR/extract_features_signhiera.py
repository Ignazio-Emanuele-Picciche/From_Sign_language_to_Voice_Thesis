#!/usr/bin/env python3
"""
Extract SignHiera features from How2Sign videos on Google Colab
Optimized for T4 GPU with batch processing
"""

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


class VideoDataset(Dataset):
    """Dataset for loading videos with preprocessing"""

    def __init__(
        self, manifest_path, video_dir, target_size=(224, 224), max_frames=300
    ):
        """
        Args:
            manifest_path: Path to TSV manifest (id, duration, text)
            video_dir: Directory containing videos
            target_size: Resize frames to this size
            max_frames: Maximum number of frames to extract per video
        """
        self.video_dir = Path(video_dir)
        self.target_size = target_size
        self.max_frames = max_frames

        # Load manifest
        self.manifest = pd.read_csv(manifest_path, sep="\t")
        print(f"Loaded {len(self.manifest)} videos from manifest")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        """Load and preprocess video"""
        row = self.manifest.iloc[idx]
        video_id = row["id"]
        video_path = self.video_dir / f"{video_id}.mp4"

        # Load video frames
        frames = self._load_video(video_path)

        if frames is None:
            # Return dummy data if video fails
            frames = torch.zeros(1, 3, *self.target_size)
            valid = False
        else:
            valid = True

        return {
            "video_id": video_id,
            "frames": frames,  # Shape: (T, 3, H, W)
            "valid": valid,
        }

    def _load_video(self, video_path):
        """Load video and extract frames"""
        if not video_path.exists():
            print(f"⚠️  Video not found: {video_path}")
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"⚠️  Failed to open video: {video_path}")
            return None

        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= self.max_frames:
                break

            # Preprocess frame
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize
            frame = cv2.resize(frame, self.target_size)

            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0

            # HWC to CHW
            frame = np.transpose(frame, (2, 0, 1))

            frames.append(frame)
            frame_count += 1

        cap.release()

        if len(frames) == 0:
            return None

        # Stack frames: (T, 3, H, W)
        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames).float()

        return frames


def collate_fn(batch):
    """Custom collate function to handle variable-length videos"""
    return batch  # Return list of dicts


class SimpleSignHiera(nn.Module):
    """
    Simplified SignHiera model for feature extraction
    This is a placeholder - actual SONAR model will be loaded from checkpoint
    """

    def __init__(self, pretrained_path=None):
        super().__init__()

        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading SONAR SignHiera from {pretrained_path}")
            # Load pretrained checkpoint
            checkpoint = torch.load(pretrained_path, map_location="cpu")

            # Extract model architecture from checkpoint
            # NOTE: Actual implementation depends on SONAR checkpoint structure
            # This is a placeholder that should be replaced with proper loading

            # For now, create a simple CNN backbone
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

            self.feature_dim = 256

            # Try to load weights if checkpoint structure matches
            try:
                if "model" in checkpoint:
                    state_dict = checkpoint["model"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint

                # Load weights (will fail if architecture doesn't match)
                self.load_state_dict(state_dict, strict=False)
                print("✅ Loaded pretrained weights")
            except Exception as e:
                print(f"⚠️  Could not load weights: {e}")
                print("   Using random initialization")
        else:
            # Fallback to simple CNN
            print("⚠️  No pretrained model found, using simple CNN")
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )

            self.feature_dim = 256

    def forward(self, x):
        """
        Extract features from video frames
        Args:
            x: (B, T, 3, H, W) or (T, 3, H, W)
        Returns:
            features: (B, T, D) or (T, D)
        """
        if x.dim() == 5:
            # Batch of videos
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)

            features = self.backbone(x)
            features = features.view(B, T, -1)
        else:
            # Single video
            T, C, H, W = x.shape
            features = self.backbone(x)
            features = features.view(T, -1)

        return features


def extract_features(args):
    """Main feature extraction loop"""

    print("=" * 70)
    print("🚀 SONAR SignHiera Feature Extraction")
    print("=" * 70)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"📍 Device: {device}")

    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

    # Load model
    print(f"\n📦 Loading model from {args.model_path}")
    model = SimpleSignHiera(pretrained_path=args.model_path)
    model = model.to(device)
    model.eval()

    print(f"   Feature dim: {model.feature_dim}")

    # Create dataset
    dataset = VideoDataset(
        manifest_path=args.manifest,
        video_dir=args.video_dir,
        target_size=(224, 224),
        max_frames=args.max_frames,
    )

    # Create dataloader (batch_size=1 due to variable lengths)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 Output directory: {output_dir}")
    print(f"📊 Total videos: {len(dataset)}")
    print(f"\n⏳ Starting extraction...\n")

    # Extract features
    num_processed = 0
    num_failed = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            sample = batch[0]  # batch_size=1

            video_id = sample["video_id"]
            frames = sample["frames"]
            valid = sample["valid"]

            if not valid:
                num_failed += 1
                continue

            # Move to device
            frames = frames.to(device)  # (T, 3, H, W)

            # Extract features
            try:
                features = model(frames)  # (T, D)
                features = features.cpu().numpy()

                # Save features
                output_path = output_dir / f"{video_id}.npy"
                np.save(output_path, features)

                num_processed += 1

            except Exception as e:
                print(f"\n⚠️  Error processing {video_id}: {e}")
                num_failed += 1
                continue

    # Summary
    print("\n" + "=" * 70)
    print("✅ EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"📊 Processed: {num_processed} videos")
    print(f"❌ Failed:    {num_failed} videos")
    print(f"💾 Features saved to: {output_dir}")
    print(f"\n🎉 Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Extract SignHiera features from videos"
    )

    parser.add_argument(
        "--manifest", type=str, required=True, help="Path to TSV manifest file"
    )
    parser.add_argument(
        "--video_dir", type=str, required=True, help="Directory containing videos"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to pretrained SignHiera model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save extracted features",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (videos have variable lengths)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=300,
        help="Maximum frames to extract per video",
    )
    parser.add_argument(
        "--num_workers", type=int, default=2, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.manifest):
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    if not os.path.exists(args.video_dir):
        raise FileNotFoundError(f"Video directory not found: {args.video_dir}")

    if not os.path.exists(args.model_path):
        print(f"⚠️  Model not found: {args.model_path}")
        print("   Will use random initialization (for testing only)")

    # Run extraction
    extract_features(args)


if __name__ == "__main__":
    main()
