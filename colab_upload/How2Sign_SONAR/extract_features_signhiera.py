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
import math
import warnings

warnings.filterwarnings("ignore")

try:
    from hiera import hiera_base_16x224
except ImportError:
    raise ImportError("Installare la libreria: pip install hiera-transformer")


# ===========
# Dataset (Accetta DataFrame filtrato)
# ===========
class VideoDataset(Dataset):
    def __init__(self, dataframe, video_dir, target_size=(224, 224), target_frames=128):
        self.video_dir = Path(video_dir)
        self.target_size = target_size
        self.target_frames = target_frames
        self.manifest = dataframe.reset_index(drop=True)
        print(f"📋 Dataset initialized with {len(self.manifest)} videos to process.")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        video_id = row["id"]
        # Assumiamo estensione .mp4, modifica se necessario
        video_path = self.video_dir / f"{video_id}.mp4"

        frames = self._load_video(video_path)

        if frames is None:
            return None

        return {"video_id": video_id, "frames": frames}

    def _load_video(self, video_path):
        # --- FIX ROBUSTEZZA I/O (Drive) ---
        try:
            if not video_path.exists():
                return None
        except OSError:
            print(f"⚠️ I/O Error checking path (skip): {video_path}")
            return None

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        raw_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.target_size)
            raw_frames.append(frame)
        cap.release()

        if len(raw_frames) == 0:
            return None

        raw_frames = np.stack(raw_frames)

        # Sampling 128 frames
        T_orig = len(raw_frames)
        if T_orig > 0:
            indices = np.linspace(0, T_orig - 1, self.target_frames).astype(int)
            frames = raw_frames[indices]
        else:
            return None

        frames = frames.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frames = (frames - mean) / std

        # Output shape: (T, H, W, C) -> (C, T, H, W)
        frames = np.transpose(frames, (3, 0, 1, 2))
        return torch.from_numpy(frames).float()


def collate_fn_skip_none(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    video_ids = [x["video_id"] for x in batch]
    frames = torch.stack([x["frames"] for x in batch])  # (B, C, T, H, W)
    return {"video_ids": video_ids, "frames": frames}


# =========================
# MODEL: REAL SignHiera
# =========================
class RealSignHiera(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        # print(f"🏗️  Building SignHiera architecture...") # Ridotto log
        self.backbone = hiera_base_16x224(pretrained=False)
        self.backbone.head = nn.Identity()

        expected_temp_pos = 64
        current_temp_pos = self.backbone.pos_embed_temporal.shape[1]

        if current_temp_pos != expected_temp_pos:
            new_pos_embed = nn.Parameter(torch.zeros(1, expected_temp_pos, 96))
            self.backbone.pos_embed_temporal = new_pos_embed
            self.backbone.tokens_temporal = expected_temp_pos

        if pretrained_path and os.path.exists(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            state_dict = (
                checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
            )
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(state_dict, strict=False)
        else:
            raise ValueError(f"❌ Modello non trovato: {pretrained_path}")

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(0)

        x = self.backbone.patch_embed(x)
        pos_spatial = self.backbone.pos_embed_spatial
        pos_temporal = self.backbone.pos_embed_temporal

        B, L, C = x.shape
        num_spatial_tokens = pos_spatial.shape[1]
        side_dim = int(math.sqrt(num_spatial_tokens))

        if side_dim == 7:
            pos_spatial = pos_spatial.reshape(1, 1, 7, 7, C)
            pos_spatial = pos_spatial.repeat_interleave(8, dim=2).repeat_interleave(
                8, dim=3
            )
        elif side_dim == 56:
            pos_spatial = pos_spatial.reshape(1, 1, 56, 56, C)

        pos_temporal = pos_temporal.reshape(1, 64, 1, 1, C)
        pos_total = pos_spatial + pos_temporal
        pos_total = pos_total.flatten(1, 3)

        x = x + pos_total
        for block in self.backbone.blocks:
            x = block(x)
        x = self.backbone.norm(x)
        return x


# =========================
# Feature Extraction (WITH RESUME)
# =========================
def extract_features(args):
    print("=" * 70)
    print("🚀 SIGNHIERA EXTRACTION (BATCH + RESUME MODE)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Caricamento Manifest
    full_manifest = pd.read_csv(args.manifest, sep="\t")
    total_videos = len(full_manifest)

    # 2. LOGICA RESUME: Controllo file esistenti
    existing_files = {f.stem for f in out_dir.glob("*.npy")}

    # Filtriamo il dataframe: teniamo solo ID che NON sono nei file esistenti
    df_todo = full_manifest[~full_manifest["id"].isin(existing_files)].copy()

    skipped_count = total_videos - len(df_todo)
    print(
        f"📊 Total: {total_videos} | Already Done: {skipped_count} | To Do: {len(df_todo)}"
    )

    if len(df_todo) == 0:
        print("✅ Tutti i video sono già stati processati! Esco.")
        return

    # 3. Setup Modello
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    try:
        model = RealSignHiera(pretrained_path=args.model_path).to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Errore Setup Modello: {e}")
        return

    # 4. DataLoader
    dataset = VideoDataset(df_todo, args.video_dir, target_frames=128)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_skip_none,
        pin_memory=True if args.device == "cuda" else False,
    )

    num_processed = 0

    with torch.no_grad():
        # tqdm mostrerà solo i file rimanenti
        for batch in tqdm(dataloader, desc="Extracting Missing"):
            if batch is None:
                continue

            video_ids = batch["video_ids"]
            frames = batch["frames"].to(device, non_blocking=True)

            try:
                features_batch = model(frames)
                features_batch = features_batch.cpu().numpy()

                for i, vid_id in enumerate(video_ids):
                    feat = features_batch[i]
                    if len(feat.shape) == 2 and feat.shape[0] == 1:
                        feat = feat.squeeze(0)

                    np.save(out_dir / f"{vid_id}.npy", feat)
                    num_processed += 1

            except Exception as e:
                print(f"⚠️ Error processing batch: {e}")

    print(f"✅ Finito! Processati {num_processed} nuovi video.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument(
        "--model_path", type=str, default="models/dm_70h_ub_signhiera.pth"
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_frames", type=int, default=300)
    # Parametri ottimizzazione
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    extract_features(args)


if __name__ == "__main__":
    main()
