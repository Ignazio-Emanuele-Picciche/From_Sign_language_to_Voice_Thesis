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
import math  # Aggiunto per calcolo radice quadrata

import warnings

warnings.filterwarnings("ignore")

# --- Importiamo la versione VIDEO di Hiera ---
try:
    from hiera import hiera_base_16x224
except ImportError:
    raise ImportError("Installare la libreria: pip install hiera-transformer")


# ===========
# Dataset (Invariato - 128 Frames)
# ===========
class VideoDataset(Dataset):
    def __init__(
        self, manifest_path, video_dir, target_size=(224, 224), target_frames=128
    ):
        self.video_dir = Path(video_dir)
        self.target_size = target_size
        self.target_frames = target_frames
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
            frames = torch.zeros(self.target_frames, 3, *self.target_size)
            valid = False
        else:
            valid = True
        return {"video_id": video_id, "frames": frames, "valid": valid}

    def _load_video(self, video_path):
        if not video_path.exists():
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
        frames = np.transpose(frames, (0, 3, 1, 2))
        return torch.from_numpy(frames).float()


# =========================
# MODEL: REAL SignHiera (ADAPTIVE FORWARD)
# =========================
class RealSignHiera(nn.Module):
    def __init__(self, pretrained_path=None):
        super().__init__()
        print(f"🏗️  Building SignHiera architecture...")

        self.backbone = hiera_base_16x224(pretrained=False)
        self.backbone.head = nn.Identity()
        self.feature_dim = 768

        # --- FIX: Allarghiamo i pesi TEMPORALI ---
        # SignHiera usa 64 token temporali (per 128 frame input)
        expected_temp_pos = 64
        current_temp_pos = self.backbone.pos_embed_temporal.shape[1]

        if current_temp_pos != expected_temp_pos:
            print(
                f"🔧 Resizing temporal parameter from {current_temp_pos} to {expected_temp_pos}..."
            )
            new_pos_embed = nn.Parameter(torch.zeros(1, expected_temp_pos, 96))
            self.backbone.pos_embed_temporal = new_pos_embed
            self.backbone.tokens_temporal = expected_temp_pos

        # Caricamento Pesi
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"📂 Loading weights from {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            state_dict = (
                checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
            )
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            # Carichiamo tollerando i mismatch (perché pos_spatial potrebbe essere diverso dallo standard)
            missing, unexpected = self.backbone.load_state_dict(
                state_dict, strict=False
            )

            # DEBUG: Vediamo quanto è grande davvero pos_spatial dopo il caricamento
            spatial_shape = self.backbone.pos_embed_spatial.shape
            print(f"📊 DEBUG: Loaded Spatial Embed Shape: {spatial_shape}")
            print("✅ Pesi SignHiera caricati!")
        else:
            raise ValueError(f"❌ Modello non trovato: {pretrained_path}")

    def forward(self, x):
        # x: (T, C, H, W) -> (128, 3, 224, 224)
        x = x.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, 3, 128, 224, 224)

        # === MANUAL FORWARD PASS (ADAPTIVE) ===

        # 1. Patch Embedding -> (B, 200704, 96)
        x = self.backbone.patch_embed(x)

        # 2. Add Position Embeddings (ADAPTIVE LOGIC)
        # Recuperiamo i pesi attuali dal modello
        pos_spatial = (
            self.backbone.pos_embed_spatial
        )  # Shape incognita (o 1x49x96 o 1x3136x96)
        pos_temporal = self.backbone.pos_embed_temporal  # Shape (1, 64, 96)

        B, L, C = x.shape
        # L atteso = 200704 (64 * 56 * 56)

        # --- GESTIONE SPAZIALE ---
        num_spatial_tokens = pos_spatial.shape[1]  # es. 3136 o 49
        side_dim = int(math.sqrt(num_spatial_tokens))  # es. 56 o 7

        if side_dim == 7:
            # Caso Standard Hiera: i pesi sono piccoli, dobbiamo espanderli
            pos_spatial = pos_spatial.reshape(1, 1, 7, 7, C)
            # Ripetiamo 8 volte per arrivare a 56x56
            pos_spatial = pos_spatial.repeat_interleave(8, dim=2).repeat_interleave(
                8, dim=3
            )
        elif side_dim == 56:
            # Caso SignHiera Checkpoint: i pesi sono GIÀ espansi
            pos_spatial = pos_spatial.reshape(1, 1, 56, 56, C)
        else:
            raise ValueError(
                f"Dimensione spaziale imprevista nei pesi: {side_dim}x{side_dim}"
            )

        # Ora pos_spatial è sicuramente (1, 1, 56, 56, 96)

        # --- GESTIONE TEMPORALE ---
        # pos_temporal è (1, 64, 96) -> reshape a (1, 64, 1, 1, 96)
        pos_temporal = pos_temporal.reshape(1, 64, 1, 1, C)

        # --- SOMMA BROADCASTING ---
        # (1, 1, 56, 56, 96) + (1, 64, 1, 1, 96) = (1, 64, 56, 56, 96)
        pos_total = pos_spatial + pos_temporal

        # Flatten finale: (1, 200704, 96)
        pos_total = pos_total.flatten(1, 3)

        # Check sicurezza finale
        if pos_total.shape[1] != x.shape[1]:
            # Se capita questo, tronchiamo o paddiamo (fallback estremo)
            # Ma matematicamente non dovrebbe succedere ora.
            print(f"⚠️ Shape mismatch finale: Embed {pos_total.shape} vs X {x.shape}")

        # Addizione
        x = x + pos_total

        # 3. Passaggio nei Blocchi
        for block in self.backbone.blocks:
            x = block(x)

        # 4. Normale finale
        x = self.backbone.norm(x)

        return x


# =========================
# Feature Extraction
# =========================
def extract_features(args):
    print("=" * 70)
    print("🚀 SIGNHIERA EXTRACTION (ADAPTIVE)")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    try:
        model = RealSignHiera(pretrained_path=args.model_path).to(device)
        model.eval()
    except Exception as e:
        print(f"❌ Errore Setup Modello: {e}")
        return

    dataset = VideoDataset(args.manifest, args.video_dir, target_frames=128)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    num_processed = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            sample = batch[0]
            video_id = sample["video_id"]
            frames = sample["frames"]

            if not sample["valid"]:
                continue

            frames = frames.to(device)
            try:
                features = model(frames)
                features = features.cpu().numpy()
                if features.shape[0] == 1:
                    features = features.squeeze(0)

                np.save(out_dir / f"{video_id}.npy", features)
                num_processed += 1
            except Exception as e:
                print(f"⚠️ Error {video_id}: {e}")
                import traceback

                traceback.print_exc()
                if num_processed == 0:
                    break

    print(f"✅ Finito! Salvati {num_processed} video in {out_dir}")


# =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument(
        "--model_path", type=str, default="models/dm_70h_ub_signhiera.pth"
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    extract_features(args)


if __name__ == "__main__":
    main()
