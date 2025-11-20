#!/usr/bin/env python3
"""
SSVP Replication Script (Transformer-based Alignment)
=====================================================

DESCRIZIONE:
Questo script tenta di replicare l'architettura del paper SSVP-SLT (Self-Supervised Video Pretraining).
A differenza del semplice MLP, qui usiamo un Transformer Encoder completo per modellare
la dipendenza temporale tra i frame PRIMA di proiettarli nello spazio SONAR.

ARCHITETTURA (SSVP-Style):
--------------------------
1.  **Input Projection**: Linear(768 -> 512)
2.  **Positional Encoding**: Aggiunge info temporale ai frame.
3.  **Transformer Encoder**: 4 Layers, 8 Heads, 512 Hidden.
    -   Permette ai frame di "guardarsi" a vicenda.
    -   Capisce che "mano su" -> "mano giù" è diverso da "mano giù" -> "mano su".
4.  **Pooling**: Media temporale delle feature CONTESTUALIZZATE.
5.  **Output Projection**: Linear(512 -> 1024) -> Spazio SONAR.

Autore: GitHub Copilot & Ignazio Emanuele Piccichè
Data: 20 Novembre 2025
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sacrebleu

# SONAR package
try:
    from sonar.inference_pipelines.text import (
        TextToEmbeddingModelPipeline,
        EmbeddingToTextModelPipeline,
    )

    SONAR_AVAILABLE = True
except ImportError:
    SONAR_AVAILABLE = False
    print("❌ SONAR not found. Install 'sonar-space'.")


# ==============================================================================
# 1. DATASET & UTILS (Identico a prima)
# ==============================================================================


class How2SignDataset(Dataset):
    def __init__(self, features_dir: str, manifest_path: str, max_samples: int = None):
        self.features_dir = Path(features_dir)
        manifest = pd.read_csv(manifest_path, sep="\t")

        valid_samples = []
        for idx, row in manifest.iterrows():
            video_id = row["id"]
            if (self.features_dir / f"{video_id}.npy").exists() or (
                self.features_dir / f"{video_id}.pt"
            ).exists():
                valid_samples.append(row)

        self.manifest = pd.DataFrame(valid_samples)
        if max_samples:
            self.manifest = self.manifest.head(max_samples)

        print(f"📂 Dataset loaded: {len(self.manifest)} samples")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        video_id = row["id"]
        text = row["text"]

        # Load features
        npy_path = self.features_dir / f"{video_id}.npy"
        if npy_path.exists():
            features = torch.from_numpy(np.load(npy_path)).float()
        else:
            pt_path = self.features_dir / f"{video_id}.pt"
            features = torch.load(pt_path, map_location="cpu")["features"]

        return {"video_id": video_id, "features": features, "text": text}


def collate_fn(batch):
    video_ids = [item["video_id"] for item in batch]
    texts = [item["text"] for item in batch]

    # Padding
    max_len = max(item["features"].shape[0] for item in batch)
    feature_dim = batch[0]["features"].shape[1]

    padded_features = torch.zeros(len(batch), max_len, feature_dim)
    lengths = []

    for i, item in enumerate(batch):
        feat = item["features"]
        length = feat.shape[0]
        padded_features[i, :length, :] = feat
        lengths.append(length)

    return {
        "video_ids": video_ids,
        "features": padded_features,
        "lengths": torch.tensor(lengths),
        "texts": texts,
    }


# ==============================================================================
# 2. MODEL ARCHITECTURE (SSVP REPLICATION)
# ==============================================================================


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.pe[: x.size(1), :].unsqueeze(0)
        return x


class SSVPReplicator(nn.Module):
    def __init__(self, input_dim=768, device="cuda"):
        super().__init__()
        self.device = device

        # --- SSVP Encoder Architecture ---
        hidden_dim = 512
        nhead = 8
        num_layers = 4
        output_dim = 1024  # SONAR space

        print(f"🏗️  Building SSVP-style Transformer Encoder...")
        print(f"   Input: {input_dim} -> Hidden: {hidden_dim} -> Layers: {num_layers}")

        # 1. Input Projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(hidden_dim)

        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Output Projection to SONAR
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Tanh(),  # Helps keep values in check
        )

        # --- SONAR Components (Frozen) ---
        print("📥 Loading SONAR models...")
        self.text_embedder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=torch.device(device),
        )
        self.text_decoder = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_encoder",
            device=torch.device(device),
        )

        self.to(device)

    def forward(self, features, lengths=None):
        """
        Video Features -> Transformer -> Pooling -> SONAR Embedding
        """
        B, T, D = features.shape

        # 1. Project & Add Position
        x = self.input_proj(features)  # (B, T, 512)
        x = self.pos_encoder(x)

        # 2. Create Padding Mask
        # Transformer needs to know which tokens are padding
        # src_key_padding_mask: (B, T) True where padding
        src_key_padding_mask = None
        if lengths is not None:
            src_key_padding_mask = torch.zeros(
                B, T, dtype=torch.bool, device=self.device
            )
            for i in range(B):
                l = lengths[i].item()
                if l < T:
                    src_key_padding_mask[i, l:] = True

        # 3. Transformer Pass
        # (B, T, 512) -> (B, T, 512)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # 4. Pooling (Mean over valid tokens)
        if lengths is not None:
            mask = (~src_key_padding_mask).float().unsqueeze(-1)  # (B, T, 1)
            sum_features = (x * mask).sum(dim=1)
            count_features = mask.sum(dim=1).clamp(min=1.0)
            x_pooled = sum_features / count_features
        else:
            x_pooled = x.mean(dim=1)

        # 5. Project to SONAR
        embeddings = self.output_proj(x_pooled)  # (B, 1024)

        return embeddings

    def encode_texts(self, texts):
        with torch.no_grad():
            embeddings = self.text_embedder.predict(texts, source_lang="eng_Latn")
            if isinstance(embeddings, np.ndarray):
                embeddings = torch.from_numpy(embeddings).float()
            return embeddings.to(self.device)

    def decode(self, embeddings):
        with torch.no_grad():
            if embeddings.is_cuda:
                embeddings = embeddings.cpu()
            return self.text_decoder.predict(
                embeddings, target_lang="eng_Latn", max_seq_len=512
            )


# ==============================================================================
# 3. TRAINING LOOP
# ==============================================================================


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    progress = tqdm(dataloader, desc="Training")
    for batch in progress:
        features = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        texts = batch["texts"]

        # Forward
        pred_embeddings = model(features, lengths)
        target_embeddings = model.encode_texts(texts)

        # Loss
        # Cosine (Direction)
        pred_norm = torch.nn.functional.normalize(pred_embeddings, p=2, dim=1)
        target_norm = torch.nn.functional.normalize(target_embeddings, p=2, dim=1)
        loss_cosine = 1.0 - (pred_norm * target_norm).sum(dim=1).mean()

        # Magnitude (Scale)
        loss_mag = torch.nn.functional.mse_loss(
            pred_embeddings.norm(p=2, dim=1), target_embeddings.norm(p=2, dim=1)
        )

        loss = loss_cosine + 1.0 * loss_mag

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix(
            {"loss": f"{loss.item():.4f}", "cos": f"{loss_cosine.item():.3f}"}
        )

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    predictions, references = [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        features = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        texts = batch["texts"]

        embeddings = model(features, lengths)
        preds = model.decode(embeddings)

        predictions.extend(preds)
        references.extend(texts)

    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score, predictions[:5], references[:5]


# ==============================================================================
# 4. MAIN
# ==============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features", required=True)
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_features", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--output_dir", default="checkpoints_ssvp")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    train_ds = How2SignDataset(args.train_features, args.train_manifest)
    val_ds = How2SignDataset(args.val_features, args.val_manifest)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    # Model
    model = SSVPReplicator(input_dim=768, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_bleu = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Train Loss: {loss:.4f}")

        if epoch % 5 == 0:
            bleu, preds, refs = evaluate(model, val_loader, device)
            print(f"Val BLEU: {bleu:.2f}")
            print(f"Sample: {preds[0]} (Ref: {refs[0]})")

            if bleu > best_bleu:
                best_bleu = bleu
                torch.save(model.state_dict(), f"{args.output_dir}/best_model.pt")

    print(f"Done! Best BLEU: {best_bleu:.2f}")


if __name__ == "__main__":
    main()
