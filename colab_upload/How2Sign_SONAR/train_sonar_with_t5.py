#!/usr/bin/env python3
"""
SONAR Encoder (Fine-Tuned) + T5 Decoder Training
=================================================

Questo script integra l'encoder SONAR fine-tuned (già trainato) con un
decoder T5 pre-trained per traduzione ASL→English.

Pipeline:
1. Carica encoder SONAR fine-tuned (FROZEN)
2. Aggiungi projection layer (1024→512 dims)
3. Carica T5-small decoder (pre-trained)
4. Training: Solo projection + T5 decoder
5. Evaluation: BLEU score

Expected BLEU: 18-25% (vs 0.01% con LSTM)

Autore: GitHub Copilot + Ignazio Picciche
Data: Novembre 2024
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sacrebleu

# Transformers per T5
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)

print("✅ All imports successful!")


# ============================================================================
# Dataset (Same as train_sonar_encoder_only.py)
# ============================================================================


class How2SignDataset(Dataset):
    """Dataset per feature How2Sign + traduzioni"""

    def __init__(self, features_dir: str, manifest_path: str, max_samples: int = None):
        self.features_dir = Path(features_dir)

        # Carica manifest con traduzioni
        manifest_full = pd.read_csv(manifest_path, sep="\t")

        print(f"   Total samples in manifest: {len(manifest_full)}")

        # Identifica colonne
        self.id_col = None
        self.text_col = None

        for col in manifest_full.columns:
            col_lower = col.lower()
            if "id" in col_lower or "name" in col_lower:
                self.id_col = col
            if (
                "text" in col_lower
                or "translation" in col_lower
                or "sentence" in col_lower
            ):
                self.text_col = col

        if self.id_col is None:
            self.id_col = manifest_full.columns[0]
        if self.text_col is None:
            self.text_col = manifest_full.columns[-1]

        print(f"   Using ID column: '{self.id_col}'")
        print(f"   Using Text column: '{self.text_col}'")

        # Filtra solo video con feature disponibili
        available_ids = []
        for idx, row in manifest_full.iterrows():
            video_id = str(row[self.id_col])
            npy_path = self.features_dir / f"{video_id}.npy"
            pt_path = self.features_dir / f"{video_id}.pt"

            if npy_path.exists() or pt_path.exists():
                available_ids.append(idx)

        self.manifest = manifest_full.iloc[available_ids].reset_index(drop=True)

        print(f"   ✅ Found {len(self.manifest)} samples with features")

        if max_samples:
            self.manifest = self.manifest.head(max_samples)
            print(f"   Limited to {max_samples} samples")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        video_id = str(row[self.id_col])
        text = str(row[self.text_col])

        # Carica feature
        npy_path = self.features_dir / f"{video_id}.npy"
        pt_path = self.features_dir / f"{video_id}.pt"

        if npy_path.exists():
            features = np.load(npy_path)
            features = torch.from_numpy(features).float()
        elif pt_path.exists():
            features = torch.load(pt_path)
        else:
            raise FileNotFoundError(f"Feature not found: {video_id}")

        return {"video_id": video_id, "features": features, "text": text}


def collate_fn(batch, max_frames=300):
    """
    Collate function per batch variable-length
    Padding/truncation a max_frames
    """
    video_ids = [item["video_id"] for item in batch]
    texts = [item["text"] for item in batch]

    # Pad/truncate features
    features_list = []
    for item in batch:
        feat = item["features"]

        # feat può essere (T, D) o (D,)
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)  # (1, D)

        T, D = feat.shape

        if T > max_frames:
            # Truncate
            feat = feat[:max_frames]
        elif T < max_frames:
            # Pad
            padding = torch.zeros(max_frames - T, D)
            feat = torch.cat([feat, padding], dim=0)

        features_list.append(feat)

    # Stack
    features = torch.stack(features_list)  # (B, T, D)

    return {"video_ids": video_ids, "features": features, "texts": texts}


# ============================================================================
# SONAR Encoder (Same as train_sonar_encoder_only.py)
# ============================================================================


class SONAREncoder(nn.Module):
    """
    SONAR Encoder compatible with train_sonar_encoder_only.py checkpoint

    Nota: Usa stessa architettura del training originale:
    - self.projection (non self.encoder)
    - self.norm (separato)

    Input: (batch, seq_len, 256) - features da SignHiera
    Output: (batch, 1024) - SONAR embedding space
    """

    def __init__(self, input_dim=256, hidden_dim=512, output_dim=1024):
        super().__init__()

        # IMPORTANTE: Stessa architettura di train_sonar_encoder_only.py!
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

        # Layer norm separato (non dentro Sequential)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, features):
        """
        Args:
            features: (batch, seq_len, input_dim) - features video
        Returns:
            embeddings: (batch, output_dim) - sentence embedding
        """
        # Mean pooling lungo la dimensione temporale
        features_avg = features.mean(dim=1)  # (batch, input_dim)

        # Projection
        embeddings = self.projection(features_avg)  # (batch, output_dim)
        embeddings = self.norm(embeddings)

        # L2 normalization (come SONAR)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


# ============================================================================
# SONAR + T5 Model
# ============================================================================


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler (inspired by Flamingo, Alayrac et al. 2022)

    Converts a single SONAR embedding (1024-dim) into multiple latent tokens (N x dim)
    using learnable query tokens and cross-attention.

    This is more powerful than simple projection because:
    1. Learnable queries can extract different aspects of SONAR embedding
    2. Cross-attention allows adaptive weighting
    3. MLP adds non-linearity for better transformation

    Architecture:
        SONAR embedding (B, 1024)
            ↓
        [Project to hidden_dim]
            ↓
        Key/Value (B, 1, hidden_dim)
            ↓
        Query = Learnable latents (num_latents, hidden_dim)
            ↓
        [Cross-Attention] Query attends to Key/Value
            ↓
        [MLP] Process attended output
            ↓
        Output (B, num_latents, output_dim)
    """

    def __init__(
        self,
        input_dim: int = 1024,  # SONAR embedding dim
        hidden_dim: int = 768,  # Internal processing dim
        output_dim: int = 512,  # T5 hidden dim
        num_latents: int = 64,  # Number of output tokens
        num_heads: int = 8,  # Attention heads
        num_layers: int = 2,  # Depth (stacked resampler layers)
    ):
        super().__init__()

        self.num_latents = num_latents

        # Learnable query tokens (these will attend to SONAR embedding)
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_dim))

        # Project SONAR embedding to hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Cross-attention layers (query=latents, key/value=SONAR)
        self.cross_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=0.1,
                    batch_first=False,  # (seq, batch, dim)
                )
                for _ in range(num_layers)
            ]
        )

        # Layer norms
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # MLPs for processing after attention
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Dropout(0.1),
                )
                for _ in range(num_layers)
            ]
        )

        # Final projection to T5 space
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(0.1),
        )

        # Initialize latents with Xavier
        nn.init.xavier_uniform_(self.latents)

    def forward(self, sonar_embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sonar_embedding: (B, input_dim) - SONAR sentence embedding

        Returns:
            output: (B, num_latents, output_dim) - Tokens for T5
        """
        batch_size = sonar_embedding.size(0)

        # 1. Project SONAR to hidden dim
        sonar_hidden = self.input_proj(sonar_embedding)  # (B, hidden_dim)

        # 2. Prepare for attention: (seq, batch, dim)
        key_value = sonar_hidden.unsqueeze(0)  # (1, B, hidden_dim)

        # 3. Expand learnable latents for batch
        query = self.latents.unsqueeze(1).expand(
            -1, batch_size, -1
        )  # (num_latents, B, hidden_dim)

        # 4. Apply stacked cross-attention + MLP layers
        output = query
        for cross_attn, ln, mlp in zip(
            self.cross_attentions, self.layer_norms, self.mlps
        ):
            # Cross-attention: latents attend to SONAR
            attn_output, _ = cross_attn(
                query=output, key=key_value, value=key_value
            )  # (num_latents, B, hidden_dim)

            # Residual + LayerNorm
            output = ln(output + attn_output)

            # MLP with residual
            output = output + mlp(output)

        # 5. Transpose to (B, num_latents, hidden_dim)
        output = output.transpose(0, 1)  # (B, num_latents, hidden_dim)

        # 6. Final projection to T5 space
        output = self.output_proj(output)  # (B, num_latents, output_dim)

        return output


class SONARwithT5(nn.Module):
    """
    Integra SONAR Encoder fine-tuned con T5 Decoder

    SONAR Encoder può essere:
    - FROZEN: Usa pesi pre-trained fissi (preserva specializzazione ASL)
    - TRAINABLE: Fine-tuna ulteriormente con differential LR (migliora adaptation a T5)

    Architecture:
    Video Features → SONAR Encoder (frozen/trainable) → Embedding 1024 →
    Perceiver Resampler → (B, 64, 512) → T5 Decoder → Text

    Uses Perceiver Resampler (Flamingo-style) instead of simple projection
    for better adaptation of SONAR embeddings to T5 decoder.
    """

    def __init__(
        self,
        sonar_checkpoint: str,
        t5_model_name: str = "t5-small",
        freeze_encoder: bool = True,
        device: str = "cuda",
    ):
        super().__init__()

        self.device = device
        self.freeze_encoder = freeze_encoder  # Salva il flag per forward pass

        # 1. Load SONAR Encoder (fine-tuned)
        print(f"\n📦 Loading SONAR Encoder from: {sonar_checkpoint}")
        self.sonar_encoder = SONAREncoder(
            input_dim=256, hidden_dim=512, output_dim=1024
        )

        checkpoint = torch.load(sonar_checkpoint, map_location=device)

        # Handle different checkpoint formats
        if "encoder_state_dict" in checkpoint:
            # Format: {"encoder_state_dict": ..., "decoder_state_dict": ..., ...}
            encoder_state = checkpoint["encoder_state_dict"]
            print("   Format: Full checkpoint with encoder_state_dict")
        elif "model_state_dict" in checkpoint:
            # Format: {"model_state_dict": {...}}
            encoder_state = checkpoint["model_state_dict"]
            print("   Format: Full checkpoint with model_state_dict")
        else:
            # Format: Direct state_dict
            encoder_state = checkpoint
            print("   Format: Direct state_dict")

        try:
            self.sonar_encoder.load_state_dict(encoder_state, strict=True)
            print("   ✅ SONAR Encoder loaded successfully")
        except RuntimeError as e:
            print(f"   ⚠️  Warning during strict loading:")
            print(f"      {str(e)[:200]}...")
            print("   Trying to load with strict=False...")
            self.sonar_encoder.load_state_dict(encoder_state, strict=False)
            print("   ✅ SONAR Encoder loaded (some weights may be missing/extra)")

        if freeze_encoder:
            # Freeze encoder (usa pesi già trainati)
            for param in self.sonar_encoder.parameters():
                param.requires_grad = False
            print("   ❄️  SONAR Encoder FROZEN (using fine-tuned weights)")
        else:
            print("   🔥 SONAR Encoder TRAINABLE (will be fine-tuned further)")

        # 2. Perceiver Resampler: 1024-dim SONAR → 64 tokens × 512-dim T5
        # REPLACEMENT: Più potente di Projection + Attention Bridge!
        self.perceiver = PerceiverResampler(
            input_dim=1024,  # SONAR output
            hidden_dim=768,  # Internal processing
            output_dim=512,  # T5 hidden size
            num_latents=64,  # Output tokens (increased from 32!)
            num_heads=8,  # Attention heads
            num_layers=2,  # Depth (stacked resampler)
        )
        print("   ✅ Perceiver Resampler created (1024 → 64×512, 2-layer)")
        print("      Flamingo-style adapter with learnable query tokens")

        # 3. Load T5 Decoder
        print(f"\n📦 Loading T5 model: {t5_model_name}")
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
        print(
            f"   ✅ T5 loaded ({sum(p.numel() for p in self.t5.parameters()):,} params)"
        )

        # Move to device
        self.to(device)

        # Conta parametri trainable
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"\n📊 Model Summary:")
        print(f"   Total params: {total:,}")
        print(f"   Trainable params: {trainable:,}")
        print(f"   Frozen params: {total - trainable:,}")

    def forward(
        self,
        features: torch.Tensor,
        target_texts: Optional[List[str]] = None,
        max_length: int = 50,
    ):
        """
        Forward pass

        Args:
            features: (B, T, 256) - SignHiera features
            target_texts: List[str] - target texts (training only)
            max_length: int - max generation length

        Returns:
            If training: loss
            If inference: List[str] - generated texts
        """
        batch_size = features.size(0)

        # 1. Encode con SONAR (frozen o trainable)
        if self.freeze_encoder:
            # Encoder frozen: no gradient
            with torch.no_grad():
                sonar_embedding = self.sonar_encoder(features)  # (B, 1024)
        else:
            # Encoder trainable: compute gradient
            sonar_embedding = self.sonar_encoder(features)  # (B, 1024)

        # 2. Perceiver Resampler: Convert SONAR embedding to T5 tokens
        # This replaces: Projection → Expander → Attention Bridge
        # With: Learnable cross-attention resampler (Flamingo-style)
        t5_input_tokens = self.perceiver(sonar_embedding)  # (B, 64, 512)

        if target_texts is not None:
            # ===== TRAINING MODE =====

            # Tokenize targets
            target_encoding = self.tokenizer(
                target_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            target_ids = target_encoding.input_ids.to(self.device)

            # T5 Decoder (NO encoder, Flamingo-style)
            # inputs_embeds goes directly to decoder
            outputs = self.t5(
                inputs_embeds=t5_input_tokens,  # (B, 64, 512) from Perceiver
                labels=target_ids,
                return_dict=True,
            )

            return outputs.loss

        else:
            # ===== INFERENCE MODE =====

            # Generate with diversity mechanisms to prevent mode collapse
            generated_ids = self.t5.generate(
                inputs_embeds=t5_input_tokens,  # (B, 64, 512)
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=2,
                repetition_penalty=1.5,  # Penalize repeated tokens
                temperature=0.8,  # Add randomness
                top_p=0.9,  # Nucleus sampling
                do_sample=True,  # Enable sampling
            )

            # Decode
            generated_texts = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            return generated_texts


# ============================================================================
# Training
# ============================================================================


def train_epoch(
    model: SONARwithT5,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    device: str,
):
    """Train one epoch"""
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(progress_bar):
        features = batch["features"].to(device)
        texts = batch["texts"]

        # Forward
        loss = model(features, target_texts=texts)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler:
            scheduler.step()

        # Stats
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)

        progress_bar.set_postfix(
            {"loss": f"{loss.item():.4f}", "avg_loss": f"{avg_loss:.4f}"}
        )

    return total_loss / len(train_loader)


def evaluate(model: SONARwithT5, val_loader: DataLoader, device: str):
    """Evaluate on validation set"""
    model.eval()

    all_predictions = []
    all_references = []

    print("\n🔍 Evaluating...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            features = batch["features"].to(device)
            texts = batch["texts"]

            # Generate
            predictions = model(features)

            all_predictions.extend(predictions)
            all_references.extend(texts)

    # Calculate BLEU
    bleu = sacrebleu.corpus_bleu(all_predictions, [all_references])

    # Show examples
    print("\n📊 Sample Translations:")
    for i in range(min(5, len(all_predictions))):
        print(f"\n  Sample {i+1}:")
        print(f"    GT:   {all_references[i]}")
        print(f"    Pred: {all_predictions[i]}")

    return bleu.score, all_predictions, all_references


# ============================================================================
# Main Training Loop
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Train SONAR+T5 for ASL→English")

    # Paths
    parser.add_argument(
        "--sonar_checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned SONAR encoder",
    )
    parser.add_argument(
        "--train_features",
        type=str,
        required=True,
        help="Directory with training features",
    )
    parser.add_argument(
        "--train_manifest", type=str, required=True, help="Training manifest TSV"
    )
    parser.add_argument(
        "--val_features",
        type=str,
        required=True,
        help="Directory with validation features",
    )
    parser.add_argument(
        "--val_manifest", type=str, required=True, help="Validation manifest TSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/sonar_t5",
        help="Output directory for checkpoints",
    )

    # Model
    parser.add_argument(
        "--t5_model",
        type=str,
        default="t5-small",
        choices=["t5-small", "t5-base", "t5-large"],
        help="T5 model size",
    )
    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        default=True,
        help="Freeze SONAR encoder (recommended)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=500, help="Warmup steps for scheduler"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None, help="Max samples (for testing)"
    )

    # Hardware
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device"
    )

    args = parser.parse_args()

    # Setup
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Starting SONAR+T5 Training")
    print(f"   Device: {device}")
    print(f"   SONAR checkpoint: {args.sonar_checkpoint}")
    print(f"   T5 model: {args.t5_model}")
    print(f"   Freeze encoder: {args.freeze_encoder}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ===== LOAD DATA =====
    print("\n📂 Loading datasets...")

    print("\n  Training set:")
    train_dataset = How2SignDataset(
        features_dir=args.train_features,
        manifest_path=args.train_manifest,
        max_samples=args.max_samples,
    )

    print("\n  Validation set:")
    val_dataset = How2SignDataset(
        features_dir=args.val_features,
        manifest_path=args.val_manifest,
        max_samples=args.max_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print(f"\n✅ Data loaded:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Batches per epoch: {len(train_loader)}")

    # ===== CREATE MODEL =====
    model = SONARwithT5(
        sonar_checkpoint=args.sonar_checkpoint,
        t5_model_name=args.t5_model,
        freeze_encoder=args.freeze_encoder,
        device=device,
    )

    # ===== OPTIMIZER & SCHEDULER =====
    # Differential learning rates per evitare di "rovinare" il pre-training SONAR

    if args.freeze_encoder:
        # Encoder frozen: ottimizza solo projection + T5
        optimizer_params = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": args.learning_rate,
            }
        ]
        print(f"\n🔧 Optimizer: Single LR (encoder frozen)")
    else:
        # Encoder unfrozen: usa differential learning rates
        encoder_params = []
        perceiver_params = []
        t5_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            if "sonar_encoder" in name:
                encoder_params.append(param)
            elif "perceiver" in name:
                # Perceiver Resampler parameters (replaces projection + attention)
                perceiver_params.append(param)
            elif "t5" in name:
                t5_params.append(param)

        # SONAR Encoder: LR molto basso (1/5 del normale) per preservare pre-training
        # Perceiver: LR normale (learnable adapter)
        # T5: LR normale
        optimizer_params = [
            {
                "params": encoder_params,
                "lr": args.learning_rate / 5,
                "name": "sonar_encoder",
            },
            {
                "params": perceiver_params,
                "lr": args.learning_rate,
                "name": "perceiver",
            },
            {"params": t5_params, "lr": args.learning_rate, "name": "t5"},
        ]

        print(f"\n🔧 Optimizer: Differential Learning Rates")
        print(
            f"   SONAR Encoder LR: {args.learning_rate / 5:.2e} (1/5x, preserve pre-training)"
        )
        print(
            f"   Perceiver Resampler LR: {args.learning_rate:.2e} (1x, learnable adapter)"
        )
        print(f"   T5 LR: {args.learning_rate:.2e} (1x)")
        print(f"   Trainable params:")
        print(f"     - SONAR Encoder: {len(encoder_params)}")
        print(f"     - Perceiver Resampler: {len(perceiver_params)}")
        print(f"     - T5: {len(t5_params)}")

    optimizer = torch.optim.AdamW(optimizer_params, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    print(f"\n🎯 Training configuration:")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Total steps: {total_steps}")
    print(f"   Warmup steps: {args.warmup_steps}")

    # ===== TRAINING LOOP =====
    best_bleu = 0.0
    training_log = []

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print(f"{'='*60}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, epoch, device
        )

        # Evaluate
        val_bleu, predictions, references = evaluate(model, val_loader, device)

        # Log
        log_entry = {"epoch": epoch, "train_loss": train_loss, "val_bleu": val_bleu}
        training_log.append(log_entry)

        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val BLEU: {val_bleu:.2f}%")

        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_bleu": val_bleu,
                "args": vars(args),
            },
            checkpoint_path,
        )
        print(f"   💾 Checkpoint saved: {checkpoint_path}")

        # Save best model
        if val_bleu > best_bleu:
            best_bleu = val_bleu
            best_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_bleu": val_bleu,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"   🏆 NEW BEST MODEL! BLEU: {val_bleu:.2f}%")

            # Save predictions
            pred_path = os.path.join(args.output_dir, "best_predictions.txt")
            with open(pred_path, "w") as f:
                for ref, pred in zip(references, predictions):
                    f.write(f"GT:   {ref}\n")
                    f.write(f"Pred: {pred}\n")
                    f.write("\n")

    # ===== TRAINING COMPLETE =====
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\n🏆 Best BLEU: {best_bleu:.2f}%")
    print(f"📁 Checkpoints saved in: {args.output_dir}")

    # Save training log
    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"📊 Training log saved: {log_path}")

    print("\n✅ All done! 🎉")


if __name__ == "__main__":
    main()
