#!/usr/bin/env python3
"""
Fine-tuning SONAR Decoder per How2Sign
Usa le feature estratte da SignHiera per addestrare il decoder
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from sacrebleu.metrics import BLEU


# ============================================================================
# Dataset
# ============================================================================


class How2SignFeatureDataset(Dataset):
    """Dataset che carica feature pre-estratte"""

    def __init__(self, features_dir, manifest_path, max_length=300):
        self.features_dir = Path(features_dir)
        self.max_length = max_length

        # Carica manifest
        self.manifest = pd.read_csv(manifest_path, sep="\t")

        # Filtra solo video con feature disponibili
        # Supporta sia .pt che .npy
        self.samples = []
        for idx, row in self.manifest.iterrows():
            # Prova prima .pt, poi .npy
            feature_path_pt = self.features_dir / f"{row['id']}.pt"
            feature_path_npy = self.features_dir / f"{row['id']}.npy"

            if feature_path_pt.exists():
                feature_path = feature_path_pt
            elif feature_path_npy.exists():
                feature_path = feature_path_npy
            else:
                continue

            self.samples.append(
                {"id": row["id"], "text": row["text"], "feature_path": feature_path}
            )

        print(f"✅ Loaded {len(self.samples)}/{len(self.manifest)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Carica feature (.pt o .npy)
        feature_path = sample["feature_path"]

        if feature_path.suffix == ".pt":
            # PyTorch format
            data = torch.load(feature_path, map_location="cpu")
            features = data["features"]  # Shape: (T, 256)
        else:
            # NumPy format (.npy)
            import numpy as np

            features = np.load(feature_path)
            features = torch.from_numpy(features).float()

        # Pad/truncate to max_length
        T, D = features.shape
        if T > self.max_length:
            features = features[: self.max_length]
        elif T < self.max_length:
            padding = torch.zeros(self.max_length - T, D)
            features = torch.cat([features, padding], dim=0)

        return {
            "features": features,
            "text": sample["text"],
            "video_id": sample["id"],
            "seq_length": min(T, self.max_length),
        }


# ============================================================================
# Model: Simple Decoder
# ============================================================================


class SimpleTranslationDecoder(nn.Module):
    """
    Decoder semplice: Feature → Hidden → Text Embedding
    Usa un vocabulary predefinito per generare traduzioni
    """

    def __init__(self, input_dim=256, hidden_dim=512, vocab_size=10000):
        super().__init__()

        # Encoder: feature temporali → rappresentazione fissa
        self.temporal_encoder = nn.LSTM(
            input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True
        )

        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Text decoder head
        self.text_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, seq_lengths):
        """
        Args:
            features: (B, T, 256) - Feature estratte
            seq_lengths: (B,) - Lunghezze reali delle sequenze

        Returns:
            embeddings: (B, hidden_dim) - Rappresentazione per traduzione
        """
        # LSTM encoding
        packed = nn.utils.rnn.pack_padded_sequence(
            features, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.temporal_encoder(packed)

        # Concatena forward e backward dell'ultimo layer
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (B, hidden_dim*2)

        # Project to fixed dimension
        embeddings = self.projection(hidden)  # (B, hidden_dim)

        return embeddings


# ============================================================================
# Tokenizer semplice (per calcolo loss)
# ============================================================================


class SimpleTokenizer:
    """Tokenizer basico per convertire testo in token IDs"""

    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.reverse_vocab = {}

    def build_vocab(self, texts):
        """Costruisce vocabulary da lista di testi"""
        words = set()
        for text in texts:
            words.update(text.lower().split())

        # Crea mappatura word → ID
        self.vocab = {word: idx for idx, word in enumerate(sorted(words), start=4)}
        self.vocab["<PAD>"] = 0
        self.vocab["<UNK>"] = 1
        self.vocab["<BOS>"] = 2
        self.vocab["<EOS>"] = 3

        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        print(f"✅ Vocabulary size: {len(self.vocab)}")

    def encode(self, text, max_length=50):
        """Converte testo in lista di token IDs"""
        words = text.lower().split()
        ids = [self.vocab.get(word, self.vocab["<UNK>"]) for word in words]

        # Pad/truncate
        if len(ids) > max_length:
            ids = ids[:max_length]
        else:
            ids += [self.vocab["<PAD>"]] * (max_length - len(ids))

        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids):
        """Converte token IDs in testo"""
        words = [self.reverse_vocab.get(int(id), "<UNK>") for id in ids]
        # Rimuovi padding e special tokens
        words = [w for w in words if w not in ["<PAD>", "<BOS>", "<EOS>"]]
        return " ".join(words)


# ============================================================================
# Training
# ============================================================================


def train_epoch(model, dataloader, optimizer, criterion, device, tokenizer):
    """Training loop per un epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        features = batch["features"].to(device)
        seq_lengths = batch["seq_length"]
        texts = batch["text"]

        # Encode texts to target IDs
        target_ids = torch.stack([tokenizer.encode(text) for text in texts]).to(device)

        # Forward pass
        embeddings = model(features, seq_lengths)  # (B, hidden_dim)

        # Predict text tokens (semplificato: usa solo primo token come target)
        logits = model.text_head(embeddings)  # (B, vocab_size)

        # Loss: predici prima parola della traduzione
        first_word_ids = target_ids[:, 0]
        loss = criterion(logits, first_word_ids)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, tokenizer):
    """Valutazione con BLEU score"""
    model.eval()

    predictions = []
    references = []

    bleu_scorer = BLEU()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            features = batch["features"].to(device)
            seq_lengths = batch["seq_length"]
            texts = batch["text"]

            # Generate embeddings
            embeddings = model(features, seq_lengths)
            logits = model.text_head(embeddings)

            # Decode (semplificato: genera solo prima parola)
            pred_ids = torch.argmax(logits, dim=-1)
            pred_texts = [tokenizer.decode([id.item()]) for id in pred_ids]

            predictions.extend(pred_texts)
            references.extend(texts)

    # Calcola BLEU
    bleu_score = bleu_scorer.corpus_score(predictions, [references]).score

    return bleu_score, predictions, references


# ============================================================================
# Main
# ============================================================================


def main(args):
    print("=" * 80)
    print("🚀 SONAR Fine-Tuning for How2Sign")
    print("=" * 80)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"📍 Device: {device}\n")

    # Load datasets
    print("📦 Loading datasets...")
    train_dataset = How2SignFeatureDataset(
        args.train_features, args.train_manifest, max_length=args.max_frames
    )

    val_dataset = How2SignFeatureDataset(
        args.val_features, args.val_manifest, max_length=args.max_frames
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Build tokenizer
    print("\n📝 Building vocabulary...")
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
    all_texts = [sample["text"] for sample in train_dataset.samples]
    tokenizer.build_vocab(all_texts)

    # Create model
    print(f"\n🤖 Creating model...")
    model = SimpleTranslationDecoder(
        input_dim=256, hidden_dim=args.hidden_dim, vocab_size=len(tokenizer.vocab)
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Optimizer & loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab["<PAD>"])

    # Training loop
    print(f"\n🏋️ Training for {args.epochs} epochs...\n")

    best_bleu = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\n📍 Epoch {epoch+1}/{args.epochs}")

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, tokenizer
        )
        print(f"   Train Loss: {train_loss:.4f}")

        # Validate every N epochs
        if (epoch + 1) % args.eval_every == 0:
            bleu_score, preds, refs = evaluate(model, val_loader, device, tokenizer)
            print(f"   Val BLEU: {bleu_score:.2f}")

            # Save best model
            if bleu_score > best_bleu:
                best_bleu = bleu_score
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "bleu": bleu_score,
                    "tokenizer_vocab": tokenizer.vocab,
                }
                torch.save(checkpoint, f"{args.output_dir}/best_model.pt")
                print(f"   ✅ Saved best model (BLEU: {bleu_score:.2f})")

            # Save sample predictions
            with open(f"{args.output_dir}/predictions_epoch{epoch+1}.json", "w") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "bleu": bleu_score,
                        "samples": [
                            {"prediction": pred, "reference": ref}
                            for pred, ref in zip(preds[:10], refs[:10])
                        ],
                    },
                    f,
                    indent=2,
                )

    print(f"\n✅ Training completed! Best BLEU: {best_bleu:.2f}")
    print(f"💾 Models saved in: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument(
        "--train_features",
        type=str,
        required=True,
        help="Path to train features directory",
    )
    parser.add_argument(
        "--train_manifest", type=str, required=True, help="Path to train manifest TSV"
    )
    parser.add_argument(
        "--val_features", type=str, required=True, help="Path to val features directory"
    )
    parser.add_argument(
        "--val_manifest", type=str, required=True, help="Path to val manifest TSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Output directory for checkpoints",
    )

    # Model hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--max_frames", type=int, default=300)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--eval_every", type=int, default=5, help="Evaluate every N epochs"
    )
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args)
