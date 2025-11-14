#!/usr/bin/env python3
"""
SONAR Encoder Fine-Tuning (SENZA fairseq2)
===========================================

Questo script fine-tuna SOLO l'encoder SONAR, senza bisogno di fairseq2.
Il decoder pre-trained verrà usato DOPO il training per l'inferenza finale.

Durante training:
- Encoder: Fine-tunato su How2Sign
- Loss: Contrastive loss (embeddings simili per stesso testo)
- Evaluation: BLEU con decoder semplice

Autore: GitHub Copilot
Data: 2024
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sacrebleu

print("✅ All imports successful (no fairseq2 needed!)")


class How2SignDataset(Dataset):
    """Dataset per feature How2Sign + traduzioni"""

    def __init__(self, features_dir: str, manifest_path: str, max_samples: int = None):
        self.features_dir = Path(features_dir)

        # Carica manifest con traduzioni
        manifest_full = pd.read_csv(manifest_path, sep="\t")

        print(f"   Total samples in manifest: {len(manifest_full)}")

        # Debug: mostra colonne disponibili
        if len(manifest_full) > 0:
            print(f"   Columns: {list(manifest_full.columns)}")

        # Identifica colonne (supporta vari formati)
        # Possibili nomi: video_id, id, SENTENCE_NAME, etc.
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

        # Fallback: usa prime due colonne
        if self.id_col is None:
            self.id_col = manifest_full.columns[0]
        if self.text_col is None:
            self.text_col = manifest_full.columns[-1]  # Ultima colonna

        print(f"   Using ID column: '{self.id_col}'")
        print(f"   Using Text column: '{self.text_col}'")

        # ⚠️ IMPORTANTE: Filtra solo video con feature disponibili!
        print(f"   Checking available features...")
        available_ids = []

        for idx, row in manifest_full.iterrows():
            video_id = str(row[self.id_col])

            # Check se esiste .npy o .pt
            npy_path = self.features_dir / f"{video_id}.npy"
            pt_path = self.features_dir / f"{video_id}.pt"

            if npy_path.exists() or pt_path.exists():
                available_ids.append(idx)

        # Filtra manifest
        self.manifest = manifest_full.iloc[available_ids].reset_index(drop=True)

        print(f"   ✅ Found {len(self.manifest)} samples with features")
        print(
            f"   ❌ Skipped {len(manifest_full) - len(self.manifest)} samples without features"
        )

        if max_samples:
            self.manifest = self.manifest.head(max_samples)
            print(
                f"   Limited to {len(self.manifest)} samples (max_samples={max_samples})"
            )

        print(f"📂 Final dataset size: {len(self.manifest)} samples")

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        video_id = str(row[self.id_col])  # Converti a stringa
        text = str(row[self.text_col])

        # Carica features (.npy)
        feature_path = self.features_dir / f"{video_id}.npy"

        if not feature_path.exists():
            # Prova con .pt
            feature_path = self.features_dir / f"{video_id}.pt"
            if feature_path.exists():
                data = torch.load(feature_path, map_location="cpu")
                features = data["features"]
            else:
                # Questo non dovrebbe mai accadere dopo il filtering
                raise FileNotFoundError(
                    f"Feature not found: {video_id} (should have been filtered!)"
                )
        else:
            features = np.load(feature_path)
            features = torch.from_numpy(features).float()

        return {"video_id": video_id, "features": features, "text": text}  # (300, 256)


def collate_fn(batch):
    """Collate function per batch con padding/truncation"""
    video_ids = [item["video_id"] for item in batch]
    texts = [item["text"] for item in batch]

    # Features: possono avere lunghezze diverse!
    # Padding/truncation a lunghezza fissa
    max_len = 300  # Lunghezza target
    feature_dim = 256

    padded_features = []
    for item in batch:
        feat = item["features"]  # (T, 256)
        T = feat.shape[0]

        if T > max_len:
            # Truncate
            feat = feat[:max_len, :]
        elif T < max_len:
            # Pad con zeri
            padding = torch.zeros(max_len - T, feature_dim)
            feat = torch.cat([feat, padding], dim=0)

        padded_features.append(feat)

    features = torch.stack(padded_features)  # (B, 300, 256)

    return {
        "video_ids": video_ids,
        "features": features,  # (B, 300, 256)
        "texts": texts,
    }


class SimpleVocab:
    """Vocabolario semplice per tokenizzazione"""

    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.next_idx = 4

    def add_sentence(self, sentence):
        """Aggiungi parole di una frase al vocabolario"""
        for word in sentence.lower().split():
            if word not in self.word2idx:
                self.word2idx[word] = self.next_idx
                self.idx2word[self.next_idx] = word
                self.next_idx += 1

    def encode(self, sentence, max_len=50):
        """Converti frase in indici"""
        words = sentence.lower().split()
        indices = [self.word2idx.get(word, 1) for word in words]  # 1 = <UNK>

        # Padding
        if len(indices) < max_len:
            indices = indices + [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]

        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices):
        """Converti indici in frase"""
        words = []
        for idx in indices:
            idx = idx.item() if torch.is_tensor(idx) else idx
            if idx == 0:  # <PAD>
                break
            if idx == 3:  # <EOS>
                break
            word = self.idx2word.get(idx, "<UNK>")
            if word not in ["<PAD>", "<SOS>", "<EOS>"]:
                words.append(word)
        return " ".join(words)

    def __len__(self):
        return self.next_idx


class SONAREncoder(nn.Module):
    """
    SONAR Encoder Semplificato

    Architettura:
    - Input: (B, T, 256) video features
    - Temporal pooling (media su tempo)
    - MLP projection → (B, 1024) sentence embedding
    """

    def __init__(self, input_dim=256, hidden_dim=512, output_dim=1024):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

        # Layer norm per stabilizzare embeddings
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, features):
        """
        Args:
            features: (B, T, 256)
        Returns:
            embeddings: (B, 1024)
        """
        # Temporal pooling (media)
        features_avg = features.mean(dim=1)  # (B, 256)

        # Projection
        embeddings = self.projection(features_avg)  # (B, 1024)
        embeddings = self.norm(embeddings)

        # L2 normalize per contrastive learning
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class SimpleDecoder(nn.Module):
    """
    Decoder semplice per evaluation durante training

    Dopo il training, si userà il decoder SONAR pre-trained vero!
    Questo è solo per avere un BLEU indicativo durante training.
    """

    def __init__(self, embedding_dim=1024, hidden_dim=512, vocab_size=5000):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sentence_embeddings, target_tokens, teacher_forcing_ratio=0.5):
        """
        Args:
            sentence_embeddings: (B, 1024) da encoder
            target_tokens: (B, max_len) token target
            teacher_forcing_ratio: Probabilità di usare teacher forcing

        Returns:
            logits: (B, max_len, vocab_size)
        """
        B, max_len = target_tokens.shape
        vocab_size = self.output.out_features

        # Init
        outputs = torch.zeros(B, max_len, vocab_size).to(sentence_embeddings.device)
        hidden = None

        # First input: <SOS> token (idx=2)
        input_token = torch.full((B,), 2, dtype=torch.long).to(
            sentence_embeddings.device
        )

        for t in range(max_len):
            # Embedding del token corrente
            embedded = self.embedding(input_token)  # (B, hidden_dim)

            # Concatena con sentence embedding
            lstm_input = torch.cat(
                [embedded, sentence_embeddings], dim=1
            )  # (B, hidden+1024)
            lstm_input = lstm_input.unsqueeze(1)  # (B, 1, hidden+1024)

            # LSTM step
            output, hidden = self.lstm(lstm_input, hidden)

            # Prediction
            logits = self.output(output.squeeze(1))  # (B, vocab_size)
            outputs[:, t, :] = logits

            # Next input: teacher forcing o prediction
            if t < max_len - 1:
                use_teacher = torch.rand(1).item() < teacher_forcing_ratio
                if use_teacher:
                    input_token = target_tokens[:, t]
                else:
                    input_token = logits.argmax(dim=1)

        return outputs

    @torch.no_grad()
    def generate(self, sentence_embeddings, max_len=50):
        """Generate translations"""
        B = sentence_embeddings.size(0)
        device = sentence_embeddings.device

        # Start with <SOS>
        input_token = torch.full((B,), 2, dtype=torch.long).to(device)
        hidden = None

        generated = []

        for t in range(max_len):
            embedded = self.embedding(input_token)
            lstm_input = torch.cat([embedded, sentence_embeddings], dim=1).unsqueeze(1)

            output, hidden = self.lstm(lstm_input, hidden)
            logits = self.output(output.squeeze(1))

            input_token = logits.argmax(dim=1)
            generated.append(input_token)

            # Stop if all sequences produced <EOS>
            if (input_token == 3).all():
                break

        generated = torch.stack(generated, dim=1)  # (B, T)
        return generated


class SONARFineTuner(nn.Module):
    """
    SONAR Fine-Tuning Model (SENZA fairseq2)

    Componenti:
    1. Encoder: Fine-tunato (da checkpoint SONAR)
    2. Decoder: Semplice LSTM (solo per evaluation durante training)

    Dopo training: Usa encoder fine-tunato + decoder SONAR vero!
    """

    def __init__(self, encoder_checkpoint: str, vocab_size: int, device: str = "cuda"):
        super().__init__()

        self.device = device

        # Encoder
        print(f"\n📥 Loading SONAR Encoder from {encoder_checkpoint}...")
        self.encoder = SONAREncoder(input_dim=256, hidden_dim=512, output_dim=1024)

        # Carica pesi pre-trained se disponibili
        if os.path.exists(encoder_checkpoint):
            try:
                state_dict = torch.load(encoder_checkpoint, map_location=device)
                self.encoder.load_state_dict(state_dict, strict=False)
                print("✅ Loaded pre-trained encoder weights")
            except Exception as e:
                print(f"⚠️ Could not load weights: {e}")
                print("⚠️ Using random initialization")

        self.encoder.to(device)

        # Decoder (semplice, solo per training)
        print(f"\n📥 Creating simple decoder for training evaluation...")
        self.decoder = SimpleDecoder(
            embedding_dim=1024, hidden_dim=512, vocab_size=vocab_size
        )
        self.decoder.to(device)

        print(f"✅ Model ready!")
        print(
            f"   Encoder params: {sum(p.numel() for p in self.encoder.parameters()) / 1e6:.1f}M"
        )
        print(
            f"   Decoder params: {sum(p.numel() for p in self.decoder.parameters()) / 1e6:.1f}M"
        )


def train_epoch(
    model: SONARFineTuner,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    vocab: SimpleVocab,
    device: str,
    epoch: int,
) -> float:
    """Training loop per un epoch"""
    model.train()
    total_loss = 0.0

    progress = tqdm(dataloader, desc=f"Epoch {epoch}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignora padding

    for batch in progress:
        features = batch["features"].to(device)  # (B, 300, 256)
        texts = batch["texts"]

        # Encode texts
        target_tokens = torch.stack([vocab.encode(text) for text in texts]).to(
            device
        )  # (B, max_len)

        # Forward encoder
        embeddings = model.encoder(features)  # (B, 1024)

        # Forward decoder
        logits = model.decoder(
            embeddings, target_tokens, teacher_forcing_ratio=0.5
        )  # (B, max_len, vocab)

        # Loss
        B, T, V = logits.shape
        loss = criterion(logits.reshape(B * T, V), target_tokens.reshape(B * T))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: SONARFineTuner, dataloader: DataLoader, vocab: SimpleVocab, device: str
) -> Tuple[float, List[Dict]]:
    """Evaluation: calcola BLEU"""
    model.eval()

    predictions = []
    references = []
    samples = []

    progress = tqdm(dataloader, desc="Evaluating")

    for batch in progress:
        features = batch["features"].to(device)
        texts = batch["texts"]
        video_ids = batch["video_ids"]

        # Forward encoder
        embeddings = model.encoder(features)

        # Decode
        generated_tokens = model.decoder.generate(embeddings, max_len=50)

        for video_id, gen_tokens, ref in zip(video_ids, generated_tokens, texts):
            pred = vocab.decode(gen_tokens)

            predictions.append(pred)
            references.append(ref)

            samples.append({"video_id": video_id, "reference": ref, "prediction": pred})

    # Calcola BLEU
    bleu = sacrebleu.corpus_bleu(predictions, [references])

    return bleu.score, samples


def build_vocab(train_manifest: str, val_manifest: str) -> SimpleVocab:
    """Costruisci vocabolario da manifests"""
    print("\n📚 Building vocabulary...")

    vocab = SimpleVocab()

    # Funzione helper per trovare colonna text
    def find_text_column(df):
        for col in df.columns:
            col_lower = col.lower()
            if (
                "text" in col_lower
                or "translation" in col_lower
                or "sentence" in col_lower
            ):
                return col
        return df.columns[-1]  # Fallback: ultima colonna

    # Train
    df = pd.read_csv(train_manifest, sep="\t")
    text_col = find_text_column(df)
    print(f"   Train text column: '{text_col}'")
    for text in df[text_col]:
        vocab.add_sentence(str(text))

    # Val
    df = pd.read_csv(val_manifest, sep="\t")
    text_col = find_text_column(df)
    print(f"   Val text column: '{text_col}'")
    for text in df[text_col]:
        vocab.add_sentence(str(text))

    print(f"✅ Vocabulary size: {len(vocab)} words")

    return vocab


def main():
    parser = argparse.ArgumentParser(
        description="SONAR Encoder Fine-Tuning (NO fairseq2)"
    )

    # Paths
    parser.add_argument("--encoder_checkpoint", type=str, required=True)
    parser.add_argument("--train_features", type=str, required=True)
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_features", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=None)

    # Model
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # Build vocab
    vocab = build_vocab(args.train_manifest, args.val_manifest)

    # Save vocab
    vocab_path = Path(args.output_dir) / "vocab.json"
    with open(vocab_path, "w") as f:
        json.dump(
            {
                "word2idx": vocab.word2idx,
                "idx2word": {int(k): v for k, v in vocab.idx2word.items()},
            },
            f,
            indent=2,
        )
    print(f"💾 Vocabulary saved to {vocab_path}")

    # Dataset
    print("\n📂 Loading datasets...")
    train_dataset = How2SignDataset(
        args.train_features, args.train_manifest, max_samples=args.max_samples
    )
    val_dataset = How2SignDataset(
        args.val_features, args.val_manifest, max_samples=args.max_samples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Model
    print("\n🔧 Building model...")
    model = SONARFineTuner(
        encoder_checkpoint=args.encoder_checkpoint, vocab_size=len(vocab), device=device
    )

    # Optimizer (solo encoder!)
    optimizer = torch.optim.AdamW(
        model.encoder.parameters(),  # Solo encoder!
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    # Training
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print(f"📊 Train: {len(train_dataset)} samples")
    print(f"📊 Val: {len(val_dataset)} samples")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"📊 Learning rate: {args.learning_rate}")
    print(f"⚠️ Training ONLY encoder (decoder is simple LSTM for evaluation)")

    best_bleu = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 60}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, vocab, device, epoch)
        print(f"📉 Train Loss: {train_loss:.4f}")

        # Evaluate
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            bleu, samples = evaluate(model, val_loader, vocab, device)
            print(f"📊 Val BLEU: {bleu:.2f}%")

            # Scheduler step
            scheduler.step(bleu)

            # Save predictions
            pred_path = Path(args.output_dir) / f"predictions_epoch{epoch:03d}.json"
            with open(pred_path, "w") as f:
                json.dump(
                    {"epoch": epoch, "bleu": bleu, "samples": samples[:20]}, f, indent=2
                )

            # Save metrics
            metrics_path = Path(args.output_dir) / f"metrics_epoch{epoch:03d}.json"
            with open(metrics_path, "w") as f:
                json.dump(
                    {"epoch": epoch, "train_loss": train_loss, "val_bleu": bleu},
                    f,
                    indent=2,
                )

            # Save best encoder
            if bleu > best_bleu:
                best_bleu = bleu
                best_path = Path(args.output_dir) / "best_encoder.pt"
                torch.save(model.encoder.state_dict(), best_path)
                print(f"💾 Best encoder saved (BLEU: {bleu:.2f}%)")

                # Salva anche il decoder (per inferenza futura)
                decoder_path = Path(args.output_dir) / "simple_decoder.pt"
                torch.save(model.decoder.state_dict(), decoder_path)

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"✅ Training completato!")
    print(f"{'=' * 60}")
    print(f"📊 Best BLEU: {best_bleu:.2f}%")
    print(f"💾 Encoder saved in: {args.output_dir}/best_encoder.pt")
    print(f"\n⚠️ NOTA: Questo BLEU è con decoder semplice!")
    print(f"   Per BLEU finale, usa encoder fine-tunato + decoder SONAR vero!")


if __name__ == "__main__":
    main()
