#!/usr/bin/env python3
"""
Fine-tuning SONAR Decoder per How2Sign - Versione Seq2Seq con Attention
Decoder completo autoregressivo per generare traduzioni complete
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
import numpy as np
from sacrebleu.metrics import BLEU
import random


# ============================================================================
# Tokenizer
# ============================================================================


class Tokenizer:
    """Tokenizer per convertire testo in token IDs e viceversa"""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0

        # Special tokens
        self.PAD_token = 0
        self.SOS_token = 1  # Start of sentence
        self.EOS_token = 2  # End of sentence
        self.UNK_token = 3

        self.word2idx["<PAD>"] = self.PAD_token
        self.word2idx["<SOS>"] = self.SOS_token
        self.word2idx["<EOS>"] = self.EOS_token
        self.word2idx["<UNK>"] = self.UNK_token
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = 4

    def build_vocab(self, texts, min_freq=2):
        """Costruisce vocabulary da lista di testi"""
        # Conta frequenze
        word_freq = {}
        for text in texts:
            for word in text.lower().split():
                word_freq[word] = word_freq.get(word, 0) + 1

        # Aggiungi parole con frequenza >= min_freq
        for word, freq in sorted(word_freq.items()):
            if freq >= min_freq:
                self.word2idx[word] = self.vocab_size
                self.idx2word[self.vocab_size] = word
                self.vocab_size += 1

        print(f"✅ Vocabulary size: {self.vocab_size} (min_freq={min_freq})")

    def encode(self, text, max_length=50):
        """Converte testo in lista di token IDs"""
        words = text.lower().split()
        ids = [self.SOS_token]
        ids.extend([self.word2idx.get(word, self.UNK_token) for word in words])
        ids.append(self.EOS_token)

        # Pad to max_length
        if len(ids) < max_length:
            ids.extend([self.PAD_token] * (max_length - len(ids)))
        else:
            ids = ids[: max_length - 1] + [self.EOS_token]

        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids, skip_special=True):
        """Converte token IDs in testo"""
        words = []
        for id in ids:
            if isinstance(id, torch.Tensor):
                id = id.item()

            if skip_special and id in [self.PAD_token, self.SOS_token, self.EOS_token]:
                if id == self.EOS_token:
                    break
                continue

            word = self.idx2word.get(id, "<UNK>")
            if word not in ["<PAD>", "<SOS>", "<EOS>"]:
                words.append(word)

        return " ".join(words)


# ============================================================================
# Dataset
# ============================================================================


class How2SignFeatureDataset(Dataset):
    """Dataset che carica feature pre-estratte"""

    def __init__(
        self, features_dir, manifest_path, tokenizer, max_frames=300, max_text_len=50
    ):
        self.features_dir = Path(features_dir)
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.max_text_len = max_text_len

        # Carica manifest
        self.manifest = pd.read_csv(manifest_path, sep="\t")

        # Filtra solo video con feature disponibili
        self.samples = []
        for idx, row in self.manifest.iterrows():
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

        # Carica feature
        feature_path = sample["feature_path"]
        if feature_path.suffix == ".pt":
            data = torch.load(feature_path, map_location="cpu")
            features = data["features"]
        else:
            features = np.load(feature_path)
            features = torch.from_numpy(features).float()

        # Pad/truncate features
        T, D = features.shape
        actual_length = min(T, self.max_frames)

        if T > self.max_frames:
            features = features[: self.max_frames]
        elif T < self.max_frames:
            padding = torch.zeros(self.max_frames - T, D)
            features = torch.cat([features, padding], dim=0)

        # Encode text
        target_ids = self.tokenizer.encode(sample["text"], max_length=self.max_text_len)

        return {
            "features": features,
            "target_ids": target_ids,
            "text": sample["text"],
            "video_id": sample["id"],
            "feature_length": actual_length,
        }


# ============================================================================
# Model: Attention Mechanism
# ============================================================================


class BahdanauAttention(nn.Module):
    """Attention mechanism per guardare feature rilevanti"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, hidden, encoder_outputs, mask=None):
        """
        Args:
            hidden: (B, hidden_dim) - stato hidden decoder
            encoder_outputs: (B, T, hidden_dim) - output encoder
            mask: (B, T) - maschera per padding
        Returns:
            context: (B, hidden_dim) - weighted sum degli encoder outputs
            attention_weights: (B, T) - pesi di attenzione
        """
        B, T, H = encoder_outputs.shape

        # Replica hidden per ogni timestep
        hidden_expanded = hidden.unsqueeze(1).expand(B, T, H)

        # Concatena e calcola score
        energy = torch.cat([hidden_expanded, encoder_outputs], dim=2)
        scores = self.attention(energy).squeeze(2)  # (B, T)

        # Applica mask (ignora padding)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax per ottenere pesi
        attention_weights = torch.softmax(scores, dim=1)

        # Weighted sum
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context = context.squeeze(1)  # (B, hidden_dim)

        return context, attention_weights


# ============================================================================
# Model: Seq2Seq con Attention
# ============================================================================


class Seq2SeqTranslationModel(nn.Module):
    """
    Encoder-Decoder con Attention per traduzione ASL→Text
    """

    def __init__(
        self, input_dim=256, hidden_dim=512, vocab_size=5000, num_layers=2, dropout=0.3
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # ENCODER: Feature temporali → Hidden states
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Projection per ridurre bidirectional a hidden_dim
        self.encoder_projection = nn.Linear(hidden_dim * 2, hidden_dim)

        # DECODER: Hidden states + previous word → next word
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)

        self.attention = BahdanauAttention(hidden_dim)

        self.decoder = nn.LSTM(
            hidden_dim * 2,  # embedding + context
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, features, target_ids, feature_lengths, teacher_forcing_ratio=0.5):
        """
        Args:
            features: (B, T, 256) - Feature estratte
            target_ids: (B, max_len) - Target token IDs
            feature_lengths: (B,) - Lunghezze reali delle feature
            teacher_forcing_ratio: Probabilità di usare ground truth durante training

        Returns:
            outputs: (B, max_len, vocab_size) - Logits per ogni timestep
        """
        B, T_enc, _ = features.shape
        _, T_dec = target_ids.shape

        # === ENCODER ===
        # Pack per gestire lunghezze variabili
        packed = nn.utils.rnn.pack_padded_sequence(
            features, feature_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        encoder_outputs, (hidden, cell) = self.encoder(packed)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            encoder_outputs, batch_first=True
        )

        # Project encoder outputs (bidirectional → hidden_dim)
        encoder_outputs = self.encoder_projection(encoder_outputs)

        # Inizializza decoder hidden state (usa ultimo hidden dell'encoder)
        # Concatena forward e backward
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (B, hidden*2)
        hidden = (
            self.encoder_projection(hidden).unsqueeze(0).repeat(self.num_layers, 1, 1)
        )

        cell = torch.zeros_like(hidden)

        # Mask per attention (ignora padding)
        mask = torch.arange(T_enc, device=features.device).expand(
            B, T_enc
        ) < feature_lengths.unsqueeze(1)

        # === DECODER ===
        outputs = []
        input_token = target_ids[:, 0]  # <SOS> token

        for t in range(1, T_dec):
            # Embed current token
            embedded = self.embedding(input_token)  # (B, hidden_dim)

            # Attention: guarda encoder outputs rilevanti
            context, _ = self.attention(hidden[-1], encoder_outputs, mask)

            # Concatena embedding + context
            decoder_input = torch.cat([embedded, context], dim=1).unsqueeze(1)

            # Decoder step
            output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))

            # Project to vocabulary
            logits = self.output_projection(output.squeeze(1))
            outputs.append(logits)

            # Teacher forcing: usa ground truth o prediction
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            if use_teacher_forcing:
                input_token = target_ids[:, t]
            else:
                input_token = logits.argmax(dim=1)

        outputs = torch.stack(outputs, dim=1)  # (B, T_dec-1, vocab_size)

        return outputs

    def generate(self, features, feature_lengths, tokenizer, max_length=50):
        """
        Genera traduzione in modalità inferenza (greedy decoding)
        """
        self.eval()
        B, T_enc, _ = features.shape
        device = features.device

        # Encoder
        packed = nn.utils.rnn.pack_padded_sequence(
            features, feature_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        encoder_outputs, (hidden, cell) = self.encoder(packed)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(
            encoder_outputs, batch_first=True
        )
        encoder_outputs = self.encoder_projection(encoder_outputs)

        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        hidden = (
            self.encoder_projection(hidden).unsqueeze(0).repeat(self.num_layers, 1, 1)
        )
        cell = torch.zeros_like(hidden)

        mask = torch.arange(T_enc, device=device).expand(
            B, T_enc
        ) < feature_lengths.unsqueeze(1)

        # Decoder (greedy)
        input_token = torch.tensor([tokenizer.SOS_token] * B, device=device)
        generated = []

        for _ in range(max_length):
            embedded = self.embedding(input_token)
            context, _ = self.attention(hidden[-1], encoder_outputs, mask)
            decoder_input = torch.cat([embedded, context], dim=1).unsqueeze(1)
            output, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            logits = self.output_projection(output.squeeze(1))

            predicted = logits.argmax(dim=1)
            generated.append(predicted)

            # Stop se tutti hanno generato <EOS>
            if (predicted == tokenizer.EOS_token).all():
                break

            input_token = predicted

        generated = torch.stack(generated, dim=1)  # (B, len)
        return generated


# ============================================================================
# Training
# ============================================================================


def train_epoch(model, dataloader, optimizer, criterion, device, tokenizer, epoch):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        features = batch["features"].to(device)
        target_ids = batch["target_ids"].to(device)
        feature_lengths = batch["feature_length"]

        # Forward
        teacher_forcing_ratio = 0.7 if epoch < 20 else 0.5
        outputs = model(features, target_ids, feature_lengths, teacher_forcing_ratio)

        # Loss (ignora <SOS>)
        target = target_ids[:, 1:].contiguous()
        outputs = outputs.contiguous()

        loss = criterion(outputs.view(-1, model.vocab_size), target.view(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, tokenizer):
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            features = batch["features"].to(device)
            feature_lengths = batch["feature_length"]
            texts = batch["text"]

            # Generate
            generated_ids = model.generate(
                features, feature_lengths, tokenizer, max_length=50
            )

            # Decode
            for ids, ref in zip(generated_ids, texts):
                pred = tokenizer.decode(ids)
                predictions.append(pred)
                references.append(ref)

    # BLEU
    bleu_scorer = BLEU()
    bleu_score = bleu_scorer.corpus_score(predictions, [references]).score

    return bleu_score, predictions, references


# ============================================================================
# Main
# ============================================================================


def main(args):
    print("=" * 80)
    print("🚀 Seq2Seq Fine-Tuning for How2Sign")
    print("=" * 80)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"📍 Device: {device}\n")

    # Tokenizer
    print("📝 Building tokenizer...")
    tokenizer = Tokenizer()
    train_manifest = pd.read_csv(args.train_manifest, sep="\t")
    tokenizer.build_vocab(train_manifest["text"].tolist(), min_freq=args.min_freq)

    # Datasets
    print("\n📦 Loading datasets...")
    train_dataset = How2SignFeatureDataset(
        args.train_features,
        args.train_manifest,
        tokenizer,
        max_frames=args.max_frames,
        max_text_len=args.max_text_len,
    )

    val_dataset = How2SignFeatureDataset(
        args.val_features,
        args.val_manifest,
        tokenizer,
        max_frames=args.max_frames,
        max_text_len=args.max_text_len,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Model
    print(f"\n🤖 Creating model...")
    model = Seq2SeqTranslationModel(
        input_dim=256,
        hidden_dim=args.hidden_dim,
        vocab_size=tokenizer.vocab_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")

    # Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.PAD_token)

    # Training
    print(f"\n🏋️ Training for {args.epochs} epochs...\n")

    best_bleu = 0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, tokenizer, epoch
        )
        print(f"\nEpoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f}")

        if epoch % args.eval_every == 0:
            bleu, preds, refs = evaluate(model, val_loader, device, tokenizer)
            print(f"Val BLEU: {bleu:.2f}")

            if bleu > best_bleu:
                best_bleu = bleu
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "bleu": bleu,
                    "tokenizer": {
                        "word2idx": tokenizer.word2idx,
                        "idx2word": tokenizer.idx2word,
                        "vocab_size": tokenizer.vocab_size,
                    },
                }
                torch.save(checkpoint, f"{args.output_dir}/best_model.pt")
                print(f"✅ Saved best model (BLEU: {bleu:.2f})")

            # Save samples
            with open(f"{args.output_dir}/predictions_epoch{epoch}.json", "w") as f:
                json.dump(
                    {
                        "epoch": epoch,
                        "bleu": bleu,
                        "samples": [
                            {"prediction": p, "reference": r}
                            for p, r in zip(preds[:10], refs[:10])
                        ],
                    },
                    f,
                    indent=2,
                )

    print(f"\n✅ Training completed! Best BLEU: {best_bleu:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--train_features", type=str, required=True)
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_features", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints_seq2seq")

    # Model
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--max_text_len", type=int, default=50)
    parser.add_argument("--min_freq", type=int, default=2)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    main(args)
