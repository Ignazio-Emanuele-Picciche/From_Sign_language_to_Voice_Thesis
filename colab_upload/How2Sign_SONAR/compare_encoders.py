#!/usr/bin/env python3
"""
Confronto Encoder Pre-trained vs Fine-tunato
=============================================

Questo script confronta:
1. Encoder SONAR pre-trained (zero-shot)
2. Encoder SONAR fine-tunato su How2Sign

Usando lo stesso decoder semplice per entrambi → confronto equo!
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import sacrebleu


class SONAREncoder(nn.Module):
    """SONAR Encoder"""

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

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, features):
        features_avg = features.mean(dim=1)
        embeddings = self.projection(features_avg)
        embeddings = self.norm(embeddings)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


class SimpleVocab:
    """Vocabolario semplice"""

    def __init__(self, vocab_path):
        with open(vocab_path, "r") as f:
            data = json.load(f)

        self.word2idx = data["word2idx"]
        self.idx2word = {int(k): v for k, v in data["idx2word"].items()}

    def encode(self, sentence, max_len=50):
        words = sentence.lower().split()
        indices = [self.word2idx.get(word, 1) for word in words]

        if len(indices) < max_len:
            indices = indices + [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]

        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices):
        words = []
        for idx in indices:
            idx = idx.item() if torch.is_tensor(idx) else idx
            if idx == 0:
                break
            if idx == 3:
                break
            word = self.idx2word.get(idx, "<UNK>")
            if word not in ["<PAD>", "<SOS>", "<EOS>"]:
                words.append(word)
        return " ".join(words)


class SimpleDecoder(nn.Module):
    """Decoder semplice (stesso usato in training)"""

    def __init__(self, embedding_dim=1024, hidden_dim=512, vocab_size=5000):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim + embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    @torch.no_grad()
    def generate(self, sentence_embeddings, max_len=50):
        B = sentence_embeddings.size(0)
        device = sentence_embeddings.device

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

            if (input_token == 3).all():
                break

        generated = torch.stack(generated, dim=1)
        return generated


def load_test_data(features_dir, manifest_path, max_samples=None):
    """Load test features"""
    manifest = pd.read_csv(manifest_path, sep="\t")

    # Find columns
    id_col = text_col = None
    for col in manifest.columns:
        col_lower = col.lower()
        if "id" in col_lower or "name" in col_lower:
            id_col = col
        if "text" in col_lower:
            text_col = col

    if id_col is None:
        id_col = manifest.columns[0]
    if text_col is None:
        text_col = manifest.columns[-1]

    features_list = []
    texts = []

    features_dir = Path(features_dir)

    count = 0
    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Loading"):
        video_id = str(row[id_col])
        text = str(row[text_col])

        npy_path = features_dir / f"{video_id}.npy"
        pt_path = features_dir / f"{video_id}.pt"

        if npy_path.exists():
            feat = np.load(npy_path)
            feat = torch.from_numpy(feat).float()
        elif pt_path.exists():
            data = torch.load(pt_path, map_location="cpu")
            feat = data["features"]
        else:
            continue

        # Pad/truncate
        T = feat.shape[0]
        if T > 300:
            feat = feat[:300, :]
        elif T < 300:
            padding = torch.zeros(300 - T, 256)
            feat = torch.cat([feat, padding], dim=0)

        features_list.append(feat)
        texts.append(text)

        count += 1
        if max_samples and count >= max_samples:
            break

    features = torch.stack(features_list)

    print(f"✅ Loaded {len(features)} test samples")

    return features, texts


@torch.no_grad()
def evaluate_encoder(encoder, decoder, vocab, features, texts, device="cpu"):
    """Evaluate encoder + decoder"""

    encoder.eval()
    decoder.eval()

    features = features.to(device)

    print("🔄 Encoding...")
    embeddings = encoder(features)

    print("🔄 Decoding...")
    generated_tokens = decoder.generate(embeddings, max_len=50)

    predictions = []
    for tokens in generated_tokens:
        pred = vocab.decode(tokens)
        predictions.append(pred)

    bleu = sacrebleu.corpus_bleu(predictions, [texts])

    return bleu.score, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_encoder",
        type=str,
        required=True,
        help="Pre-trained SONAR encoder checkpoint",
    )
    parser.add_argument(
        "--finetuned_encoder",
        type=str,
        required=True,
        help="Fine-tuned encoder checkpoint",
    )
    parser.add_argument(
        "--decoder", type=str, required=True, help="Simple decoder checkpoint"
    )
    parser.add_argument("--vocab", type=str, required=True, help="Vocabulary JSON")
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output", type=str, default="comparison_results.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device}\n")

    # Load vocab
    print("📚 Loading vocabulary...")
    vocab = SimpleVocab(args.vocab)
    print(f"✅ Vocab size: {len(vocab.word2idx)}")

    # Load decoder
    print("\n📥 Loading decoder...")
    decoder = SimpleDecoder(vocab_size=len(vocab.word2idx))
    decoder.load_state_dict(torch.load(args.decoder, map_location=device))
    decoder.to(device)
    print("✅ Decoder loaded")

    # Load test data
    print("\n📂 Loading test data...")
    features, texts = load_test_data(args.features, args.manifest, args.max_samples)

    # Test 1: Pre-trained encoder (BASELINE)
    print("\n" + "=" * 60)
    print("TEST 1: Pre-trained Encoder (Zero-Shot Baseline)")
    print("=" * 60)

    encoder_pretrained = SONAREncoder()
    encoder_pretrained.load_state_dict(
        torch.load(args.pretrained_encoder, map_location=device), strict=False
    )
    encoder_pretrained.to(device)

    bleu_pretrained, preds_pretrained = evaluate_encoder(
        encoder_pretrained, decoder, vocab, features, texts, device
    )

    print(f"\n📊 BLEU (Pre-trained): {bleu_pretrained:.2f}%")

    # Test 2: Fine-tuned encoder
    print("\n" + "=" * 60)
    print("TEST 2: Fine-tuned Encoder (How2Sign Adapted)")
    print("=" * 60)

    encoder_finetuned = SONAREncoder()
    encoder_finetuned.load_state_dict(
        torch.load(args.finetuned_encoder, map_location=device)
    )
    encoder_finetuned.to(device)

    bleu_finetuned, preds_finetuned = evaluate_encoder(
        encoder_finetuned, decoder, vocab, features, texts, device
    )

    print(f"\n📊 BLEU (Fine-tuned): {bleu_finetuned:.2f}%")

    # Comparison
    improvement = bleu_finetuned - bleu_pretrained
    improvement_pct = (improvement / max(bleu_pretrained, 0.01)) * 100

    print("\n" + "=" * 60)
    print("📊 CONFRONTO FINALE")
    print("=" * 60)
    print(f"Pre-trained (Baseline):  {bleu_pretrained:.2f}%")
    print(f"Fine-tuned (How2Sign):   {bleu_finetuned:.2f}%")
    print(f"Improvement:             +{improvement:.2f}% ({improvement_pct:+.1f}%)")
    print("=" * 60)

    # Save results
    results = {
        "pretrained_bleu": bleu_pretrained,
        "finetuned_bleu": bleu_finetuned,
        "improvement": improvement,
        "improvement_percent": improvement_pct,
        "num_samples": len(texts),
        "samples": [
            {"reference": ref, "pretrained_pred": pred_pre, "finetuned_pred": pred_ft}
            for ref, pred_pre, pred_ft in zip(
                texts[:10], preds_pretrained[:10], preds_finetuned[:10]
            )
        ],
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to {args.output}")

    if improvement > 0:
        print("\n✅ Fine-tuning IMPROVED performance!")
    else:
        print("\n⚠️ Fine-tuning did not improve (may need more epochs/data)")


if __name__ == "__main__":
    main()
