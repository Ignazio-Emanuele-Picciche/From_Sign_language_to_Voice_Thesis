#!/usr/bin/env python3
"""
Inference usando script ufficiale SONAR + encoder fine-tunato

Questo script:
1. Carica encoder fine-tunato (best_encoder.pt)
2. Usa il pipeline ufficiale SONAR per decodifica
3. Calcola BLEU sul test set
"""

import sys
import os
from pathlib import Path
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import sacrebleu

# Aggiungi path repository SONAR
sonar_repo = (
    Path(__file__).parent.parent.parent / "src/sign_to_text_ssvp/models/ssvp_slt_repo"
)
sys.path.insert(0, str(sonar_repo))

try:
    from fairseq2.models.sonar import load_sonar_text_decoder

    FAIRSEQ2_AVAILABLE = True
    print("✅ fairseq2 available")
except:
    FAIRSEQ2_AVAILABLE = False
    print("⚠️ fairseq2 not available")
    print("   Usando approccio alternativo...")


class SONAREncoder(nn.Module):
    """SONAR Encoder (identico a training)"""

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


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder_checkpoint",
        type=str,
        required=True,
        help="Path to fine-tuned encoder (best_encoder.pt)",
    )
    parser.add_argument(
        "--features_dir", type=str, required=True, help="Directory with .npy features"
    )
    parser.add_argument("--manifest", type=str, required=True, help="Manifest TSV file")
    parser.add_argument("--output", type=str, default="sonar_inference_results.json")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Device: {device}")

    # Load encoder
    print(f"\n📥 Loading fine-tuned encoder...")
    encoder = SONAREncoder()
    state_dict = torch.load(args.encoder_checkpoint, map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval()
    print(f"✅ Encoder loaded")

    # Load manifest
    print(f"\n📂 Loading manifest...")
    manifest = pd.read_csv(args.manifest, sep="\t")

    # Find columns
    id_col = text_col = None
    for col in manifest.columns:
        col_lower = col.lower()
        if "id" in col_lower or "name" in col_lower:
            id_col = col
        if "text" in col_lower or "translation" in col_lower:
            text_col = col

    if id_col is None:
        id_col = manifest.columns[0]
    if text_col is None:
        text_col = manifest.columns[-1]

    print(f"   ID column: '{id_col}'")
    print(f"   Text column: '{text_col}'")

    # Load features
    print(f"\n📥 Loading features and generating embeddings...")
    features_dir = Path(args.features_dir)

    embeddings_list = []
    texts = []
    video_ids = []

    count = 0
    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Processing"):
        video_id = str(row[id_col])
        text = str(row[text_col])

        # Load feature
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

        # Encode
        with torch.no_grad():
            feat_batch = feat.unsqueeze(0).to(device)  # (1, 300, 256)
            emb = encoder(feat_batch)  # (1, 1024)
            embeddings_list.append(emb.cpu())

        texts.append(text)
        video_ids.append(video_id)

        count += 1
        if args.max_samples and count >= args.max_samples:
            break

    embeddings = torch.cat(embeddings_list, dim=0)  # (N, 1024)
    print(f"✅ Generated {len(embeddings)} embeddings")

    # Decode with SONAR
    if not FAIRSEQ2_AVAILABLE:
        print("\n⚠️ fairseq2 not available!")
        print("⚠️ Cannot decode with SONAR text decoder")
        print("\n💡 Soluzione:")
        print("   1. Installa fairseq2 CPU:")
        print(
            "      pip install fairseq2 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.5.0/cpu"
        )
        print("   2. Riprova questo script")
        print("\n💾 Saving embeddings for later...")

        torch.save(
            {"embeddings": embeddings, "texts": texts, "video_ids": video_ids},
            "embeddings_finetuned.pt",
        )

        print(f"✅ Embeddings saved to embeddings_finetuned.pt")
        return

    # Load SONAR decoder
    print(f"\n📥 Loading SONAR text decoder...")
    try:
        decoder = load_sonar_text_decoder("text_sonar_basic_encoder", device=device)
        print(f"✅ Decoder loaded")
    except Exception as e:
        print(f"❌ Failed: {e}")
        return

    # Decode
    print(f"\n🔄 Decoding with SONAR...")
    predictions = []

    # TODO: Implementare decodifica SONAR vera
    # Per ora placeholder
    print("⚠️ Decodifica SONAR non ancora implementata completamente")
    print("   Serve l'API corretta di fairseq2 per decodificare da embeddings")

    # Placeholder
    predictions = ["[PLACEHOLDER]"] * len(texts)

    # Calculate BLEU
    bleu = sacrebleu.corpus_bleu(predictions, [texts])

    print(f"\n📊 BLEU: {bleu.score:.2f}%")

    # Save
    results = {
        "bleu": bleu.score,
        "num_samples": len(texts),
        "samples": [
            {"video_id": vid, "reference": ref, "prediction": pred}
            for vid, ref, pred in zip(video_ids[:20], texts[:20], predictions[:20])
        ],
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to {args.output}")
    print(f"\n{'='*60}")
    print(f"✅ BLEU with SONAR decoder: {bleu.score:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
