#!/usr/bin/env python3
"""
Confronto Encoder tramite Embedding Similarity
===============================================

Invece di BLEU (che richiede decoder potente), confronta:
- Cosine similarity tra embeddings
- Clustering quality
- Nearest neighbor accuracy

Questo mostra se l'encoder fine-tunato produce embeddings migliori!
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


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
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


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
    video_ids = []

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
        video_ids.append(video_id)

        count += 1
        if max_samples and count >= max_samples:
            break

    features = torch.stack(features_list)

    print(f"✅ Loaded {len(features)} test samples")

    return features, texts, video_ids


@torch.no_grad()
def compute_embeddings(encoder, features, device="cpu"):
    """Compute embeddings"""
    encoder.eval()
    features = features.to(device)

    embeddings = encoder(features)

    return embeddings.cpu().numpy()


def compute_text_similarity_matrix(texts):
    """Compute text similarity (word overlap)"""
    n = len(texts)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        words_i = set(texts[i].lower().split())
        for j in range(i, n):
            words_j = set(texts[j].lower().split())

            if len(words_i) == 0 or len(words_j) == 0:
                sim = 0.0
            else:
                # Jaccard similarity
                intersection = len(words_i & words_j)
                union = len(words_i | words_j)
                sim = intersection / union if union > 0 else 0.0

            similarity_matrix[i, j] = sim
            similarity_matrix[j, i] = sim

    return similarity_matrix


def nearest_neighbor_accuracy(embeddings, text_similarity, k=5):
    """
    Nearest neighbor accuracy:
    Per ogni sample, controlla se i K nearest neighbors hanno testi simili
    """
    # Cosine similarity tra embeddings
    emb_similarity = cosine_similarity(embeddings)

    n = len(embeddings)
    correct = 0
    total = 0

    for i in range(n):
        # Top-K nearest neighbors (escludi se stesso)
        neighbors = np.argsort(emb_similarity[i])[::-1][1 : k + 1]

        # Controlla se hanno testi simili (text_similarity > 0.3)
        for j in neighbors:
            if text_similarity[i, j] > 0.3:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0
    return accuracy


def compute_intra_inter_distance(embeddings, texts):
    """
    Intra-class distance: distanza tra embeddings con testi simili
    Inter-class distance: distanza tra embeddings con testi diversi

    Encoder migliore → bassa intra-distance, alta inter-distance
    """
    emb_similarity = cosine_similarity(embeddings)
    text_similarity = compute_text_similarity_matrix(texts)

    # Similar pairs (text similarity > 0.3)
    similar_mask = text_similarity > 0.3
    np.fill_diagonal(similar_mask, False)  # Escludi diagonale

    # Dissimilar pairs (text similarity < 0.1)
    dissimilar_mask = text_similarity < 0.1

    # Distances (1 - cosine similarity)
    distances = 1 - emb_similarity

    intra_distance = distances[similar_mask].mean() if similar_mask.sum() > 0 else 0.0
    inter_distance = (
        distances[dissimilar_mask].mean() if dissimilar_mask.sum() > 0 else 0.0
    )

    # Separation score (higher is better)
    separation = inter_distance - intra_distance

    return intra_distance, inter_distance, separation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_encoder", type=str, required=True)
    parser.add_argument("--finetuned_encoder", type=str, required=True)
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output", type=str, default="embedding_comparison.json")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"🚀 Device: {device}\n")

    # Load test data
    print("📂 Loading test data...")
    features, texts, video_ids = load_test_data(
        args.features, args.manifest, args.max_samples
    )

    # Compute text similarity
    print("\n📊 Computing text similarity matrix...")
    text_similarity = compute_text_similarity_matrix(texts)

    # Test 1: Pre-trained encoder
    print("\n" + "=" * 60)
    print("TEST 1: Pre-trained Encoder (Zero-Shot)")
    print("=" * 60)

    encoder_pretrained = SONAREncoder()
    encoder_pretrained.load_state_dict(
        torch.load(args.pretrained_encoder, map_location=device), strict=False
    )
    encoder_pretrained.to(device)

    print("🔄 Computing embeddings...")
    emb_pretrained = compute_embeddings(encoder_pretrained, features, device)

    print("📊 Computing metrics...")
    nn_acc_pre = nearest_neighbor_accuracy(emb_pretrained, text_similarity, k=5)
    intra_pre, inter_pre, sep_pre = compute_intra_inter_distance(emb_pretrained, texts)

    print(f"\n📈 Nearest Neighbor Accuracy (k=5): {nn_acc_pre*100:.2f}%")
    print(f"📉 Intra-class Distance: {intra_pre:.4f}")
    print(f"📈 Inter-class Distance: {inter_pre:.4f}")
    print(f"🎯 Separation Score: {sep_pre:.4f}")

    # Test 2: Fine-tuned encoder
    print("\n" + "=" * 60)
    print("TEST 2: Fine-tuned Encoder (How2Sign)")
    print("=" * 60)

    encoder_finetuned = SONAREncoder()
    encoder_finetuned.load_state_dict(
        torch.load(args.finetuned_encoder, map_location=device)
    )
    encoder_finetuned.to(device)

    print("🔄 Computing embeddings...")
    emb_finetuned = compute_embeddings(encoder_finetuned, features, device)

    print("📊 Computing metrics...")
    nn_acc_ft = nearest_neighbor_accuracy(emb_finetuned, text_similarity, k=5)
    intra_ft, inter_ft, sep_ft = compute_intra_inter_distance(emb_finetuned, texts)

    print(f"\n📈 Nearest Neighbor Accuracy (k=5): {nn_acc_ft*100:.2f}%")
    print(f"📉 Intra-class Distance: {intra_ft:.4f}")
    print(f"📈 Inter-class Distance: {inter_ft:.4f}")
    print(f"🎯 Separation Score: {sep_ft:.4f}")

    # Comparison
    print("\n" + "=" * 60)
    print("📊 CONFRONTO FINALE")
    print("=" * 60)

    print(f"\nNearest Neighbor Accuracy:")
    print(f"  Pre-trained:  {nn_acc_pre*100:.2f}%")
    print(f"  Fine-tuned:   {nn_acc_ft*100:.2f}%")
    print(f"  Improvement:  {(nn_acc_ft-nn_acc_pre)*100:+.2f}%")

    print(f"\nSeparation Score (higher = better):")
    print(f"  Pre-trained:  {sep_pre:.4f}")
    print(f"  Fine-tuned:   {sep_ft:.4f}")
    print(f"  Improvement:  {(sep_ft-sep_pre):+.4f}")

    print(f"\nIntra-class Distance (lower = better):")
    print(f"  Pre-trained:  {intra_pre:.4f}")
    print(f"  Fine-tuned:   {intra_ft:.4f}")
    print(f"  Improvement:  {(intra_pre-intra_ft):+.4f}")

    print("=" * 60)

    # Overall assessment
    improvements = 0
    if nn_acc_ft > nn_acc_pre:
        improvements += 1
    if sep_ft > sep_pre:
        improvements += 1
    if intra_ft < intra_pre:
        improvements += 1

    print(f"\n🎯 Metriche migliorate: {improvements}/3")

    if improvements >= 2:
        print("✅ Fine-tuning ha MIGLIORATO la qualità degli embeddings!")
    elif improvements == 1:
        print("⚠️ Fine-tuning ha migliorato parzialmente")
    else:
        print("❌ Fine-tuning non ha migliorato (serve più training)")

    # Save results
    results = {
        "pretrained": {
            "nn_accuracy": float(nn_acc_pre),
            "intra_distance": float(intra_pre),
            "inter_distance": float(inter_pre),
            "separation_score": float(sep_pre),
        },
        "finetuned": {
            "nn_accuracy": float(nn_acc_ft),
            "intra_distance": float(intra_ft),
            "inter_distance": float(inter_ft),
            "separation_score": float(sep_ft),
        },
        "improvements": {
            "nn_accuracy_delta": float(nn_acc_ft - nn_acc_pre),
            "separation_delta": float(sep_ft - sep_pre),
            "intra_distance_delta": float(intra_pre - intra_ft),
            "metrics_improved": improvements,
        },
        "num_samples": len(texts),
    }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to {args.output}")


if __name__ == "__main__":
    main()
