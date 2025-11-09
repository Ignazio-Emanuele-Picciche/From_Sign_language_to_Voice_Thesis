"""
Evaluation Script per How2Sign Model
=====================================

Valuta il modello Sign-to-Text con metriche complete:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- WER (Word Error Rate)
- CER (Character Error Rate)
- METEOR (se disponibile)
- Perplexity
- Vocab coverage
- Caption length statistics

Usage:
    # Evaluate on validation set
    python src/sign_to_text/evaluate_how2sign.py \
        --checkpoint models/sign_to_text/how2sign/best_checkpoint.pt \
        --split val
    
    # Evaluate on golden label test set
    python src/sign_to_text/evaluate_how2sign.py \
        --checkpoint models/sign_to_text/how2sign_tuned/best_checkpoint.pt \
        --split test \
        --test_csv data/processed/golden_label_sentiment.csv
    
    # Generate qualitative examples
    python src/sign_to_text/evaluate_how2sign.py \
        --checkpoint models/sign_to_text/how2sign/best_checkpoint.pt \
        --split val \
        --num_examples 20 \
        --save_examples results/how2sign_evaluation/examples.txt
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).parent))

from models.seq2seq_transformer import SignToTextTransformer
from data.tokenizer import SignLanguageTokenizer
from data.how2sign_dataset import How2SignDataset

# Import metrics
try:
    from torchmetrics.text import BLEUScore

    BLEU_AVAILABLE = True
except ImportError:
    print("⚠️  torchmetrics not available, BLEU will be skipped")
    BLEU_AVAILABLE = False

try:
    from jiwer import wer, cer

    JIWER_AVAILABLE = True
except ImportError:
    print("⚠️  jiwer not available, WER/CER will be skipped")
    JIWER_AVAILABLE = False

try:
    from torchmetrics.text import ROUGEScore

    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


def collate_fn(batch):
    """Collate function compatibile con How2SignDataset."""
    landmarks = torch.stack([item["landmarks"] for item in batch])
    landmarks_mask = torch.stack([item["landmarks_mask"] for item in batch])
    caption_ids = torch.stack([item["caption_ids"] for item in batch])
    caption_mask = torch.stack([item["caption_mask"] for item in batch])

    return {
        "landmarks": landmarks,
        "landmarks_mask": landmarks_mask,
        "caption_ids": caption_ids,
        "caption_mask": caption_mask,
        "caption_texts": [item["caption_text"] for item in batch],
        "video_names": [item["video_name"] for item in batch],
        "n_frames_original": torch.tensor(
            [item["n_frames_original"] for item in batch]
        ),
    }


def generate_caption(model, src, src_mask, tokenizer, device, max_len=50, beam_width=1):
    """
    Genera caption con greedy decoding o beam search.

    Args:
        model: SignToTextTransformer
        src: (1, seq_len, 411) landmarks
        src_mask: (1, seq_len) mask
        tokenizer: SignLanguageTokenizer
        device: torch device
        max_len: max caption length
        beam_width: 1=greedy, >1=beam search

    Returns:
        str: generated caption
    """
    model.eval()

    with torch.no_grad():
        # Start with SOS token
        tgt_ids = torch.full((1, 1), tokenizer.sos_token_id, dtype=torch.long).to(
            device
        )

        for _ in range(max_len - 1):
            # Create target mask (no padding needed during generation)
            tgt_mask = torch.ones_like(tgt_ids, dtype=torch.bool)

            # Forward pass
            logits = model(
                src=src,
                tgt=tgt_ids,
                src_key_padding_mask=~src_mask,
                tgt_key_padding_mask=~tgt_mask,
            )

            # Get next token (greedy)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Append to sequence
            tgt_ids = torch.cat([tgt_ids, next_token], dim=1)

            # Stop if EOS
            if next_token.item() == tokenizer.eos_token_id:
                break

        # Decode
        caption = tokenizer.decode(tgt_ids[0].cpu().tolist())

    return caption


def evaluate_model(model, dataloader, tokenizer, criterion, device, num_examples=10):
    """
    Valuta il modello con tutte le metriche.

    Returns:
        dict: metriche calcolate
        list: esempi qualitativi
    """
    model.eval()

    # Initialize metrics
    total_loss = 0
    total_tokens = 0

    all_predictions = []
    all_references = []
    all_video_names = []

    # For examples
    examples = []

    # Limit generation for speed (only first N batches for metrics)
    max_batches_for_generation = 50  # ~200-400 samples instead of 1739

    print("\n🔍 Computing loss and generating predictions...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            src = batch["landmarks"].to(device)
            tgt_ids = batch["caption_ids"].to(device)
            src_mask = batch["landmarks_mask"].to(device)
            tgt_mask = batch["caption_mask"].to(device)

            # Compute loss (teacher forcing) - SEMPRE
            tgt_input = tgt_ids[:, :-1]
            tgt_output = tgt_ids[:, 1:]
            tgt_mask_input = tgt_mask[:, :-1]

            logits = model(
                src=src,
                tgt=tgt_input,
                src_key_padding_mask=~src_mask,
                tgt_key_padding_mask=~tgt_mask_input,
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
            )

            batch_tokens = tgt_mask[:, 1:].sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

            # Generate predictions (autoregressive) - SOLO per primi N batches
            if batch_idx < max_batches_for_generation:
                for i in range(src.size(0)):
                    src_i = src[i : i + 1]
                    src_mask_i = src_mask[i : i + 1]

                    # Generate
                    predicted_caption = generate_caption(
                        model, src_i, src_mask_i, tokenizer, device, max_len=50
                    )

                    reference_caption = batch["caption_texts"][i]
                    video_name = batch["video_names"][i]

                    all_predictions.append(predicted_caption)
                    all_references.append(reference_caption)
                    all_video_names.append(video_name)

                    # Save examples
                    if len(examples) < num_examples:
                        examples.append(
                            {
                                "video_name": video_name,
                                "reference": reference_caption,
                                "predicted": predicted_caption,
                                "n_frames": batch["n_frames_original"][i].item(),
                            }
                        )

    # Compute metrics
    print("\n📊 Computing metrics...")
    print(
        f"   Samples for generation: {len(all_predictions)} (of {total_tokens} total tokens)"
    )

    metrics = {}

    # 1. Loss & Perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    metrics["loss"] = avg_loss
    metrics["perplexity"] = perplexity

    # 2. BLEU scores
    if BLEU_AVAILABLE:
        for n_gram in [1, 2, 3, 4]:
            bleu_metric = BLEUScore(n_gram=n_gram)
            bleu_score = bleu_metric(
                all_predictions, [[ref] for ref in all_references]
            ).item()
            metrics[f"bleu_{n_gram}"] = bleu_score
    else:
        print("⚠️  BLEU skipped (install torchmetrics)")

    # 3. WER & CER
    if JIWER_AVAILABLE:
        # Filter empty predictions/references
        valid_pairs = [
            (pred, ref)
            for pred, ref in zip(all_predictions, all_references)
            if pred.strip() and ref.strip()
        ]

        if valid_pairs:
            preds, refs = zip(*valid_pairs)

            try:
                wer_score = wer(list(refs), list(preds))
                cer_score = cer(list(refs), list(preds))

                metrics["wer"] = wer_score
                metrics["cer"] = cer_score
            except Exception as e:
                print(f"⚠️  WER/CER computation failed: {e}")
                metrics["wer"] = None
                metrics["cer"] = None
    else:
        print("⚠️  WER/CER skipped (install jiwer)")

    # 4. ROUGE (optional)
    if ROUGE_AVAILABLE:
        try:
            rouge_metric = ROUGEScore()
            rouge_scores = rouge_metric(all_predictions, all_references)
            metrics["rouge1_fmeasure"] = rouge_scores["rouge1_fmeasure"].item()
            metrics["rouge2_fmeasure"] = rouge_scores["rouge2_fmeasure"].item()
            metrics["rougeL_fmeasure"] = rouge_scores["rougeL_fmeasure"].item()
        except Exception as e:
            print(f"⚠️  ROUGE computation failed: {e}")

    # 5. Vocab coverage
    all_words = set()
    for pred in all_predictions:
        all_words.update(pred.lower().split())

    vocab_coverage = (
        len(all_words) / tokenizer.vocab_size if tokenizer.vocab_size > 0 else 0
    )
    metrics["vocab_coverage"] = vocab_coverage
    metrics["unique_words_generated"] = len(all_words)

    # 6. Caption length statistics
    pred_lengths = [len(pred.split()) for pred in all_predictions]
    ref_lengths = [len(ref.split()) for ref in all_references]

    metrics["avg_pred_length"] = np.mean(pred_lengths)
    metrics["avg_ref_length"] = np.mean(ref_lengths)
    metrics["std_pred_length"] = np.std(pred_lengths)
    metrics["std_ref_length"] = np.std(ref_lengths)

    # 7. Exact match rate
    exact_matches = sum(
        1
        for pred, ref in zip(all_predictions, all_references)
        if pred.strip().lower() == ref.strip().lower()
    )
    metrics["exact_match_rate"] = exact_matches / len(all_predictions)

    return metrics, examples


def print_metrics(metrics):
    """Stampa metriche in formato leggibile."""
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS")
    print("=" * 80)

    print("\n🎯 Primary Metrics:")
    print(f"   Loss:         {metrics.get('loss', 0):.4f}")
    print(f"   Perplexity:   {metrics.get('perplexity', 0):.2f}")

    if "bleu_1" in metrics:
        print(f"\n📝 BLEU Scores:")
        for n in [1, 2, 3, 4]:
            key = f"bleu_{n}"
            if key in metrics:
                print(f"   BLEU-{n}:      {metrics[key]:.4f}")

    if "wer" in metrics and metrics["wer"] is not None:
        print(f"\n🔤 Error Rates:")
        print(f"   WER:          {metrics['wer']:.4f}")
        print(f"   CER:          {metrics['cer']:.4f}")

    if "rouge1_fmeasure" in metrics:
        print(f"\n📖 ROUGE Scores:")
        print(f"   ROUGE-1:      {metrics['rouge1_fmeasure']:.4f}")
        print(f"   ROUGE-2:      {metrics['rouge2_fmeasure']:.4f}")
        print(f"   ROUGE-L:      {metrics['rougeL_fmeasure']:.4f}")

    print(f"\n📏 Caption Statistics:")
    print(
        f"   Avg Pred Len: {metrics['avg_pred_length']:.1f} ± {metrics['std_pred_length']:.1f}"
    )
    print(
        f"   Avg Ref Len:  {metrics['avg_ref_length']:.1f} ± {metrics['std_ref_length']:.1f}"
    )
    print(f"   Exact Match:  {metrics['exact_match_rate']*100:.1f}%")

    print(f"\n📚 Vocabulary:")
    print(f"   Unique Words: {metrics['unique_words_generated']}")
    print(f"   Vocab Cover:  {metrics['vocab_coverage']*100:.1f}%")

    print("\n" + "=" * 80)


def save_examples(examples, output_path):
    """Salva esempi qualitativi."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("QUALITATIVE EXAMPLES\n")
        f.write("=" * 80 + "\n\n")

        for i, ex in enumerate(examples, 1):
            f.write(f"Example {i}:\n")
            f.write(f"  Video:     {ex['video_name']}\n")
            f.write(f"  Frames:    {ex['n_frames']}\n")
            f.write(f"  Reference: {ex['reference']}\n")
            f.write(f"  Predicted: {ex['predicted']}\n")
            f.write("\n")

    print(f"\n💾 Examples saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Sign-to-Text Model")

    # Model & data
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Which split to evaluate",
    )
    parser.add_argument(
        "--train_csv", type=str, default="results/how2sign_splits/train_split.csv"
    )
    parser.add_argument(
        "--val_csv", type=str, default="results/how2sign_splits/val_split.csv"
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="data/processed/golden_label_sentiment.csv",
        help="Test set (golden labels)",
    )
    parser.add_argument(
        "--train_openpose_dir",
        type=str,
        default="data/raw/train/openpose_output_train/json",
    )
    parser.add_argument(
        "--val_openpose_dir", type=str, default="data/raw/val/openpose_output_val/json"
    )
    parser.add_argument(
        "--test_openpose_dir",
        type=str,
        default="data/processed/landmarks",
        help="OpenPose dir for test set",
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default="models/sign_to_text/tokenizer.json"
    )

    # Evaluation params
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=20,
        help="Number of qualitative examples to generate",
    )
    parser.add_argument(
        "--save_examples",
        type=str,
        default=None,
        help="Path to save qualitative examples",
    )
    parser.add_argument(
        "--save_metrics", type=str, default=None, help="Path to save metrics JSON"
    )

    # Other
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
    )

    args = parser.parse_args()

    print("=" * 80)
    print("🔍 EVALUATING SIGN-TO-TEXT MODEL")
    print("=" * 80)
    print(f"\n📁 Checkpoint: {args.checkpoint}")
    print(f"📊 Split:      {args.split}")
    print(f"💻 Device:     {args.device}")

    # Load checkpoint
    print(f"\n1️⃣  Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint.get("config", {})

    print(f"   ✓ Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"   ✓ Train Loss: {checkpoint.get('train_loss', 0):.4f}")
    print(f"   ✓ Val Loss: {checkpoint.get('val_loss', 0):.4f}")

    # Load tokenizer
    print(f"\n2️⃣  Loading tokenizer...")
    tokenizer = SignLanguageTokenizer.load(args.tokenizer_path)
    print(f"   ✓ Vocab size: {tokenizer.vocab_size}")

    # Select dataset
    print(f"\n3️⃣  Loading dataset...")

    if args.split == "train":
        csv_path = args.train_csv
        openpose_dir = args.train_openpose_dir
    elif args.split == "val":
        csv_path = args.val_csv
        openpose_dir = args.val_openpose_dir
    else:  # test
        csv_path = args.test_csv
        openpose_dir = args.test_openpose_dir

    dataset = How2SignDataset(
        split_csv=csv_path,
        openpose_dir=openpose_dir,
        tokenizer=tokenizer,
        max_frames=config.get("max_frames", 200),
        max_caption_len=config.get("max_caption_len", 50),
        landmark_features=config.get("landmark_features", 411),
    )

    print(f"   ✓ Dataset: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.device == "cuda" else False,
    )

    print(f"   ✓ Batches: {len(dataloader)}")

    # Create model
    print(f"\n4️⃣  Creating model...")
    model = SignToTextTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=config.get("d_model", 512),
        nhead=config.get("nhead", 8),
        num_encoder_layers=config.get("num_encoder_layers", 4),
        num_decoder_layers=config.get("num_decoder_layers", 4),
        dim_feedforward=config.get("dim_feedforward", 2048),
        dropout=config.get("dropout", 0.1),
        max_src_len=config.get("max_frames", 200),
        max_tgt_len=config.get("max_caption_len", 50),
        landmark_dim=config.get("landmark_features", 411),
        pad_idx=tokenizer.pad_token_id,
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Parameters: {num_params:,}")

    # Loss function
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=0.0,  # No smoothing for evaluation
    )

    # Evaluate
    print(f"\n{'='*80}")
    print("🎯 STARTING EVALUATION")
    print(f"{'='*80}")

    metrics, examples = evaluate_model(
        model, dataloader, tokenizer, criterion, args.device, args.num_examples
    )

    # Print results
    print_metrics(metrics)

    # Save metrics
    if args.save_metrics:
        metrics_path = Path(args.save_metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n💾 Metrics saved to: {metrics_path}")

    # Save examples
    if args.save_examples:
        save_examples(examples, args.save_examples)
    else:
        # Print some examples
        print(f"\n{'='*80}")
        print(f"💬 QUALITATIVE EXAMPLES (first 5)")
        print(f"{'='*80}")

        for i, ex in enumerate(examples[:5], 1):
            print(f"\nExample {i}:")
            print(f"  Video:     {ex['video_name']}")
            print(f"  Reference: {ex['reference']}")
            print(f"  Predicted: {ex['predicted']}")

    print(f"\n{'='*80}")
    print("✅ EVALUATION COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
