"""
Post-Training Validation Script for Perceiver Resampler Architecture
Validates BLEU scores, translation quality, and mode collapse detection
"""

import torch
import json
import os
from pathlib import Path
from collections import Counter
import numpy as np
from tqdm import tqdm
import argparse

# Import components
from train_sonar_with_t5 import SONARwithT5
from transformers import T5Tokenizer
from sacrebleu.metrics import BLEU


class PerceiverValidator:
    """Comprehensive validation for trained Perceiver model"""

    def __init__(self, model_checkpoint, device="cuda"):
        self.device = device
        print(f"🔍 Loading model from: {model_checkpoint}")

        # Load model
        checkpoint = torch.load(model_checkpoint, map_location=device)
        self.model = SONARwithT5(
            sonar_checkpoint=None,  # Already included in model checkpoint
            freeze_encoder=False,  # Will load frozen state from checkpoint
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained("t5-small")

        print(f"✅ Model loaded successfully!")
        print(f"   Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"   Best BLEU: {checkpoint.get('best_bleu', 'unknown'):.2f}%")

    def load_validation_data(self, features_dir, manifest_path):
        """Load validation dataset"""
        print(f"\n📂 Loading validation data...")

        samples = []
        with open(manifest_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    video_id = parts[0]
                    text = parts[1]

                    # Load feature
                    feature_path = os.path.join(features_dir, f"{video_id}.pt")
                    if os.path.exists(feature_path):
                        feature = torch.load(feature_path, map_location="cpu")
                        samples.append(
                            {
                                "video_id": video_id,
                                "feature": feature,
                                "reference": text,
                            }
                        )

        print(f"✅ Loaded {len(samples)} validation samples")
        return samples

    def generate_translations(self, samples, max_samples=None):
        """Generate translations for all samples"""
        print(f"\n🔄 Generating translations...")

        if max_samples:
            samples = samples[:max_samples]

        translations = []
        references = []

        with torch.no_grad():
            for sample in tqdm(samples, desc="Translating"):
                # Prepare input
                feature = sample["feature"].unsqueeze(0).to(self.device)  # (1, 1024)

                # Generate
                outputs = self.model.generate(
                    sonar_features=feature,
                    max_length=128,
                    num_beams=5,
                    early_stopping=True,
                )

                # Decode
                translation = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )

                translations.append(translation)
                references.append(sample["reference"])

        return translations, references

    def compute_bleu(self, translations, references):
        """Compute BLEU score"""
        bleu = BLEU()
        score = bleu.corpus_score(translations, [references])
        return score.score

    def detect_mode_collapse(self, translations, threshold=0.5):
        """
        Detect mode collapse by checking translation diversity
        Returns:
            - collapse_ratio: % of duplicate translations
            - most_common: Most repeated translation
            - is_collapsed: True if mode collapse detected
        """
        print(f"\n🔍 Checking for mode collapse...")

        # Count unique translations
        translation_counts = Counter(translations)
        total = len(translations)
        unique = len(translation_counts)

        # Most common translation
        most_common, most_common_count = translation_counts.most_common(1)[0]
        collapse_ratio = most_common_count / total

        # Detect collapse
        is_collapsed = collapse_ratio > threshold

        print(f"   Total translations: {total}")
        print(f"   Unique translations: {unique}")
        print(f"   Diversity ratio: {unique/total*100:.1f}%")
        print(
            f"   Most common: '{most_common}' ({most_common_count} times, {collapse_ratio*100:.1f}%)"
        )

        if is_collapsed:
            print(f"   ⚠️  MODE COLLAPSE DETECTED! (>{threshold*100:.0f}% duplicates)")
        else:
            print(f"   ✅ Good diversity (no mode collapse)")

        return {
            "collapse_ratio": collapse_ratio,
            "most_common": most_common,
            "most_common_count": most_common_count,
            "unique_count": unique,
            "total_count": total,
            "is_collapsed": is_collapsed,
        }

    def analyze_translation_quality(self, translations, references, num_samples=10):
        """Show sample translations for qualitative analysis"""
        print(f"\n📝 Sample Translations (first {num_samples}):")
        print("=" * 100)

        for i in range(min(num_samples, len(translations))):
            print(f"\n[Sample {i+1}]")
            print(f"Reference:   {references[i]}")
            print(f"Translation: {translations[i]}")
            print("-" * 100)

    def compute_length_statistics(self, translations, references):
        """Analyze length statistics"""
        print(f"\n📏 Length Statistics:")

        trans_lengths = [len(t.split()) for t in translations]
        ref_lengths = [len(r.split()) for r in references]

        print(
            f"   Translations - Mean: {np.mean(trans_lengths):.1f}, "
            f"Std: {np.std(trans_lengths):.1f}, "
            f"Min: {np.min(trans_lengths)}, "
            f"Max: {np.max(trans_lengths)}"
        )

        print(
            f"   References   - Mean: {np.mean(ref_lengths):.1f}, "
            f"Std: {np.std(ref_lengths):.1f}, "
            f"Min: {np.min(ref_lengths)}, "
            f"Max: {np.max(ref_lengths)}"
        )

        return {
            "translation_mean": float(np.mean(trans_lengths)),
            "translation_std": float(np.std(trans_lengths)),
            "reference_mean": float(np.mean(ref_lengths)),
            "reference_std": float(np.std(ref_lengths)),
        }

    def save_results(self, output_dir, results):
        """Save validation results to file"""
        os.makedirs(output_dir, exist_ok=True)

        # Save summary
        summary_path = os.path.join(output_dir, "validation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Saved summary to: {summary_path}")

        # Save translations
        translations_path = os.path.join(output_dir, "translations.txt")
        with open(translations_path, "w") as f:
            for i, (trans, ref) in enumerate(
                zip(results["translations"], results["references"])
            ):
                f.write(f"[Sample {i+1}]\n")
                f.write(f"Reference:   {ref}\n")
                f.write(f"Translation: {trans}\n")
                f.write("-" * 100 + "\n\n")
        print(f"💾 Saved translations to: {translations_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate Perceiver model post-training"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (best_model.pt)",
    )
    parser.add_argument(
        "--features",
        type=str,
        required=True,
        help="Path to validation features directory",
    )
    parser.add_argument(
        "--manifest", type=str, required=True, help="Path to validation manifest file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max samples to validate (None = all)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    print("=" * 100)
    print("🚀 PERCEIVER RESAMPLER VALIDATION")
    print("=" * 100)

    # Initialize validator
    validator = PerceiverValidator(args.checkpoint, args.device)

    # Load data
    samples = validator.load_validation_data(args.features, args.manifest)

    # Generate translations
    translations, references = validator.generate_translations(
        samples, args.max_samples
    )

    # Compute BLEU
    print(f"\n📊 Computing BLEU score...")
    bleu_score = validator.compute_bleu(translations, references)
    print(f"   BLEU: {bleu_score:.2f}%")

    # Check mode collapse
    collapse_stats = validator.detect_mode_collapse(translations)

    # Length statistics
    length_stats = validator.compute_length_statistics(translations, references)

    # Show samples
    validator.analyze_translation_quality(translations, references, num_samples=10)

    # Prepare results
    results = {
        "bleu_score": float(bleu_score),
        "mode_collapse": collapse_stats,
        "length_statistics": length_stats,
        "num_samples": len(translations),
        "translations": translations,
        "references": references,
    }

    # Save results
    validator.save_results(args.output, results)

    # Final summary
    print("\n" + "=" * 100)
    print("✅ VALIDATION COMPLETE!")
    print("=" * 100)
    print(f"📊 BLEU Score: {bleu_score:.2f}%")
    print(
        f"🎯 Mode Collapse: {'⚠️  DETECTED' if collapse_stats['is_collapsed'] else '✅ None'}"
    )
    print(f"📏 Avg Translation Length: {length_stats['translation_mean']:.1f} words")
    print(f"💾 Results saved to: {args.output}/")
    print("=" * 100)

    # Evaluation verdict
    print("\n🎯 EVALUATION VERDICT:")
    if bleu_score >= 10:
        print("   ✅ EXCELLENT! Perceiver architecture successful!")
    elif bleu_score >= 5:
        print("   ✅ GOOD! Significant improvement over baseline")
    elif bleu_score >= 3:
        print("   ⚠️  MODERATE. Some improvement but room for optimization")
    else:
        print(
            "   ❌ POOR. Consider architecture modifications or hyperparameter tuning"
        )

    if collapse_stats["is_collapsed"]:
        print(
            "   ⚠️  Mode collapse detected - model needs more training or regularization"
        )
    else:
        print("   ✅ Good translation diversity")

    print("\n")


if __name__ == "__main__":
    main()
