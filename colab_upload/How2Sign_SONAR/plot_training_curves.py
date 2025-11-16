"""
Plot Training Curves - Simple Post-Training Visualization
Reads training logs and plots Loss + BLEU curves
"""

import json
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def plot_training_curves(checkpoint_dir, output_dir=None):
    """
    Plot training curves from training_log.json

    Args:
        checkpoint_dir: Directory with training_log.json
        output_dir: Where to save plots (default: checkpoint_dir)
    """

    # Load training log
    log_path = Path(checkpoint_dir) / "training_log.json"
    if not log_path.exists():
        print(f"❌ Training log not found: {log_path}")
        return

    print(f"📂 Loading training log from: {log_path}")
    with open(log_path, "r") as f:
        log_data = json.load(f)

    # Extract data
    epochs = [entry["epoch"] for entry in log_data]
    train_loss = [entry["train_loss"] for entry in log_data]
    val_loss = [entry["val_loss"] for entry in log_data]
    bleu_scores = [entry["bleu"] for entry in log_data]

    print(f"✅ Loaded {len(epochs)} epochs of training data")
    print(
        f"   Best BLEU: {max(bleu_scores):.2f}% at epoch {bleu_scores.index(max(bleu_scores))+1}"
    )

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Loss curves
    ax1.plot(epochs, train_loss, "b-", label="Train Loss", linewidth=2, marker="o")
    ax1.plot(epochs, val_loss, "r-", label="Val Loss", linewidth=2, marker="s")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training and Validation Loss", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: BLEU score
    ax2.plot(epochs, bleu_scores, "g-", label="BLEU Score", linewidth=2, marker="D")
    ax2.axhline(
        y=max(bleu_scores),
        color="g",
        linestyle="--",
        alpha=0.5,
        label=f"Best: {max(bleu_scores):.2f}%",
    )
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("BLEU Score (%)", fontsize=12)
    ax2.set_title("BLEU Score Evolution", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    if output_dir is None:
        output_dir = checkpoint_dir
    output_path = Path(output_dir) / "training_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"💾 Saved plot to: {output_path}")

    # Show plot
    plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("📊 TRAINING SUMMARY")
    print("=" * 60)
    print(f"Total Epochs: {len(epochs)}")
    print(f"Final Train Loss: {train_loss[-1]:.4f}")
    print(f"Final Val Loss: {val_loss[-1]:.4f}")
    print(f"Final BLEU: {bleu_scores[-1]:.2f}%")
    print(
        f"Best BLEU: {max(bleu_scores):.2f}% (Epoch {bleu_scores.index(max(bleu_scores))+1})"
    )
    print(
        f"BLEU Improvement: {bleu_scores[-1] - bleu_scores[0]:.2f}% (Epoch 1 → {len(epochs)})"
    )
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from logs")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Directory containing training_log.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as checkpoint_dir)",
    )

    args = parser.parse_args()

    print("📈 PLOTTING TRAINING CURVES")
    print("=" * 60)
    plot_training_curves(args.checkpoint_dir, args.output_dir)


if __name__ == "__main__":
    main()
