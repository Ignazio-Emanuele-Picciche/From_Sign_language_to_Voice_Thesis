"""
Training How2Sign con Best Hyperparameters
===========================================

Lancia training How2Sign usando hyperparameters ottimizzati da Optuna tuning.

Usage:
    # Usa best params da tuning
    python src/sign_to_text/train_how2sign_from_tuning.py --epochs 50
    
    # Override alcuni parametri
    python src/sign_to_text/train_how2sign_from_tuning.py \
        --epochs 30 \
        --batch_size 32
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Train How2Sign with tuned hyperparameters"
    )

    # Path to best params
    parser.add_argument(
        "--best_params_json",
        type=str,
        default="results/how2sign_tuning/best_hyperparameters.json",
        help="Path to best hyperparameters JSON from tuning",
    )

    # Override params (optional)
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/sign_to_text/how2sign_tuned",
        help="Output directory for checkpoints",
    )

    args = parser.parse_args()

    # Load best params
    best_params_path = Path(args.best_params_json)

    if not best_params_path.exists():
        print(f"❌ Best params file not found: {best_params_path}")
        print(f"\n⚠️  Run tuning first:")
        print(f"   python src/sign_to_text/tune_how2sign.py --n_trials 20 --epochs 5")
        sys.exit(1)

    print("=" * 80)
    print("🚀 TRAINING HOW2SIGN WITH TUNED HYPERPARAMETERS")
    print("=" * 80)

    print(f"\n📂 Loading best params from: {best_params_path}")

    with open(best_params_path, "r") as f:
        best_params = json.load(f)

    print(f"\n🏆 Best Hyperparameters:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")

    # Build command
    cmd = [
        sys.executable,  # Python interpreter
        "src/sign_to_text/train_how2sign.py",
        "--output_dir",
        args.output_dir,
    ]

    # Add best params
    for key, value in best_params.items():
        cmd.extend([f"--{key}", str(value)])

    # Override if specified
    if args.epochs is not None:
        # Remove epochs from cmd if present
        try:
            idx = cmd.index("--epochs")
            cmd[idx : idx + 2] = []
        except ValueError:
            pass
        cmd.extend(["--epochs", str(args.epochs)])
        print(f"\n   ⚠️  Override epochs: {args.epochs}")
    else:
        # Default epochs for full training
        cmd.extend(["--epochs", "50"])
        print(f"\n   📅 Using default epochs: 50")

    if args.batch_size is not None:
        try:
            idx = cmd.index("--batch_size")
            cmd[idx : idx + 2] = []
        except ValueError:
            pass
        cmd.extend(["--batch_size", str(args.batch_size)])
        print(f"   ⚠️  Override batch_size: {args.batch_size}")

    if args.lr is not None:
        try:
            idx = cmd.index("--lr")
            cmd[idx : idx + 2] = []
        except ValueError:
            pass
        cmd.extend(["--lr", str(args.lr)])
        print(f"   ⚠️  Override lr: {args.lr}")

    # Add extra params
    cmd.extend(
        [
            "--num_workers",
            "4",
            "--grad_clip",
            "1.0",
        ]
    )

    print(f"\n🎯 Launching training...")
    print(f"   Output: {args.output_dir}")
    print(f"\n{'='*80}\n")

    # Run training
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        sys.exit(1)

    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📁 Checkpoints: {args.output_dir}")
    print(f"\n")


if __name__ == "__main__":
    main()
