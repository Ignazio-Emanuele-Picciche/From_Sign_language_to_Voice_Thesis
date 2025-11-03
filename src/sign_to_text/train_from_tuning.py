"""
Train Sign-to-Text model using best hyperparameters from Optuna tuning
======================================================================

Carica automaticamente i migliori parametri da results/best_hyperparameters.json
e lancia training completo.

Usage:
    python src/sign_to_text/train_from_tuning.py

    # Override epochs
    python src/sign_to_text/train_from_tuning.py --epochs 100
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train con best params da tuning")
    parser.add_argument("--epochs", type=int, default=50, help="Override num epochs")
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )
    parser.add_argument(
        "--best_params_file",
        type=str,
        default="results/best_hyperparameters.json",
        help="Path to best hyperparameters JSON",
    )
    args = parser.parse_args()

    # Load best params
    best_params_path = Path(args.best_params_file)

    if not best_params_path.exists():
        print(f"❌ Best parameters file not found: {best_params_path}")
        print(f"\n💡 Run hyperparameter tuning first:")
        print(f"   ./run_tuning.sh")
        print(f"   OR")
        print(f"   .venv/bin/python src/sign_to_text/tune.py --n_trials 20 --epochs 5")
        sys.exit(1)

    with open(best_params_path, "r") as f:
        best_config = json.load(f)

    params = best_config["params"]

    print("\n" + "=" * 80)
    print("🚀 TRAINING WITH BEST HYPERPARAMETERS")
    print("=" * 80)
    print(f"  Source: {best_params_path}")
    print(f"  Trial: {best_config.get('trial_number', 'N/A')}")
    print(
        f"  Best {best_config.get('optimize_metric', 'N/A')}: {best_config.get('best_value', 'N/A'):.4f}"
    )
    print("\n  Hyperparameters:")
    for key, value in params.items():
        print(f"    {key}: {value}")
    print(f"\n  Training config:")
    print(f"    epochs: {args.epochs}")
    print(f"    patience: {args.patience}")
    print("=" * 80 + "\n")

    # Build command
    cmd = [
        ".venv/bin/python",
        "src/sign_to_text/train.py",
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(params["batch_size"]),
        "--d_model",
        str(params["d_model"]),
        "--nhead",
        str(params["nhead"]),
        "--num_encoder_layers",
        str(params["num_encoder_layers"]),
        "--num_decoder_layers",
        str(params["num_decoder_layers"]),
        "--lr",
        str(params["lr"]),
        "--patience",
        str(args.patience),
    ]

    # Optional: add dim_feedforward, dropout se nel train.py
    # (attualmente train.py non ha questi args, ma potrebbero essere aggiunti)

    print(f"🏃 Running command:")
    print(f"  {' '.join(cmd)}\n")

    # Run training
    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✅ Training completato con successo!")
    else:
        print(f"\n❌ Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == "__main__":
    main()
