"""
Quick test del training pipeline
=================================

Test rapido (1 epoch) per verificare che tutto funzioni.
"""

import subprocess
import sys

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🧪 QUICK TRAINING TEST (1 epoch)")
    print("=" * 80)
    print("\nTesting training pipeline con:")
    print("  - 1 epoch")
    print("  - batch_size=4 (small)")
    print("  - d_model=128 (small)")
    print("\n")

    cmd = [
        sys.executable,
        "src/sign_to_text/train.py",
        "--epochs",
        "1",
        "--batch_size",
        "4",
        "--d_model",
        "128",
        "--nhead",
        "4",
        "--num_encoder_layers",
        "2",
        "--num_decoder_layers",
        "2",
        "--patience",
        "5",
    ]

    subprocess.run(cmd, cwd=".")
