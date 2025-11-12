"""
Test SONAR Model Loading
=========================

Test if SONAR models can be loaded without crashing.
"""

import torch
import sys
from pathlib import Path


def test_model_loading():
    """Test loading SONAR models."""

    print("=" * 80)
    print("🧪 SONAR MODEL LOADING TEST")
    print("=" * 80)

    # Check CUDA availability
    print(f"\n1️⃣  Device check:")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA device: {torch.cuda.get_device_name(0)}")
        print(
            f"   CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Model paths
    signhiera_path = Path("../../models/pretrained_ssvp/dm_70h_ub_signhiera.pth")
    encoder_path = Path("../../models/pretrained_ssvp/dm_70h_ub_sonar_encoder.pth")

    print(f"\n2️⃣  Model files:")
    print(f"   SignHiera: {signhiera_path.exists()} ({signhiera_path})")
    print(f"   Encoder:   {encoder_path.exists()} ({encoder_path})")

    if not signhiera_path.exists() or not encoder_path.exists():
        print("\n❌ Model files not found!")
        return False

    # Test loading SignHiera (CPU only to avoid crash)
    print(f"\n3️⃣  Loading SignHiera (CPU)...")
    try:
        signhiera_ckpt = torch.load(signhiera_path, map_location="cpu")
        print(f"   ✓ SignHiera loaded")
        print(f"   Keys: {list(signhiera_ckpt.keys())[:5]}...")

        # Check size
        total_params = sum(
            p.numel() for p in signhiera_ckpt.values() if isinstance(p, torch.Tensor)
        )
        print(f"   Parameters: {total_params:,}")

        # Free memory
        del signhiera_ckpt
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"   ❌ Failed to load SignHiera: {e}")
        return False

    # Test loading SONAR Encoder
    print(f"\n4️⃣  Loading SONAR Encoder (CPU)...")
    try:
        encoder_ckpt = torch.load(encoder_path, map_location="cpu")
        print(f"   ✓ SONAR Encoder loaded")
        print(f"   Keys: {list(encoder_ckpt.keys())[:5]}...")

        # Free memory
        del encoder_ckpt
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"   ❌ Failed to load SONAR Encoder: {e}")
        return False

    print(f"\n✅ MODELS CAN BE LOADED SUCCESSFULLY!")
    print(f"\n💡 Next step: Try inference on CPU (slower but won't crash)")

    return True


if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
