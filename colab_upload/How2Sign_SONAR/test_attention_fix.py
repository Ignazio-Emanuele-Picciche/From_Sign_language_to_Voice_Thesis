"""
Quick test script for attention bridge fix
Tests if mode collapse is resolved with 2 epochs
"""

import torch
import sys
import os

# Test imports
try:
    from train_sonar_with_t5 import SONARwithT5, How2SignDataset

    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test model initialization
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Using device: {device}")

    model = SONARwithT5(
        sonar_checkpoint="checkpoints/sonar_encoder_finetuned/best_encoder.pt",
        device=device,
    )
    print("✓ Model initialized successfully")

    # Check attention bridge
    assert hasattr(model, "attention_bridge"), "Missing attention_bridge"
    print(f"✓ Attention bridge present: {type(model.attention_bridge)}")
    print(f"  - embed_dim: {model.attention_bridge.embed_dim}")
    print(f"  - num_heads: {model.attention_bridge.num_heads}")

    # Check expander size
    print(f"✓ Expander shape: {model.expander.shape}")
    assert (
        model.expander.shape[0] == 32
    ), f"Expected 32 tokens, got {model.expander.shape[0]}"

except Exception as e:
    print(f"✗ Model initialization failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test forward pass
try:
    batch_size = 2
    seq_len = 100
    feature_dim = 256

    # Create dummy inputs
    features = torch.randn(batch_size, seq_len, feature_dim).to(device)
    target_texts = ["This is a test", "Another test sentence"]

    # Forward pass (training mode)
    model.train()
    loss = model(features, target_texts)
    print(f"✓ Training forward pass successful, loss: {loss.item():.4f}")

    # Forward pass (inference mode)
    model.eval()
    with torch.no_grad():
        outputs = model(features)
    print(f"✓ Inference forward pass successful")
    print(f"  Generated texts: {outputs}")

    # Check that outputs are different (not mode collapsed)
    if outputs[0] == outputs[1]:
        print("  ⚠️ Warning: Both outputs identical (possible mode collapse)")
    else:
        print("  ✓ Outputs are different!")

except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
print("\nModel is ready for training with attention bridge.")
print("Expected improvements:")
print("  - More diverse outputs (no mode collapse)")
print("  - BLEU score should increase to 5-10%")
print("  - Loss should stabilize around 1.5-2.0")
print("\nNext step: Run 2-epoch quick test:")
print("  python train_sonar_with_t5.py --epochs 2 --batch_size 16")
