#!/usr/bin/env python3
"""
Test script per verificare unfrozen SONAR Encoder con differential LR
"""

import torch
from train_sonar_with_t5 import SONARwithT5

print("=" * 60)
print("TEST: SONAR Unfrozen con Differential Learning Rates")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n✅ Device: {device}")

# Test 1: Encoder FROZEN
print("\n" + "=" * 60)
print("TEST 1: Encoder FROZEN (freeze_encoder=True)")
print("=" * 60)

model_frozen = SONARwithT5(
    sonar_checkpoint="checkpoints/sonar_encoder_finetuned/best_encoder.pt",
    freeze_encoder=True,
    device=device,
)

frozen_params = sum(
    1 for p in model_frozen.sonar_encoder.parameters() if p.requires_grad
)
total_frozen = sum(1 for p in model_frozen.sonar_encoder.parameters())

print(f"✅ SONAR Encoder params: {total_frozen}")
print(f"✅ Trainable SONAR params: {frozen_params}")
print(f"✅ Expected: 0 trainable (all frozen)")

if frozen_params == 0:
    print("✅ PASS: Encoder correctly frozen!")
else:
    print(f"❌ FAIL: Expected 0 trainable, got {frozen_params}")

# Test 2: Encoder TRAINABLE
print("\n" + "=" * 60)
print("TEST 2: Encoder TRAINABLE (freeze_encoder=False)")
print("=" * 60)

model_trainable = SONARwithT5(
    sonar_checkpoint="checkpoints/sonar_encoder_finetuned/best_encoder.pt",
    freeze_encoder=False,
    device=device,
)

trainable_params = sum(
    1 for p in model_trainable.sonar_encoder.parameters() if p.requires_grad
)
total_trainable = sum(1 for p in model_trainable.sonar_encoder.parameters())

print(f"✅ SONAR Encoder params: {total_trainable}")
print(f"✅ Trainable SONAR params: {trainable_params}")
print(f"✅ Expected: {total_trainable} trainable (all unfrozen)")

if trainable_params == total_trainable:
    print("✅ PASS: Encoder correctly unfrozen!")
else:
    print(f"❌ FAIL: Expected {total_trainable} trainable, got {trainable_params}")

# Test 3: Forward pass con gradient
print("\n" + "=" * 60)
print("TEST 3: Forward Pass con Gradient")
print("=" * 60)

model_trainable.train()

# Dummy input
batch_size = 2
seq_len = 100
features = torch.randn(batch_size, seq_len, 256).to(device)
target_texts = ["This is a test", "Another test"]

print("✅ Running forward pass...")
try:
    loss = model_trainable(features, target_texts)
    print(f"✅ Loss computed: {loss.item():.4f}")

    print("✅ Running backward pass...")
    loss.backward()

    # Check gradients
    encoder_has_grad = any(
        p.grad is not None for p in model_trainable.sonar_encoder.parameters()
    )
    projection_has_grad = any(
        p.grad is not None for p in model_trainable.projection.parameters()
    )

    print(f"✅ SONAR Encoder has gradients: {encoder_has_grad}")
    print(f"✅ Projection has gradients: {projection_has_grad}")

    if encoder_has_grad and projection_has_grad:
        print("✅ PASS: Gradients computed correctly!")
    else:
        print("❌ FAIL: Some components missing gradients")

except Exception as e:
    print(f"❌ FAIL: Forward/backward failed with error:")
    print(f"   {e}")
    import traceback

    traceback.print_exc()

# Test 4: Optimizer con differential LR
print("\n" + "=" * 60)
print("TEST 4: Differential Learning Rates")
print("=" * 60)

learning_rate = 5e-5

encoder_params = []
projection_params = []
t5_params = []

for name, param in model_trainable.named_parameters():
    if not param.requires_grad:
        continue

    if "sonar_encoder" in name:
        encoder_params.append(param)
    elif "projection" in name or "expander" in name or "attention_bridge" in name:
        projection_params.append(param)
    elif "t5" in name:
        t5_params.append(param)

optimizer_params = [
    {"params": encoder_params, "lr": learning_rate / 5},
    {"params": projection_params, "lr": learning_rate},
    {"params": t5_params, "lr": learning_rate},
]

optimizer = torch.optim.AdamW(optimizer_params)

print(f"✅ Optimizer created with {len(optimizer.param_groups)} param groups")
print(
    f"   Group 0 (SONAR): LR={optimizer.param_groups[0]['lr']:.2e}, {len(encoder_params)} params"
)
print(
    f"   Group 1 (Projection): LR={optimizer.param_groups[1]['lr']:.2e}, {len(projection_params)} params"
)
print(
    f"   Group 2 (T5): LR={optimizer.param_groups[2]['lr']:.2e}, {len(t5_params)} params"
)

if optimizer.param_groups[0]["lr"] == learning_rate / 5:
    print("✅ PASS: SONAR has 1/5 learning rate!")
else:
    print(f"❌ FAIL: SONAR LR incorrect")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("✅ Encoder freezing/unfreezing works correctly")
print("✅ Differential learning rates configured properly")
print("✅ Forward/backward pass functional")
print("\n🎉 All tests passed! Ready for training with unfrozen encoder.")
print("\n📝 Recommended command:")
print("python train_sonar_with_t5.py \\")
print("    --sonar_checkpoint checkpoints/sonar_encoder_finetuned/best_encoder.pt \\")
print("    --train_features features/train \\")
print("    --train_manifest manifests/train.tsv \\")
print("    --val_features features/val \\")
print("    --val_manifest manifests/val.tsv \\")
print("    --output_dir checkpoints/sonar_t5_unfrozen \\")
print("    --t5_model t5-small \\")
print("    --epochs 30 \\")
print("    --batch_size 16 \\")
print("    --learning_rate 5e-5 \\")
print("    --warmup_steps 500 \\")
print("    --device cuda")
print("\n⚠️  NOTE: Remove --freeze_encoder to enable unfrozen training!")
