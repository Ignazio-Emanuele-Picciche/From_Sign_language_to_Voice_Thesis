# COLAB Training Guide - Attention Bridge Fix for Mode Collapse

## Problem Solved

**Mode Collapse**: T5 was generating the same output for all inputs ("And I'm going to make sure that I don't have a problem with it")

**Root Cause**: Using `inputs_embeds` without proper cross-attention meant T5 couldn't "see" the SONAR embedding information

**Solution**: Added `nn.MultiheadAttention` attention bridge that allows learnable query tokens to attend to SONAR embeddings

---

## Architecture Changes

### Before (Mode Collapse):

```python
# Simple addition - no attention
t5_embedding_expanded = t5_embedding.repeat(1, 8, 1)  # (B, 8, 512)
expander_batch = self.expander.expand(B, -1, -1)
t5_embedding_final = t5_embedding_expanded + expander_batch  # Just add them
```

### After (Attention Bridge):

```python
# Cross-attention mechanism
self.attention_bridge = nn.MultiheadAttention(embed_dim=512, num_heads=8)

# Forward pass:
query = self.expander.expand(B, -1, -1)  # (B, 32, 512) learnable queries
key_value = t5_embedding.unsqueeze(1)    # (B, 1, 512) SONAR info

attended_output, _ = self.attention_bridge(
    query=query,
    key=key_value,
    value=key_value
)  # Query tokens attend to SONAR embedding!
```

### Additional Improvements:

- **Sequence length**: 8 → 32 tokens (more capacity)
- **Diversity mechanisms**: Added `repetition_penalty=1.5`, `temperature=0.8`, `top_p=0.9`
- **Sampling**: Enabled `do_sample=True` during generation

---

## Expected Improvements

| Metric           | Before        | Expected After         |
| ---------------- | ------------- | ---------------------- |
| BLEU Score       | 1.54%         | 5-10% (hopefully more) |
| Output Diversity | All identical | Different per input    |
| Mode Collapse    | Yes ❌        | No ✓                   |

---

## Colab Training Steps

### Cell 1: Setup

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/How2Sign_SONAR')
!pwd
```

### Cell 2: Install Dependencies

```python
!pip install transformers datasets sacrebleu torch tqdm
```

### Cell 3: Verify Files

```python
import os

required_files = [
    'train_sonar_with_t5.py',
    'checkpoints/sonar_encoder_finetuned/best_encoder.pt',
    'manifests/train.tsv',
    'manifests/val.tsv',
    'features/train',  # Directory with .npy or .pt files
    'features/val'     # Directory with .npy or .pt files
]

for f in required_files:
    exists = os.path.exists(f)
    if os.path.isdir(f):
        # Count files in directory
        try:
            count = len([x for x in os.listdir(f) if x.endswith(('.npy', '.pt'))])
            print(f"{'✓' if exists else '✗'} {f} ({count} feature files)")
        except:
            print(f"{'✓' if exists else '✗'} {f}")
    else:
        print(f"{'✓' if exists else '✗'} {f}")
```

### Cell 4: Test Architecture (NEW!)

```python
# Validate attention bridge is working
!python test_attention_fix.py
```

### Cell 5: Quick 2-Epoch Test (RECOMMENDED FIRST!)

```python
# Test if mode collapse is fixed before full training
!python train_sonar_with_t5.py \
    --sonar_checkpoint checkpoints/sonar_encoder_finetuned/best_encoder.pt \
    --train_features features/train \
    --train_manifest manifests/train.tsv \
    --val_features features/val \
    --val_manifest manifests/val.tsv \
    --output_dir checkpoints/sonar_t5_attention_test \
    --t5_model t5-small \
    --freeze_encoder \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --warmup_steps 100 \
    --device cuda
```

**⚠️ CHECK THE OUTPUT:**

- Are the 5 sample predictions **different** from each other?
- Is BLEU score **>2%** (ideally 5-10%)?
- If YES → proceed to full training (Cell 6)
- If NO → mode collapse persists, need more fixes

### Cell 6: Full 20-Epoch Training (ONLY IF Cell 5 SUCCEEDS!)

```python
!python train_sonar_with_t5.py \
    --sonar_checkpoint checkpoints/sonar_encoder_finetuned/best_encoder.pt \
    --train_features features/train \
    --train_manifest manifests/train.tsv \
    --val_features features/val \
    --val_manifest manifests/val.tsv \
    --output_dir checkpoints/sonar_t5_attention_final \
    --t5_model t5-small \
    --freeze_encoder \
    --epochs 20 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --warmup_steps 500 \
    --device cuda
```

### Cell 7: Evaluate

```python
# Load best checkpoint and evaluate on validation set
import torch
from train_sonar_with_t5 import SONARwithT5, evaluate_model, How2SignDataset
from torch.utils.data import DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model with best checkpoint
model = SONARwithT5(
    sonar_checkpoint="checkpoints/sonar_encoder_finetuned/best_encoder.pt",
    device=device
)
checkpoint = torch.load("checkpoints/sonar_t5_attention_final/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate
val_dataset = How2SignDataset("data/How2Sign_val.json")
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

bleu, loss, predictions = evaluate_model(model, val_loader, device)
print(f"\nFinal Results:")
print(f"  BLEU: {bleu:.2f}%")
print(f"  Loss: {loss:.4f}")
print(f"\nSample Predictions:")
for i, (gt, pred) in enumerate(predictions[:5]):
    print(f"\n{i+1}. Ground Truth: {gt}")
    print(f"   Prediction:    {pred}")
```

### Cell 8: Download Checkpoint

```python
from google.colab import files

# Download best model
files.download('checkpoints/sonar_t5_attention_final/best_model.pt')

# Download logs
files.download('logs/train_attention_final.log')
```

---

## What Changed in train_sonar_with_t5.py

### 1. Added Attention Bridge in `__init__`:

```python
# Line ~310
self.attention_bridge = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.1,
    batch_first=False  # Expects (seq, batch, dim)
)
```

### 2. Increased Sequence Length:

```python
# Line ~308
self.expander = nn.Parameter(torch.randn(32, 512))  # Was 8, now 32
```

### 3. Updated Forward Pass:

```python
# Lines ~350-365
# Create query tokens from learnable expander
query = self.expander.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 32, 512)

# Use SONAR embedding as key and value
key_value = t5_embedding.unsqueeze(1)  # (B, 1, 512)

# Apply cross-attention
query_t = query.transpose(0, 1)  # (32, B, 512)
key_value_t = key_value.transpose(0, 1)  # (1, B, 512)

attended_output, _ = self.attention_bridge(
    query=query_t,
    key=key_value_t,
    value=key_value_t
)

t5_embedding_final = attended_output.transpose(0, 1)  # (B, 32, 512)
```

### 4. Added Diversity to Generation:

```python
# Lines ~390-398
generated_ids = self.t5.generate(
    inputs_embeds=t5_embedding_final,
    max_length=max_length,
    num_beams=4,
    early_stopping=True,
    no_repeat_ngram_size=2,
    repetition_penalty=1.5,  # NEW
    temperature=0.8,          # NEW
    top_p=0.9,               # NEW
    do_sample=True,          # NEW
)
```

---

## Troubleshooting

### If mode collapse persists after Cell 5:

**Symptoms**: All 5 predictions still identical, BLEU <2%

**Try these in order**:

1. **Increase sequence length to 64**:

   ```python
   # Line 308
   self.expander = nn.Parameter(torch.randn(64, 512))
   ```

2. **Make attention bridge deeper**:

   ```python
   self.attention_bridge_1 = nn.MultiheadAttention(embed_dim=512, num_heads=8)
   self.attention_bridge_2 = nn.MultiheadAttention(embed_dim=512, num_heads=8)
   # Apply twice in forward pass
   ```

3. **Add LayerNorm after attention**:

   ```python
   self.attention_norm = nn.LayerNorm(512)
   # After attention:
   t5_embedding_final = self.attention_norm(attended_output.transpose(0, 1))
   ```

4. **Try prefix tuning approach** (alternative architecture):

   ```python
   # Instead of attention bridge, prepend learned prefix to T5 decoder
   # See IMPROVEMENTS_T5.md for details
   ```

5. **Unfreeze SONAR encoder** (last resort):
   ```python
   # Line 300
   # Comment out: for param in self.sonar_encoder.parameters(): param.requires_grad = False
   ```

### If BLEU improves but still <10%:

- Increase training epochs to 50
- Try T5-base instead of T5-small (larger model)
- Collect more training data if possible
- Check data quality (are ground truth translations correct?)

---

## Success Criteria

✓ **MINIMUM (Cell 5)**: BLEU >2%, diverse outputs  
✓ **GOOD (Cell 6)**: BLEU 5-10%, coherent translations  
✓ **EXCELLENT (Cell 6)**: BLEU 10-18%, accurate translations  
✓ **IDEAL (Cell 6)**: BLEU >18%, near-human quality

Remember: Even 5-10% BLEU is a **huge improvement** from 1.54% and proves the encoder is learning meaningful representations!

---

## Questions?

Check:

- `IMPROVEMENTS_T5.md` - Debugging guide
- `FIX_FAIRSEQ2_DEFINITIVO.md` - Why we can't use SONAR decoder
- `test_attention_fix.py` - Architecture validation script
