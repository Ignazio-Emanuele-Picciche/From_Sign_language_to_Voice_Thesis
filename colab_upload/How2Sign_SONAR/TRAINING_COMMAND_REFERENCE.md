# Quick Reference: Training SONAR+T5

## Required Directory Structure on Colab

```
/content/drive/MyDrive/How2Sign_SONAR/
├── train_sonar_with_t5.py          # Training script (UPDATED with attention bridge)
├── test_attention_fix.py           # Validation script
├── checkpoints/
│   └── sonar_encoder_finetuned/
│       └── best_encoder.pt         # Your fine-tuned SONAR encoder
├── manifests/
│   ├── train.tsv                   # Training manifest (id, duration, text)
│   └── val.tsv                     # Validation manifest (id, duration, text)
└── features/
    ├── train/                      # Training features
    │   ├── video1.npy             # SignHiera features (T, 256)
    │   ├── video2.npy
    │   └── ...
    └── val/                        # Validation features
        ├── video1.npy
        └── ...
```

## Command Structure

The training script requires these arguments:

```bash
python train_sonar_with_t5.py \
    --sonar_checkpoint <path_to_checkpoint>    # Required
    --train_features <train_features_dir>      # Required
    --train_manifest <train_manifest.tsv>      # Required
    --val_features <val_features_dir>          # Required
    --val_manifest <val_manifest.tsv>          # Required
    --output_dir <checkpoint_output_dir>       # Optional (default: checkpoints/sonar_t5)
    --t5_model <t5-small|t5-base|t5-large>    # Optional (default: t5-small)
    --freeze_encoder                           # Optional flag (recommended)
    --epochs <num_epochs>                      # Optional (default: 10)
    --batch_size <batch_size>                  # Optional (default: 16)
    --learning_rate <lr>                       # Optional (default: 1e-4)
    --warmup_steps <steps>                     # Optional (default: 500)
    --max_samples <num>                        # Optional (for testing)
    --device <cuda|cpu>                        # Optional (default: cuda)
```

## Example: Quick 2-Epoch Test

```bash
python train_sonar_with_t5.py \
    --sonar_checkpoint checkpoints/sonar_encoder_finetuned/best_encoder.pt \
    --train_features features/train \
    --train_manifest manifests/train.tsv \
    --val_features features/val \
    --val_manifest manifests/val.tsv \
    --output_dir checkpoints/sonar_t5_attention_test \
    --epochs 2 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --warmup_steps 100
```

## Example: Full 20-Epoch Training

```bash
python train_sonar_with_t5.py \
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
    --warmup_steps 500
```

## Manifest File Format (.tsv)

The manifest files should be tab-separated with at least these columns:

```
id                              duration    text
video_id_1                      10.5        This is the English translation.
video_id_2                      15.2        Another translation here.
...
```

The script will automatically:

- Detect which column contains IDs (looks for "id" or "name")
- Detect which column contains text (looks for "text", "translation", or "sentence")
- Match feature files by looking for `{id}.npy` or `{id}.pt` in the features directory

## Feature Files Format

Each feature file should be a NumPy array saved as `.npy` or PyTorch tensor as `.pt`:

- **Shape**: `(T, 256)` where T is the number of frames
- **Type**: float32
- **Content**: SignHiera video features extracted from sign language videos

Example:

```python
import numpy as np
features = np.random.randn(100, 256)  # 100 frames, 256 dims
np.save('features/train/video_id_1.npy', features)
```

## Troubleshooting

### Error: "the following arguments are required"

- Make sure you're providing all 5 required arguments
- Check paths are correct (relative to where you run the command)

### Error: "Found 0 samples with features"

- Check that feature filenames match the IDs in the manifest
- Verify feature files are in the correct directory
- Make sure files have `.npy` or `.pt` extension

### Error: "shape '[X, Y, Z]' is invalid"

- This should be fixed with the attention bridge update
- Make sure you're using the latest `train_sonar_with_t5.py` with the fix

### Mode Collapse (all outputs identical)

- This should be improved with the attention bridge
- If it persists, try:
  - Increase sequence length to 64 tokens
  - Add more warmup steps (1000+)
  - Reduce learning rate to 1e-5
  - Train longer (30-50 epochs)

## Output

The script will create:

```
checkpoints/sonar_t5_attention_final/
├── best_model.pt              # Best model by validation BLEU
├── last_model.pt              # Last epoch model
└── training_log.txt           # Training metrics
```

Each checkpoint contains:

```python
{
    'epoch': <epoch_number>,
    'model_state_dict': <model_weights>,
    'optimizer_state_dict': <optimizer_state>,
    'scheduler_state_dict': <scheduler_state>,
    'best_bleu': <best_bleu_score>,
    'train_loss': <training_loss>,
    'val_loss': <validation_loss>,
    'val_bleu': <validation_bleu>
}
```
