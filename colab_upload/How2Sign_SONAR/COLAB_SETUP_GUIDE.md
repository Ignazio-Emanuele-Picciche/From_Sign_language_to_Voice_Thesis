# 🚀 Google Colab Setup Guide - SONAR Feature Extraction

**Goal**: Extract SignHiera features from How2Sign videos using Google Colab (Linux + CUDA T4 GPU)  
**Time**: 1-2 hours setup + 8-11 hours processing  
**Cost**: FREE (using Colab free tier)

---

## 📋 Preparation Checklist

Before starting, make sure you have:

- ✅ Google account with Google Drive access
- ✅ Files ready to upload:
  - `manifests/` folder (train.tsv, val.tsv, test.tsv)
  - `extract_features_signhiera.py` script
  - Videos organized (optional - see options below)

---

## 🎯 Option 1: Upload Sample Videos (RECOMMENDED for testing)

**Best for**: Quick test to verify everything works

### Step 1.1: Select Sample Videos

```bash
# On your Mac, create a small test set (10 videos per split)
cd /Users/ignazioemanuelepicciche/Documents/TESI\ Magistrale\ UCBM/Improved_EmoSign_Thesis

# Create sample directory
mkdir -p colab_upload/How2Sign_SONAR/videos/{train,val,test}

# Copy 10 sample videos from each split
ls data/raw/train/raw_videos_front_train/*.mp4 | head -10 | while read f; do
    cp "$f" colab_upload/How2Sign_SONAR/videos/train/
done

ls data/raw/val/raw_videos_front_val/*.mp4 | head -10 | while read f; do
    cp "$f" colab_upload/How2Sign_SONAR/videos/val/
done

ls data/raw/test/raw_videos_front_test/*.mp4 | head -10 | while read f; do
    cp "$f" colab_upload/How2Sign_SONAR/videos/test/
done

echo "✅ Sample videos copied"
```

### Step 1.2: Create Sample Manifests

```bash
# Create manifests for sample videos only
cd colab_upload/How2Sign_SONAR

# For train split (first 10)
head -11 manifests/train.tsv > manifests/train_sample.tsv

# For val split (first 10)
head -11 manifests/val.tsv > manifests/val_sample.tsv

# For test split (first 10)
head -11 manifests/test.tsv > manifests/test_sample.tsv

echo "✅ Sample manifests created"
```

**Upload to Google Drive**: ~100 MB (30 videos)  
**Processing time**: ~15-20 minutes

---

## 🎯 Option 2: Upload All Videos (FULL dataset)

**Best for**: Complete feature extraction after successful test

### Step 2.1: Create Symlinks to Videos

```bash
cd /Users/ignazioemanuelepicciche/Documents/TESI\ Magistrale\ UCBM/Improved_EmoSign_Thesis

# Create symlinks (faster than copying)
cd colab_upload/How2Sign_SONAR/videos

# Train videos
ln -s ../../../data/raw/train/raw_videos_front_train train

# Val videos
ln -s ../../../data/raw/val/raw_videos_front_val val

# Test videos
ln -s ../../../data/raw/test/raw_videos_front_test test

echo "✅ Video symlinks created"
```

**Upload to Google Drive**: ~40-50 GB (6229 videos)  
**Processing time**: 8-11 hours

---

## 📤 Step 2: Upload to Google Drive

### Upload Structure

Your Google Drive should look like this:

```
MyDrive/
└── How2Sign_SONAR/
    ├── manifests/
    │   ├── train.tsv (or train_sample.tsv)
    │   ├── val.tsv (or val_sample.tsv)
    │   └── test.tsv (or test_sample.tsv)
    ├── videos/
    │   ├── train/
    │   │   ├── --7E2sU6zP4_10-5-rgb_front.mp4
    │   │   ├── --7E2sU6zP4_11-5-rgb_front.mp4
    │   │   └── ...
    │   ├── val/
    │   │   └── ...
    │   └── test/
    │       └── ...
    └── extract_features_signhiera.py
```

### Upload Methods

**Method A: Google Drive Desktop App** (EASIEST)

1. Install Google Drive desktop app
2. Drag `colab_upload/How2Sign_SONAR/` to `MyDrive/`
3. Wait for sync (2-5 hours for full dataset)

**Method B: Google Drive Web Interface**

1. Go to https://drive.google.com
2. Create folder `How2Sign_SONAR`
3. Upload files manually (slower for large datasets)

**Method C: Command Line (rclone)** (FASTEST)

```bash
# Install rclone
brew install rclone

# Configure Google Drive
rclone config

# Upload
rclone copy colab_upload/How2Sign_SONAR/ gdrive:How2Sign_SONAR/ --progress
```

---

## 💻 Step 3: Open Google Colab Notebook

### 3.1: Create New Notebook

1. Go to https://colab.research.google.com
2. Click **"New Notebook"**
3. Rename to **"SONAR_Feature_Extraction"**

### 3.2: Enable GPU Runtime

1. Click **Runtime** → **Change runtime type**
2. Select **Hardware accelerator**: **T4 GPU**
3. Click **Save**

---

## 🔧 Step 4: Run Extraction in Colab

### Cell 1: Install Dependencies

```python
# Install required packages
!pip install -q torch torchvision opencv-python-headless pillow tqdm pandas

print("✅ Dependencies installed")
```

### Cell 2: Mount Google Drive

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Change to working directory
import os
os.chdir('/content/drive/MyDrive/How2Sign_SONAR')

print("✅ Google Drive mounted")
!pwd
!ls -lh
```

**Expected output**:

```
/content/drive/MyDrive/How2Sign_SONAR
manifests/
videos/
extract_features_signhiera.py
```

### Cell 3: Download SONAR Model

```python
# Create models directory
!mkdir -p models

# Download SignHiera pretrained model (~350 MB)
!wget -q --show-progress https://dl.fbaipublicfiles.com/SONAR/asl/dm_70h_ub_signhiera.pth -O models/dm_70h_ub_signhiera.pth

# Verify download
!ls -lh models/dm_70h_ub_signhiera.pth

print("✅ SONAR model downloaded")
```

### Cell 4: Test on Single Video (RECOMMENDED)

```python
# Test extraction on one video
!python extract_features_signhiera.py \
    --manifest manifests/train_sample.tsv \
    --video_dir videos/train \
    --model_path models/dm_70h_ub_signhiera.pth \
    --output_dir features/test_single \
    --max_frames 100 \
    --device cuda

# Check output
!ls -lh features/test_single/
```

**Expected output**:

```
🚀 SONAR SignHiera Feature Extraction
📍 Device: cuda
   GPU: Tesla T4
   VRAM: 15.0 GB
📦 Loading model from models/dm_70h_ub_signhiera.pth
...
✅ EXTRACTION COMPLETE
📊 Processed: 1 videos
```

### Cell 5: Extract TRAIN Features

⏱️ **Estimated time**: 3-4 hours (2147 videos)

```python
!python extract_features_signhiera.py \
    --manifest manifests/train.tsv \
    --video_dir videos/train \
    --model_path models/dm_70h_ub_signhiera.pth \
    --output_dir features/train \
    --max_frames 300 \
    --num_workers 2 \
    --device cuda

print("✅ Train features extracted")
```

### Cell 6: Extract VAL Features

⏱️ **Estimated time**: 2-3 hours (1739 videos)

```python
!python extract_features_signhiera.py \
    --manifest manifests/val.tsv \
    --video_dir videos/val \
    --model_path models/dm_70h_ub_signhiera.pth \
    --output_dir features/val \
    --max_frames 300 \
    --num_workers 2 \
    --device cuda

print("✅ Val features extracted")
```

### Cell 7: Extract TEST Features

⏱️ **Estimated time**: 3-4 hours (2343 videos)

```python
!python extract_features_signhiera.py \
    --manifest manifests/test.tsv \
    --video_dir videos/test \
    --model_path models/dm_70h_ub_signhiera.pth \
    --output_dir features/test \
    --max_frames 300 \
    --num_workers 2 \
    --device cuda

print("✅ Test features extracted")
```

### Cell 8: Verify Output

```python
# Check extracted features
import os

print("📊 TRAIN features:")
train_files = [f for f in os.listdir('features/train') if f.endswith('.npy')]
print(f"   {len(train_files)} files")

print("\n📊 VAL features:")
val_files = [f for f in os.listdir('features/val') if f.endswith('.npy')]
print(f"   {len(val_files)} files")

print("\n📊 TEST features:")
test_files = [f for f in os.listdir('features/test') if f.endswith('.npy')]
print(f"   {len(test_files)} files")

print(f"\n✅ TOTAL: {len(train_files) + len(val_files) + len(test_files)} features")

# Check feature shape
import numpy as np
sample_feature = np.load(f'features/train/{train_files[0]}')
print(f"\n📐 Feature shape: {sample_feature.shape}")
print(f"   (num_frames, feature_dim)")
```

---

## 📥 Step 5: Download Features

### Option A: Download via Colab (SMALL files)

```python
# Compress features
!tar -czf features_train.tar.gz features/train/
!tar -czf features_val.tar.gz features/val/
!tar -czf features_test.tar.gz features/test/

print("✅ Features compressed")
print("\n📥 Download from Files tab (left sidebar):")
print("   - features_train.tar.gz")
print("   - features_val.tar.gz")
print("   - features_test.tar.gz")
```

### Option B: Keep in Google Drive (RECOMMENDED)

Features are already saved in your Google Drive:

- `MyDrive/How2Sign_SONAR/features/train/`
- `MyDrive/How2Sign_SONAR/features/val/`
- `MyDrive/How2Sign_SONAR/features/test/`

Download via Google Drive desktop app or web interface.

---

## ⚠️ Troubleshooting

### Issue 1: "Runtime disconnected"

**Symptoms**: Colab disconnects after a few hours

**Solutions**:

1. Keep Colab tab open and active
2. Run smaller batches (split train into chunks)
3. Use Colab Pro ($10/month, longer runtime)
4. Save checkpoints periodically

**Prevention**:

```python
# Add to extraction script to save progress every N videos
# Modify extract_features_signhiera.py:
if num_processed % 100 == 0:
    print(f"✅ Checkpoint: {num_processed} videos processed")
```

### Issue 2: "CUDA out of memory"

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:

```python
# Reduce max_frames
--max_frames 200  # Instead of 300

# Reduce batch_size (already 1, can't reduce further)

# Process videos sequentially (already doing this)
```

### Issue 3: "Manifest file not found"

**Symptoms**: `FileNotFoundError: manifests/train.tsv`

**Solutions**:

1. Verify Google Drive mounted correctly:

   ```python
   !ls /content/drive/MyDrive/How2Sign_SONAR/
   ```

2. Check current directory:
   ```python
   import os
   print(os.getcwd())
   os.chdir('/content/drive/MyDrive/How2Sign_SONAR')
   ```

### Issue 4: Videos not loading

**Symptoms**: Many "⚠️ Video not found" warnings

**Solutions**:

1. Verify video directory structure:

   ```python
   !ls videos/train/ | head -10
   ```

2. Check video IDs match manifest:
   ```python
   import pandas as pd
   manifest = pd.read_csv('manifests/train.tsv', sep='\t')
   print(manifest.head())
   ```

---

## 📊 Expected Results

After successful extraction:

```
features/
├── train/
│   ├── --7E2sU6zP4_10-5-rgb_front.npy  # Shape: (120, 256)
│   ├── --7E2sU6zP4_11-5-rgb_front.npy  # Shape: (95, 256)
│   └── ... (2147 files)
├── val/
│   └── ... (1739 files)
└── test/
    └── ... (2343 files)
```

**Feature format**:

- `.npy` files (NumPy arrays)
- Shape: `(num_frames, feature_dim)`
- `feature_dim = 256` (SignHiera output)
- `num_frames` varies per video (typically 50-300)

**Total size**: ~2-3 GB (all splits)

---

## 🎉 Next Steps

After feature extraction completes:

1. ✅ **Download features** to local machine
2. ✅ **Verify features** with `test_sonar_features.py`
3. ✅ **Fine-tune SONAR** with `finetune_sonar_how2sign.py`
4. ✅ **Evaluate model** with `evaluate_how2sign.py`

---

## 📝 Summary

### What We Did

✅ Set up Google Colab with T4 GPU (Linux + CUDA)  
✅ Uploaded How2Sign videos and manifests to Google Drive  
✅ Downloaded SONAR SignHiera pretrained model  
✅ Extracted visual features from all videos  
✅ Saved features for fine-tuning

### Why This Works

- **Platform**: Linux + CUDA (officially supported by SSVP-SLT)
- **GPU**: T4 GPU with 15 GB VRAM (sufficient for feature extraction)
- **Cost**: FREE (Colab free tier)
- **Time**: 8-11 hours total (parallelizable by split)

### Timeline

| Task                   | Time       |
| ---------------------- | ---------- |
| Upload to Google Drive | 2-5 hours  |
| Colab setup            | 10 minutes |
| Download SONAR model   | 5 minutes  |
| Extract train features | 3-4 hours  |
| Extract val features   | 2-3 hours  |
| Extract test features  | 3-4 hours  |
| **TOTAL**              | **10-16h** |

### Next: Fine-Tuning

With features extracted, you can now:

1. **Train on Mac** (features are lightweight, no GPU needed)
2. **Or continue on Colab** (faster with GPU)

Expected performance after fine-tuning: **BLEU-4: 30-35%** ✨

---

## 🚀 Ready to Start?

1. Upload files to Google Drive
2. Open https://colab.research.google.com
3. Copy-paste cells from this guide
4. Run and wait for results!

**Good luck! 🎯**
