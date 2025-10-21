# Seven Classes Emotion Recognition Models

This folder contains models and training scripts adapted for **7-class emotion recognition**.

## 📊 The 7 Emotion Classes

1. **Extremely Negative**
2. **Negative**
3. **Somewhat Negative**
4. **Neutral**
5. **Somewhat Positive**
6. **Positive**
7. **Extremely Positive**

## 📁 Structure

```
seven_classes/
├── README.md                              # This file
├── __init__.py                            # Module initialization
├── lstm_model.py                          # ✅ LSTM architecture for 7 classes
├── stgcn_model.py                         # ✅ ST-GCN architecture for 7 classes
├── run_train.py                           # ✅ Training script adapted for 7 classes
├── test_golden_labels.py                  # ✅ Test script for golden labels
├── test_golden_labels_fixed.py            # ✅ Test with normalization fix
├── fix_golden_labels_normalization.py     # ✅ Normalization utilities
├── hyperparameter_tuning.py               # ✅ Hyperparameter optimization (needs updating)
├── qwen_vl_golden_inference.py            # ⚠️  Vision-Language model inference (may need updates)
└── vivit/                                 # ✅ ViViT model for video-based classification
    ├── vivit_model.py                     # ✅ ViViT architecture
    ├── run_train_vivit.py                 # ✅ ViViT training script
    ├── test_golden_labels_vivit.py        # ✅ ViViT test script
    ├── hyperparameter_tuning_vivit.py     # ✅ ViViT hyperparameter tuning
    └── video_dataset.py                   # ✅ Dataset loader for ViViT
```

## 🔄 Key Differences from `three_classes/`

### 1. **Number of Output Classes**

- `three_classes/`: 3 classes (Positive, Negative, Neutral)
- `seven_classes/`: 7 classes (with granular sentiment levels)

### 2. **Data Files** (⚠️ Important!)

- **Training data**: `data/processed/asllrp_video_sentiment_data_7_classes_without_golden.csv`
- **Test data (golden labels)**: `data/processed/golden_label_sentiment_7_classes.csv`
- **Configuration thresholds**: Based on Config 2 (Concentrated) from `notebooks/seven_classes/01_explore_sentiment_7_classes.ipynb`

**Note**: You also need to generate the How2Sign train/val CSV files with 7 classes using notebook 01 on train/val splits.

### 3. **Class Mapping**

```python
EMOTION_TO_IDX = {
    "Extremely Negative": 0,
    "Negative": 1,
    "Somewhat Negative": 2,
    "Neutral": 3,
    "Somewhat Positive": 4,
    "Positive": 5,
    "Extremely Positive": 6
}
```

### 4. **Evaluation Metrics**

- **Macro F1-Score**: Treats all 7 classes equally (recommended)
- **Weighted F1-Score**: Accounts for class imbalance
- **Per-class metrics**: Important due to class imbalance in golden labels

### 5. **MLflow Experiment Name**

- Changed from `"VADER 0.34 - Emotion Recognition"` to `"7 Classes - Emotion Recognition"`
- Includes `num_classes=7` parameter in all experiments

## ⚠️ Important Notes

### Golden Labels Coverage

The golden labels (`golden_label_sentiment_7_classes.csv`) contain **only 5 out of 7 classes**:

✅ **Present** (200 samples):

- Extremely Negative: 5 (2.50%)
- Negative: 78 (39.00%)
- Somewhat Negative: 16 (8.00%)
- Positive: 71 (35.50%)
- Extremely Positive: 30 (15.00%)

❌ **Missing**:

- Neutral: 0 (0%)
- Somewhat Positive: 0 (0%)

**Why?** The original EmoSign dataset has only 3 classes. When matched with our 7-class VADER classification, no examples fell into these intermediate categories.

**Recommendation**:

- Use golden labels to evaluate 5/7 classes
- Use cross-validation on training set for all 7 classes
- Document this limitation in thesis/papers

### Class Imbalance

The training data has significant class imbalance. The distribution heavily depends on the VADER compound scores.

**Solutions**:

- ✅ Use `FocalLoss` with class weights (already implemented in `run_train.py`)
- ⚠️ Consider data augmentation for underrepresented classes
- ✅ Use macro-averaged metrics for fair evaluation

## 🚀 Usage

### 1. Training LSTM Model

```bash
# Start MLflow server first
mlflow server --host 127.0.0.1 --port 8080

# Train LSTM
python src/models/seven_classes/run_train.py \
    --model_type lstm \
    --batch_size 32 \
    --hidden_size 256 \
    --num_layers 2 \
    --learning_rate 1e-4 \
    --dropout 0.5 \
    --num_epochs 100 \
    --patience 20 \
    --seed 42
```

### 2. Training ST-GCN Model

```bash
python src/models/seven_classes/run_train.py \
    --model_type stgcn \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --dropout 0.3 \
    --num_epochs 100 \
    --seed 42
```

### 3. Testing on Golden Labels

```bash
python src/models/seven_classes/test_golden_labels_fixed.py \
    --model_uri mlartifacts/[RUN_ID]/models/[MODEL_ID]/artifacts \
    --batch_size 32 \
    --save_results
```

### 4. Training ViViT Model

```bash
python src/models/seven_classes/vivit/run_train_vivit.py \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --num_epochs 50 \
    --seed 42
```

## 📈 Expected Performance

Due to the increased granularity (7 vs 3 classes), expect:

- **Lower absolute accuracy** compared to 3-class system (more classes = harder task)
- **More nuanced predictions** capturing sentiment intensity
- **Better performance on clear examples** (extremely positive/negative)
- **More confusion in intermediate categories** (somewhat positive/negative, neutral)
- **Macro F1-score** likely in range 0.25-0.40 (baseline)
- **Weighted F1-score** may be higher due to majority classes

## 🔬 Experiment Tracking

All experiments are tracked using MLflow:

```bash
mlflow server --host 127.0.0.1 --port 8080
# Then open http://127.0.0.1:8080 in browser
```

Experiments are saved under: `"7 Classes - Emotion Recognition"`

Filter by:

- `num_classes=7`
- `model_type` (lstm, stgcn, vivit)
- Configuration suffixes (DS:ratio, Norm:type)

## 📚 Related Files

### Data Generation Notebooks:

- `notebooks/seven_classes/01_explore_sentiment_7_classes.ipynb` - Threshold selection and analysis
- `notebooks/seven_classes/02_sentiment_analysis_ASLLRP_7_classes.ipynb` - ASLLRP dataset classification
- `notebooks/seven_classes/03_find_golden_labels_7_classes.ipynb` - Test set creation and golden labels

### Utility Files:

- `src/utils/seven_classes/training_utils.py` - Training utilities adapted for 7 classes
- `src/data_pipeline/landmark_dataset.py` - Landmark dataset loader (shared with 3 classes)

## ⚡ Quick Start

1. **Generate data** (if not done):

   ```bash
   # Run notebooks in order:
   # 01_explore_sentiment_7_classes.ipynb (on train mode)
   # 01_explore_sentiment_7_classes.ipynb (on val mode)
   # 02_sentiment_analysis_ASLLRP_7_classes.ipynb
   # 03_find_golden_labels_7_classes.ipynb
   ```

2. **Start MLflow**:

   ```bash
   mlflow server --host 127.0.0.1 --port 8080
   ```

3. **Train model**:

   ```bash
   python src/models/seven_classes/run_train.py \
       --model_type lstm \
       --batch_size 32 \
       --hidden_size 256 \
       --num_layers 2 \
       --learning_rate 1e-4 \
       --dropout 0.5 \
       --num_epochs 100 \
       --patience 20 \
       --seed 42
   ```

4. **Test on golden labels**:
   ```bash
   python src/models/seven_classes/test_golden_labels_fixed.py \
       --model_uri [YOUR_MODEL_URI_FROM_MLFLOW] \
       --batch_size 32 \
       --save_results
   ```

## 🎯 Next Steps

1. ✅ Model architectures created (LSTM, ST-GCN)
2. ✅ Training scripts adapted for 7 classes
3. ✅ Test scripts adapted for 7 classes
4. ✅ Utility files copied and adapted
5. ⏳ **TODO**: Generate How2Sign train/val CSV files with 7 classes (run notebook 01 on train/val modes)
6. ⏳ **TODO**: Run initial experiments
7. ⏳ **TODO**: Update hyperparameter tuning scripts if needed
8. ⏳ **TODO**: Compare results with 3-class baseline
9. ⏳ **TODO**: Document findings in thesis

## 📝 Citation

If using this code, please reference the EmoSign project and the configuration methodology for 7-class sentiment analysis described in the thesis documentation.

## 🐛 Known Issues

- ⚠️ `qwen_vl_golden_inference.py` may need adaptation for 7 classes (currently copied from 3 classes)
- ⚠️ How2Sign train/val CSV files for 7 classes need to be generated (use notebook 01)
- ⚠️ Hyperparameter tuning scripts may need minor adjustments for 7-class search space
