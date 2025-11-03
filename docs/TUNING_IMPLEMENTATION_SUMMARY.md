# Optuna Hyperparameter Tuning - Implementation Summary

## ✅ Cosa è stato implementato

### 1. Script di Tuning (`src/sign_to_text/tune.py`)

- **Framework**: Optuna con integrazione MLflow
- **Objective function**: Addestra modello con parametri suggeriti, ritorna val_bleu o val_loss
- **FastTrainer**: Versione leggera del Trainer per trials veloci (no checkpointing)
- **Parametri ottimizzati**:
  - `lr`: 1e-5 → 5e-4 (log uniform)
  - `batch_size`: [8, 16, 32]
  - `d_model`: [128, 256, 512]
  - `num_encoder_layers`: 2 → 6
  - `num_decoder_layers`: 2 → 6
  - `nhead`: [4, 8]
  - `dim_feedforward`: [512, 1024, 2048]
  - `dropout`: 0.1 → 0.4
  - `weight_decay`: 1e-6 → 1e-2 (log)
  - `label_smoothing`: 0.0 → 0.2

### 2. Scripts di Avvio

- **`run_tuning.sh`**: Lancia tuning completo (30 trials, 5 epochs, optimize BLEU)
- **`test_tuning.sh`**: Test rapido (2 trials, 2 epochs, 20% subset)
- **`train_from_tuning.py`**: Carica best params e lancia training finale

### 3. Documentazione

- **`docs/HYPERPARAMETER_TUNING_GUIDE.md`**: Guida completa con esempi, strategie, troubleshooting
- **`src/sign_to_text/README.md`**: Aggiornato con sezione tuning

### 4. Output Generati

- **`results/best_hyperparameters.json`**: Best params in formato JSON
- **`results/param_importances.html`**: Plot importanza parametri (Plotly)
- **MLflow tracking**: Tutti i trials loggati in experiment "sign_to_text_tuning"

## 🚀 Come Usare

### Quick Start (consigliato per primo test)

```bash
# Test veloce (5-10 minuti)
./test_tuning.sh

# Se funziona, lancia tuning completo
./run_tuning.sh
```

### Tuning Personalizzato

```bash
# Tuning veloce su subset (30% dati, 20 trials, 3 epochs)
.venv/bin/python src/sign_to_text/tune.py \
    --n_trials 20 \
    --epochs 3 \
    --subset_fraction 0.3 \
    --optimize bleu

# Tuning approfondito (full data, 40 trials, 10 epochs)
.venv/bin/python src/sign_to_text/tune.py \
    --n_trials 40 \
    --epochs 10 \
    --optimize bleu
```

### Training con Best Params

```bash
# Automatico (legge da results/best_hyperparameters.json)
python src/sign_to_text/train_from_tuning.py --epochs 50

# Manuale
.venv/bin/python src/sign_to_text/train.py \
    --batch_size <from_json> \
    --d_model <from_json> \
    --lr <from_json> \
    # ...
```

## 📊 Strategie Consigliate

### Strategia 1: Rapida Esplorazione + Raffinamento

1. **Esplorazione (1-2 ore)**:
   ```bash
   .venv/bin/python src/sign_to_text/tune.py \
       --n_trials 20 --epochs 3 --subset_fraction 0.3
   ```
2. **Raffinamento (3-5 ore)**:
   Prendi top 3 configs, modifica range params in `tune.py`, ri-esegui con full data
3. **Training finale (6-12 ore)**:
   `python train_from_tuning.py --epochs 50`

### Strategia 2: Grid Search Manuale

Se preferisci controllo totale:

1. Identifica da tuning quali params hanno più impatto (vedi `param_importances.html`)
2. Fissa params meno importanti, esplora grid manuale su quelli critici
3. Lancia training multipli in parallelo

### Strategia 3: Multi-Stage Tuning

1. **Stage 1**: Tuning architettura (d_model, layers) → 20 trials
2. **Stage 2**: Tuning optimizer (lr, weight_decay, label_smoothing) → 20 trials con best arch
3. **Stage 3**: Fine-tuning (dropout, batch_size) → 10 trials

## 🔧 Parametri Tuning Script

| Parametro           | Default             | Descrizione                                        |
| ------------------- | ------------------- | -------------------------------------------------- |
| `--n_trials`        | 30                  | Numero di configurazioni da testare                |
| `--epochs`          | 5                   | Epoche per ogni trial (budget limitato)            |
| `--optimize`        | bleu                | Metrica: `bleu` (max) o `loss` (min)               |
| `--subset_fraction` | 1.0                 | Frazione train set (0.3 = 30%)                     |
| `--study_name`      | sign_to_text_tuning | Nome esperimento                                   |
| `--storage`         | None                | DB per trials paralleli (es: `sqlite:///study.db`) |

## 📈 Metriche Attese

### Baseline (training senza tuning)

- Train Loss: ~2.7
- Val Loss: ~5.2
- Val BLEU: ~0.00-0.02 (quasi zero)

**Problema**: Overfitting + greedy decoding poco efficace

### Dopo Tuning (aspettative)

- Train Loss: ~2.0-2.5 (con migliore regularization)
- Val Loss: ~4.5-5.0 (gap ridotto)
- Val BLEU: ~0.05-0.15 (con beam search può salire a 0.15-0.25)

**Note**:

- BLEU 0.15+ su sign→text è considerato buono (task molto complesso)
- Se BLEU resta < 0.05 anche dopo tuning → controllare tokenizer/dataset

## ⚡ Performance Stimate

### Hardware: M3 CPU (come attuale setup)

- **1 trial** (5 epochs, full data 1.5k samples): ~5-8 min
- **20 trials**: ~2-2.5 ore
- **30 trials**: ~3-4 ore
- **50 trials**: ~5-7 ore

### Con subset 30%:

- **1 trial** (5 epochs, 450 samples): ~2-3 min
- **20 trials**: ~40-60 min
- **30 trials**: ~1.5-2 ore

### Con epochs ridotte (3 invece di 5):

- **1 trial** (3 epochs, full data): ~3-5 min
- **30 trials**: ~2-3 ore

## 🛠️ Troubleshooting

### MPS Errors

Script forza CPU per evitare bug nested tensor. Per usare GPU:

```python
# In tune.py, line ~235
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Memory Issues

Riduci spazio ricerca:

```python
# In tune.py, modifica suggest ranges:
batch_size = trial.suggest_categorical("batch_size", [8, 16])  # no 32
d_model = trial.suggest_categorical("d_model", [128, 256])  # no 512
```

### Slow Trials

- Riduci `--epochs` a 3
- Usa `--subset_fraction 0.3`
- Riduci `num_workers=0` (già fatto)

### BLEU sempre 0.00

1. Controlla tokenizer decode (spacing, punctuation)
2. Aggiungi beam search in validation
3. Verifica alignment caption/video nel dataset

## 📚 Next Steps

Dopo il tuning:

1. **Implementa beam search** in `SignToTextTransformer.generate()`
2. **Crea evaluate.py** per test set con BLEU/WER/CER
3. **Applica miglioramenti** trovati dal tuning:
   - Gradient clipping (se dropout alto migliora)
   - LR warmup (se lr piccolo vince)
   - Data augmentation (se overfitting persiste)
4. **Re-train modello finale** con best params + 50-100 epochs

## 🎯 Obiettivi Realistici

### Short-term (dopo primo tuning)

- [ ] Identificare migliori architettura (d_model, layers)
- [ ] Ridurre gap train/val loss a < 2.0
- [ ] Raggiungere BLEU > 0.05 su validation

### Mid-term (dopo training finale)

- [ ] BLEU > 0.15 su test set (con beam search)
- [ ] WER < 0.70 (70% errore di parola è OK per questo task)
- [ ] Esempi qualitativi con overlap semantico corretto

### Long-term (ottimizzazioni avanzate)

- [ ] Ensemble di modelli
- [ ] Data augmentation (temporal, noise)
- [ ] Transfer learning da modelli pre-trained
- [ ] Scheduled sampling per decoder
