# Hyperparameter Tuning per Sign-to-Text

Guida rapida all'ottimizzazione automatica degli iperparametri con Optuna.

## 📋 Quick Start

### Opzione 1: Script Automatico

```bash
./run_tuning.sh
```

### Opzione 2: Manuale con Parametri Personalizzati

```bash
# Tuning veloce (10 trials, 3 epochs)
.venv/bin/python src/sign_to_text/tune.py \
    --n_trials 10 \
    --epochs 3 \
    --optimize bleu

# Tuning approfondito (50 trials, 10 epochs)
.venv/bin/python src/sign_to_text/tune.py \
    --n_trials 50 \
    --epochs 10 \
    --optimize bleu

# Tuning su subset (30% del training, più veloce)
.venv/bin/python src/sign_to_text/tune.py \
    --n_trials 20 \
    --epochs 5 \
    --optimize bleu \
    --subset_fraction 0.3

# Ottimizza Val Loss invece di BLEU
.venv/bin/python src/sign_to_text/tune.py \
    --n_trials 30 \
    --epochs 5 \
    --optimize loss
```

## 🔧 Parametri Ottimizzati

Lo script esplora automaticamente:

| Parametro            | Range             | Tipo        |
| -------------------- | ----------------- | ----------- |
| `lr`                 | 1e-5 → 5e-4       | Log uniform |
| `batch_size`         | [8, 16, 32]       | Categorical |
| `d_model`            | [128, 256, 512]   | Categorical |
| `num_encoder_layers` | 2 → 6             | Integer     |
| `num_decoder_layers` | 2 → 6             | Integer     |
| `nhead`              | [4, 8]            | Categorical |
| `dim_feedforward`    | [512, 1024, 2048] | Categorical |
| `dropout`            | 0.1 → 0.4         | Uniform     |
| `weight_decay`       | 1e-6 → 1e-2       | Log uniform |
| `label_smoothing`    | 0.0 → 0.2         | Uniform     |

## 📊 Risultati

Dopo il tuning troverai:

1. **`results/best_hyperparameters.json`** - Migliori parametri trovati
2. **`results/param_importances.html`** - Visualizzazione importanza parametri
3. **MLflow UI** - Tutti i trials logged: `mlflow ui` → http://localhost:5000

### Esempio Output

```json
{
  "trial_number": 17,
  "best_value": 0.0234,
  "optimize_metric": "bleu",
  "params": {
    "lr": 0.00012,
    "batch_size": 16,
    "d_model": 256,
    "num_encoder_layers": 4,
    "num_decoder_layers": 4,
    "nhead": 8,
    "dim_feedforward": 1024,
    "dropout": 0.25,
    "weight_decay": 0.0001,
    "label_smoothing": 0.1
  },
  "user_attrs": {
    "best_val_loss": 5.12,
    "best_val_bleu": 0.0234
  }
}
```

## 🚀 Training con Migliori Parametri

Dopo il tuning, lancia training completo:

```bash
# Carica params da JSON
python src/sign_to_text/train_from_tuning.py

# Oppure manualmente
.venv/bin/python src/sign_to_text/train.py \
    --epochs 50 \
    --batch_size 16 \
    --d_model 256 \
    --nhead 8 \
    --num_encoder_layers 4 \
    --num_decoder_layers 4 \
    --lr 0.00012 \
    --patience 15
```

## ⚡ Consigli per Velocizzare

1. **Riduci epochs per trial**: `--epochs 3` invece di 5
2. **Usa subset del training**: `--subset_fraction 0.3` (30% dei dati)
3. **Riduci numero trials**: `--n_trials 10` per primo giro esplorativo
4. **Parallel trials** (richiede DB storage):

   ```bash
   # Terminal 1
   .venv/bin/python src/sign_to_text/tune.py \
       --n_trials 10 \
       --storage sqlite:///optuna_study.db

   # Terminal 2 (parallelizza)
   .venv/bin/python src/sign_to_text/tune.py \
       --n_trials 10 \
       --storage sqlite:///optuna_study.db
   ```

## 🧪 Strategia Consigliata

### Fase 1: Esplorazione Rapida (1-2 ore)

```bash
.venv/bin/python src/sign_to_text/tune.py \
    --n_trials 20 \
    --epochs 3 \
    --subset_fraction 0.3 \
    --optimize bleu
```

### Fase 2: Raffinamento (3-5 ore)

Prendi top 3 configurazioni dalla Fase 1, aggiusta manualmente range:

```bash
.venv/bin/python src/sign_to_text/tune.py \
    --n_trials 30 \
    --epochs 5 \
    --subset_fraction 1.0 \
    --optimize bleu
```

### Fase 3: Training Finale (6-12 ore)

Usa best params con 50 epochs, full training set, early stopping.

## 📈 Interpretare Risultati

- **BLEU < 0.05**: Modello non sta imparando → controlla dati/tokenizer
- **BLEU 0.05-0.15**: Baseline ragionevole, margine miglioramento
- **BLEU > 0.15**: Buoni risultati per task complesso come sign→text
- **Val Loss plateau ~5.0**: Indica limite capacità modello/dati

Se BLEU rimane 0.00:

1. Verifica tokenizer decode (spacing, punctuation)
2. Usa beam search invece di greedy per validazione
3. Controlla allineamento caption/video nel dataset

## 🛠️ Troubleshooting

### MPS Errors

Lo script forza CPU per evitare bug MPS. Se vuoi GPU:

```python
# In tune.py, cambia:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### Memory Issues

Riduci batch_size o d_model:

```bash
# Limita spazio ricerca
# Modifica in tune.py, linee 243-244:
batch_size = trial.suggest_categorical("batch_size", [8, 16])  # no 32
d_model = trial.suggest_categorical("d_model", [128, 256])  # no 512
```

### Slow Trials

Ogni trial con epochs=5, full data (~1.5k samples) richiede ~5-8 min su CPU.

- Con subset 30%: ~2-3 min/trial
- Con epochs=3: ~3-5 min/trial

30 trials × 5 min = **~2.5 ore** (stima ragionevole)

## 📚 Riferimenti

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [MLflow Optuna Integration](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.MLflowCallback.html)
- [Transformer Hyperparameters Guide](https://arxiv.org/abs/1706.03762)
