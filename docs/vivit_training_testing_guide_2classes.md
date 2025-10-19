# 🎯 Guida Training e Testing ViViT - 2 Classi

## 📋 Prerequisiti

1. **Avvia MLflow Server** (per tracciare gli esperimenti):

```bash
mlflow server --host 127.0.0.1 --port 8080
```

2. **Verifica la struttura dei dati**:
   - Video di training: `data/raw/train/raw_videos_front_train/`
   - Video di validazione: `data/raw/val/raw_videos_front_val/`
   - Video golden labels: `data/raw/ASLLRP/batch_utterance_video_v3_1/`
   - Annotazioni: file CSV in `data/processed/`

---

## 🚀 Training del Modello ViViT (2 Classi)

### Comando Base

```bash
python src/models/two_classes/vivit/run_train_vivit.py \
  --num_epochs 50 \
  --batch_size 1 \
  --learning_rate 2.310201887845294e-06 \
  --seed 42 \
  --weight_decay 0.0003549878832196505
```

### Altri Esempi di Comandi

```bash
# Esempio 2
python src/models/two_classes/vivit/run_train_vivit.py \
  --num_epochs 50 \
  --batch_size 1 \
  --learning_rate 1.5930522616241016e-05 \
  --seed 42 \
  --weight_decay 0.013311216080736894

# Esempio 3
python src/models/two_classes/vivit/run_train_vivit.py \
  --num_epochs 50 \
  --batch_size 1 \
  --learning_rate 2.9106359131330718e-05 \
  --seed 42 \
  --weight_decay 0.006251373574521752
```

### Parametri Principali

| Parametro              | Default                           | Descrizione                                        |
| ---------------------- | --------------------------------- | -------------------------------------------------- |
| `--model_name`         | `google/vivit-b-16x2-kinetics400` | Modello base HuggingFace                           |
| `--num_epochs`         | 10                                | Numero di epoche di training                       |
| `--batch_size`         | 4                                 | Dimensione batch (1 raccomandato per video lunghi) |
| `--learning_rate`      | 5e-5                              | Learning rate per AdamW                            |
| `--weight_decay`       | 1e-2                              | Weight decay per regolarizzazione                  |
| `--patience`           | 10                                | Pazienza per early stopping                        |
| `--scheduler_patience` | 5                                 | Pazienza per LR scheduler                          |
| `--scheduler_factor`   | 0.3                               | Fattore di riduzione LR                            |
| `--seed`               | 42                                | Seed per riproducibilità                           |
| `--num_workers`        | 2                                 | Worker per data loading                            |

### Output del Training

1. **Checkpoint locale**: Salvato in `models/vivit_emotion_model_val_f1=X.XXXX.pt`
2. **Modello registrato su MLflow**: Nome `ViViT_EmoSign_2classes`
3. **Metriche tracciate**:
   - Training loss per epoca
   - Validation: accuracy, F1 (macro), precision, recall
   - F1-Score per classe (classe 0 e classe 1)
   - Learning rate
   - Best metrics (loss, F1, F1 macro)

### Visualizzare i Risultati

Apri il browser su: `http://127.0.0.1:8080`

---

## 🧪 Testing su Golden Labels

### Passo 1: Identificare il Model URI

Dopo il training, MLflow salva il modello. Puoi trovare l'URI in due modi:

**Opzione A - Da MLflow UI:**

1. Vai su `http://127.0.0.1:8080`
2. Clicca sull'esperimento "ViViT - Emotion Recognition"
3. Seleziona il run desiderato
4. Nella sezione "Artifacts" → "model", copia il path

**Opzione B - Dalla struttura file:**

```bash
# Il path sarà simile a:
mlartifacts/<EXPERIMENT_ID>/models/<MODEL_ID>/artifacts
```

### Passo 2: Eseguire il Test

```bash
python src/models/two_classes/vivit/test_golden_labels_vivit.py \
  --model_uri mlartifacts/EXPERIMENT_ID/models/MODEL_ID/artifacts \
  --batch_size 1 \
  --save_results
```

### Parametri del Test

| Parametro        | Required | Descrizione                                                |
| ---------------- | -------- | ---------------------------------------------------------- |
| `--model_uri`    | ✅       | URI del modello MLflow (path agli artifacts)               |
| `--batch_size`   | ❌       | Batch size (default: 1, raccomandato)                      |
| `--model_name`   | ❌       | Nome modello HF (default: google/vivit-b-16x2-kinetics400) |
| `--save_results` | ❌       | Flag per salvare risultati in CSV                          |

### Output del Test

Quando usi `--save_results`, vengono creati 3 file CSV in `results/`:

1. **`vivit_golden_labels_test_results_2_classes.csv`**

   - Risultati per ogni video
   - Colonne: video_name, true_label, predicted_label, confidence per classe, max_confidence, correct_prediction

2. **`vivit_golden_labels_metrics_summary_2_classes.csv`**

   - Metriche aggregate
   - Accuracy (standard, balanced, weighted)
   - F1, Precision, Recall (macro e weighted)

3. **`vivit_golden_labels_f1_per_class_2_classes.csv`**
   - F1-Score dettagliato per ogni classe (positive/negative)

### Output Console

Lo script stampa:

- ✅ Metriche principali (Accuracy, F1, Precision, Recall)
- 📊 Distribuzione predizioni vs etichette reali
- 🎲 Analisi delle probabilità per classe
- 📈 Primi 10 campioni con predizioni
- 🔢 Confusion matrix
- 📋 Classification report completo

---

## 🔍 Esempio Completo di Workflow

### 1. Avvia MLflow

```bash
mlflow server --host 127.0.0.1 --port 8080
```

### 2. Training

```bash
python src/models/two_classes/vivit/run_train_vivit.py \
  --num_epochs 50 \
  --batch_size 1 \
  --learning_rate 2.310201887845294e-06 \
  --seed 42 \
  --weight_decay 0.0003549878832196505
```

### 3. Trova il Model URI

- Apri `http://127.0.0.1:8080`
- Trova il run appena completato
- Copia il path del modello, esempio:
  ```
  mlartifacts/123456789/models/m-abc123def456/artifacts
  ```

### 4. Test su Golden Labels

```bash
python src/models/two_classes/vivit/test_golden_labels_vivit.py \
  --model_uri mlartifacts/123456789/models/m-abc123def456/artifacts \
  --batch_size 1 \
  --save_results
```

### 5. Analizza i Risultati

```bash
# Visualizza i risultati
cat results/vivit_golden_labels_metrics_summary_2_classes.csv

# Oppure apri i CSV in Excel/Python per analisi dettagliate
```

---

## 📝 Note Importanti

### Differenze rispetto a 3 Classi

- **Dataset**: Usa `golden_label_sentiment.csv` invece di `golden_label_sentiment_with_neutral.csv`
- **Classi**: Solo 2 classi (positive/negative) invece di 3 (positive/negative/neutral)
- **Modello registrato**: Nome `ViViT_EmoSign_2classes` invece di `ViViT_EmoSign_3classes`
- **Output files**: Suffisso `_2_classes` invece di `_3_classes_with_neutral`

### Performance

- **Batch size**: Raccomandato `1` per ViViT su video lunghi per evitare OOM
- **Device**: Supporta automaticamente CUDA, MPS (Apple Silicon), o CPU
- **Memory**: Usa garbage collection e cache clearing tra le epoche

### Troubleshooting

**Problema**: `FileNotFoundError` per i video

- ✅ Verifica che i video siano nelle directory corrette
- ✅ Controlla che i nomi nei CSV corrispondano ai file video

**Problema**: Modello non trovato su MLflow

- ✅ Verifica che MLflow server sia attivo
- ✅ Controlla che il training sia completato senza errori
- ✅ Usa il path completo: `mlartifacts/.../artifacts` (non dimenticare `/artifacts` finale)

**Problema**: Out of Memory (OOM)

- ✅ Riduci batch_size a 1
- ✅ Riduci num_workers a 0
- ✅ Usa un modello più piccolo o meno frame

**Problema**: Wrong number of classes

- ✅ Assicurati di usare il file CSV corretto (`golden_label_sentiment.csv` per 2 classi)
- ✅ Verifica che il modello sia stato trainato con 2 classi

---

## 🎓 Metriche di Valutazione

### F1-Score (Metrica Principale)

- **Macro F1**: Media non pesata dell'F1 per ogni classe (usata per checkpoint)
- **Weighted F1**: F1 pesato per il supporto di ogni classe

### Accuracy

- **Standard**: % predizioni corrette sul totale
- **Balanced**: Robusta al class imbalance
- **Weighted**: Considera la distribuzione delle classi

### Confusion Matrix

Mostra le predizioni corrette e gli errori di classificazione tra positive e negative.

---

## 🆚 Confronto 2 Classi vs 3 Classi

### Quando usare 2 classi:

- ✅ Task binario semplice (positive vs negative)
- ✅ Quando la classe neutral è ambigua o poco rappresentata
- ✅ Per confronto con baseline binarie

### Quando usare 3 classi:

- ✅ Task più complesso e realistico
- ✅ Quando la classe neutral è importante per l'applicazione
- ✅ Per catturare sfumature emotive intermedie

---

## 📚 Riferimenti

- **Modello**: [ViViT: A Video Vision Transformer](https://huggingface.co/google/vivit-b-16x2-kinetics400)
- **Framework**: PyTorch + Ignite + MLflow
- **Dataset**: ASLLRP Sign Language Videos
- **Classi**: 2 (positive, negative)

---

**Ultimo aggiornamento**: 19 Ottobre 2025
