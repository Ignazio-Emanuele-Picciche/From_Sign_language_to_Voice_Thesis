# Qwen2.5-VL Golden Labels Inference

Questo documento descrive come utilizzare lo script per l'inferenza con Qwen2.5-VL sui golden labels.

## Prerequisiti

### 1. Installazione Dipendenze

```bash
# Installa le dipendenze aggiornate
pip install -r requirements.txt

# Se hai problemi con flash-attention su macOS, installalo separatamente:
pip install flash-attn --no-build-isolation
```

### 2. Modelli Supportati

Lo script supporta diversi modelli Qwen2.5-VL, dal più piccolo al più grande:

- **0.5B** (`Qwen/Qwen2-VL-0.5B-Instruct`) - Il più leggero, per PC con poca memoria
- **1.5B** (`Qwen/Qwen2-VL-1.5B-Instruct`) - Buon compromesso performance/memoria
- **7B** (`Qwen/Qwen2.5-VL-7B-Instruct`) - Alta performance, richiede più memoria
- **72B** (`Qwen/Qwen2.5-VL-72B-Instruct`) - Il migliore ma molto pesante

## Uso

### Comando Base

```bash
python src/models/qwen_vl_golden_inference.py --model_size 1.5B --save_results
```

### Opzioni Disponibili

- `--model_size`: Dimensione del modello (`0.5B`, `1.5B`, `7B`, `72B`)
- `--save_results`: Salva i risultati in file CSV
- `--max_videos`: Limita il numero di video da processare (per test)
- `--batch_size`: Non utilizzato attualmente (mantenuto per compatibilità)

### Esempi

```bash
# Test rapido con 10 video e modello piccolo
python src/models/qwen_vl_golden_inference.py --model_size 0.5B --max_videos 10 --save_results

# Inferenza completa con modello medio
python src/models/qwen_vl_golden_inference.py --model_size 1.5B --save_results

# Prova con modello grande (se hai abbastanza memoria)
python src/models/qwen_vl_golden_inference.py --model_size 7B --save_results
```

## Output

### Metriche Principali (Focus del Progetto)

- **Weighted Accuracy**: Accuracy che considera la distribuzione delle classi
- **Weighted F1**: F1-score pesato per il supporto di ogni classe

### Altre Metriche Calcolate

- Accuracy (Standard, Balanced, Weighted)
- F1-Score (Macro, Weighted)
- Precision (Macro, Weighted)
- Recall (Macro, Weighted)
- Matrice di confusione
- Classification report

### File di Output

1. **`qwen_vl_{model_size}_golden_results.csv`**: Risultati dettagliati per ogni video

   - video_name: Nome del file video
   - true_label: Etichetta reale (Positive/Negative)
   - predicted_label: Predizione del modello
   - confidence: Confidenza della predizione
   - caption: Testo del video (per riferimento)
   - correct_prediction: Boolean se la predizione è corretta

2. **`qwen_vl_{model_size}_metrics_summary.csv`**: Metriche aggregate
   - metric: Nome della metrica
   - value: Valore numerico (0-1)
   - percentage: Valore in percentuale
   - description: Descrizione della metrica

## Funzionamento Tecnico

### Processamento Video

1. **Estrazione Frame**: Il sistema estrae 8 frame distribuiti uniformemente dal video
2. **Frame Centrale**: Usa il frame centrale come rappresentativo per l'analisi
3. **Ridimensionamento**: I frame vengono ridimensionati a 224x224 pixel
4. **Formato**: Conversione da BGR (OpenCV) a RGB (PIL) per compatibilità

### Prompt Engineering

Il modello riceve questo prompt specializzato:

```
Analyze the emotional sentiment expressed in this sign language video.

Focus on:
- Facial expressions of the signer
- Body language and gestures
- Overall emotional tone conveyed
- Context from the signing content

Classify the sentiment as either:
- Positive: Happy, joyful, optimistic, enthusiastic, satisfied emotions
- Negative: Sad, angry, frustrated, disappointed, worried, stressed emotions

Respond with ONLY one word: either "Positive" or "Negative".
```

### Parsing Intelligente

Il sistema include logic di parsing robusto:

1. Cerca direttamente "Positive" o "Negative" nella risposta
2. Se non trova, analizza parole chiave correlate al sentiment
3. Fallback deterministico per garantire sempre una predizione

## Requisiti Sistema

### Memoria

- **0.5B**: ~2-3 GB VRAM/RAM
- **1.5B**: ~4-6 GB VRAM/RAM
- **7B**: ~16-20 GB VRAM/RAM
- **72B**: ~80+ GB VRAM/RAM

### Device Support

- **CUDA**: GPU NVIDIA (preferito per performance)
- **MPS**: GPU Apple Silicon (M1/M2/M3)
- **CPU**: Fallback universale (più lento)

Il sistema rileva automaticamente il device ottimale.

## Troubleshooting

### Memoria Insufficiente

Se il modello richiesto è troppo grande, lo script tenterà automaticamente di usare un modello più piccolo.

### Video Non Trovati

Lo script salta automaticamente i video mancanti e continua con quelli disponibili.

### Crash Durante l'Esecuzione

Il sistema salva risultati intermedi ogni 50 video per evitare perdite di progresso.

### Flash Attention Issues

Se hai problemi con flash-attention su macOS:

```bash
pip uninstall flash-attn
# Lo script userà automatically "eager" attention come fallback
```

## Performance Attese

I tempi dipendono dal modello e hardware:

- **0.5B + MPS**: ~2-3 secondi per video
- **1.5B + CUDA**: ~1-2 secondi per video
- **7B + CUDA**: ~3-5 secondi per video

Per ~200 golden labels, aspettati 10-20 minuti con modelli più piccoli.
