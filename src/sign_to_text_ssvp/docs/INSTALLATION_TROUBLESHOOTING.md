# 🔧 SSVP-SLT Installation Troubleshooting

## ❌ Problema Riscontrato

Durante l'installazione di SSVP-SLT con `bash scripts/install_ssvp.sh`, si verificano due errori:

### 1. Conflitto di Versioni

```
ERROR: pip's dependency resolver does not currently take into account all the packages...
sign-language-translator 0.8.1 requires torch==2.2.*, but you have torch 2.5.1
sign-language-translator 0.8.1 requires numpy==1.26.*, but you have numpy 2.2.6
```

### 2. ModuleNotFoundError durante setup

```
ModuleNotFoundError: No module named 'torch'
```

---

## 🔍 Cause

1. **Conflitto dipendenze**: Hai già installato `sign-language-translator` che richiede versioni specifiche di torch/numpy
2. **Setup.py di SSVP-SLT**: Richiede torch già installato PRIMA di eseguire `pip install -e .`
3. **Ordine di installazione**: Le dipendenze vengono installate nell'ordine sbagliato

---

## ✅ Soluzioni

### Soluzione A: Installazione Minimale (CONSIGLIATA) ⭐

Usa lo script minimale che evita i conflitti:

```bash
cd src/sign_to_text_ssvp
bash scripts/install_ssvp_minimal.sh
```

**Cosa fa**:

- ✅ Clona repository SSVP-SLT
- ✅ Installa solo dipendenze essenziali (tensorboard, sentencepiece, etc.)
- ✅ NON installa il package SSVP-SLT (evita conflitti)
- ✅ Usa il repository come reference per implementazione

**Vantaggi**:

- Nessun conflitto di versioni
- Più veloce
- Mantieni le tue versioni di torch/numpy
- Accesso completo al codice SSVP-SLT per reference

---

### Soluzione B: Installazione Completa (Aggiornata)

Lo script `install_ssvp.sh` è stato aggiornato con:

```bash
cd src/sign_to_text_ssvp
bash scripts/install_ssvp.sh
```

**Miglioramenti**:

- ✅ Controlla versione PyTorch prima dell'installazione
- ✅ Installa torch PRIMA di `pip install -e .`
- ✅ Gestisce errori gracefully (continua anche se setup fallisce)
- ✅ Logging migliorato (`/tmp/ssvp_install.log`)

---

### Soluzione C: Risolvi Conflitti Manualmente

Se vuoi le versioni esatte richieste da SSVP-SLT:

```bash
# 1. Disinstalla sign-language-translator (se non ti serve)
pip uninstall sign-language-translator -y

# 2. Installa versioni compatibili
pip install torch==2.2.2 torchvision==0.17.2
pip install numpy==1.26.4

# 3. Riprova installazione completa
bash scripts/install_ssvp.sh
```

⚠️ **Warning**: Questo potrebbe rompere altre parti del tuo progetto che dipendono da versioni più recenti.

---

## 🚀 Raccomandazione

**USA SOLUZIONE A (Minimale)** perché:

1. ✅ **Nessun conflitto** con dipendenze esistenti
2. ✅ **Più semplice** e veloce
3. ✅ **Sufficiente** per studiare SSVP-SLT e implementare fine-tuning
4. ✅ **Codice completo** disponibile in `models/ssvp_slt_repo/`

---

## 📋 Verifica Installazione

Dopo l'installazione, verifica:

```bash
# 1. Repository clonato
ls -lh models/ssvp_slt_repo/

# 2. PyTorch funzionante
python -c "import torch; print(torch.__version__)"

# 3. Dipendenze essenziali
python -c "import sentencepiece, sacrebleu, tensorboard; print('✅ OK')"
```

---

## 🎯 Prossimi Passi

Con l'installazione minimale, puoi:

### 1. Studiare Codice SSVP-SLT

```bash
cd models/ssvp_slt_repo
cat translation/README.md
ls -lh translation/*.py
```

### 2. Download Modelli Pretrained

```bash
cd ../../
python download_pretrained.py --model base
```

### 3. Preparare Dataset

```bash
bash scripts/prepare_all_splits.sh
```

### 4. Implementare Fine-tuning

Usa il codice SSVP-SLT come **reference** per implementare `finetune_how2sign.py`:

```python
# Studia questi file nel repo SSVP-SLT:
# - translation/train.py
# - translation/data/dataset.py
# - translation/models/

# Poi implementa il tuo script che:
# 1. Carica checkpoint pretrained
# 2. Setup dataloader per How2Sign
# 3. Fine-tune con training loop
# 4. Salva checkpoints
```

---

## 💡 Alternative

Se hai ancora problemi:

### Opzione 1: Ambiente Virtuale Separato

```bash
# Crea env dedicato per SSVP-SLT
python -m venv venv_ssvp
source venv_ssvp/bin/activate

# Installa solo per SSVP-SLT
pip install torch==2.2.2 torchvision==0.17.2
cd src/sign_to_text_ssvp/models/ssvp_slt_repo
pip install -r requirements.txt
pip install -e .
```

### Opzione 2: Usa Docker

```bash
# TODO: Creare Dockerfile per SSVP-SLT
# Ambiente isolato con tutte le dipendenze corrette
```

### Opzione 3: Usa Solo Reference

Non installare SSVP-SLT package, ma:

- Leggi il codice in `models/ssvp_slt_repo/`
- Copia/adatta le parti che ti servono
- Implementa fine-tuning usando paper come guida

---

## 📚 Documentazione Utile

- **SSVP-SLT README**: `models/ssvp_slt_repo/README.md`
- **Translation Guide**: `models/ssvp_slt_repo/translation/README.md`
- **Paper**: https://arxiv.org/abs/2402.09611
- **Nostro README**: `src/sign_to_text_ssvp/README.md`

---

## 🆘 Se Ancora Non Funziona

1. Condividi l'output completo di:

   ```bash
   bash scripts/install_ssvp_minimal.sh 2>&1 | tee install.log
   ```

2. Verifica versioni:

   ```bash
   python --version
   pip list | grep -E "torch|numpy|transformers"
   ```

3. Prova a:
   - Aggiornare pip: `pip install --upgrade pip`
   - Pulire cache: `pip cache purge`
   - Reinstallare in ordine specifico

---

## ✅ Status Attuale

- ✅ Script `install_ssvp.sh` aggiornato con better error handling
- ✅ Script `install_ssvp_minimal.sh` creato (raccomandato)
- ✅ Documentazione troubleshooting completa
- 🔄 Puoi procedere con installazione minimale

**Esegui ora**:

```bash
cd src/sign_to_text_ssvp
bash scripts/install_ssvp_minimal.sh
```
