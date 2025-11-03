# 📚 Documentazione TTS Emotion-Aware System - Indice Completo

**Progetto:** Improved EmoSign Thesis  
**Sistema:** ViViT → TTS Emotion-Aware → Audio Explainability  
**Ultimo aggiornamento:** 25 Ottobre 2025

---

## 🎯 QUICK START

**Vuoi capire velocemente cosa è stato fatto?**

1. **Leggi prima:** [`PROSODY_OPTIMIZATION_SUMMARY.md`](PROSODY_OPTIMIZATION_SUMMARY.md) (5 min)
2. **Confronto visuale:** [`COMPARISON_CONSERVATIVE_VS_MODERATE.txt`](COMPARISON_CONSERVATIVE_VS_MODERATE.txt) (2 min)
3. **Grafici:** [`../results/grid_search/analysis/`](../results/grid_search/analysis/) (1 min)

---

## 📖 DOCUMENTI PRINCIPALI

### 1. Report Analisi Completa

**File:** [`tts_detailed_analysis_report.md`](tts_detailed_analysis_report.md)  
**Lunghezza:** ~1500 righe (40+ pagine)  
**Contenuto:**

- Architettura sistema completo
- Dataset ASLLRP (200 video): Ground truth vs Predictions
- Metodologia TTS + Audio Explainability
- Risultati statistici (Pitch p<0.001\*\*\*, Energy ns)
- Discussione limitazioni Edge-TTS
- Confronto letteratura
- Considerazioni ViViT accuracy (56.5%)
- Raccomandazioni pre-deployment

**Quando usarlo:**

- Per scrivere capitoli tesi
- Per citazioni precise
- Per dettagli metodologici completi

---

### 2. Ottimizzazione Parametri Prosodici

#### 2.1 Report Completo Ottimizzazione

**File:** [`tts_prosody_optimization_report.md`](tts_prosody_optimization_report.md)  
**Lunghezza:** ~950 righe (25 pagine)  
**Contenuto:**

- Motivazione ottimizzazione (parametri euristici → empirici)
- Metodologia grid search (Conservative, Moderate, Aggressive)
- Risultati test set (n=10): Moderate VINCE (d=0.708)
- Risultati dataset completo (n=200): Conservative ≈ Moderate
- Interpretazione ceiling effect Edge-TTS
- Analisi limitazioni application accuracy (~20-30%)
- Raccomandazioni finali

**Quando usarlo:**

- Per capitolo "Ottimizzazione" in tesi
- Per spiegare scelte metodologiche
- Per difendere parametri scelti

#### 2.2 Summary Esecutivo

**File:** [`PROSODY_OPTIMIZATION_SUMMARY.md`](PROSODY_OPTIMIZATION_SUMMARY.md)  
**Lunghezza:** ~350 righe (5 pagine)  
**Contenuto:**

- Raccomandazione immediata (adotta MODERATE)
- Evidenze grid search (tabelle comparative)
- Grafici embedded
- Azioni immediate (checklist breve)
- Documentazione tesi (template sezioni)

**Quando usarlo:**

- Quick reference rapido
- Per presentazioni/slide
- Per discussioni con supervisore

#### 2.3 Confronto Visuale ASCII

**File:** [`COMPARISON_CONSERVATIVE_VS_MODERATE.txt`](COMPARISON_CONSERVATIVE_VS_MODERATE.txt)  
**Lunghezza:** ~150 righe  
**Contenuto:**

- Tabelle ASCII ben formattate
- Confronto parametri (Conservative vs Moderate vs Aggressive)
- Risultati grid search + dataset completo
- Interpretazione + conclusioni
- File generati (lista completa)

**Quando usarlo:**

- Vista d'insieme rapida
- Per stampe/appendici
- Per README

#### 2.4 Checklist Implementazione

**File:** [`PROSODY_OPTIMIZATION_CHECKLIST.md`](PROSODY_OPTIMIZATION_CHECKLIST.md)  
**Lunghezza:** ~650 righe  
**Contenuto:**

- Fase 1: ✅ Validazione (COMPLETATA)
- Fase 2: Implementazione codice
- Fase 3: Rigenerazione audio (OPZIONALE)
- Fase 4: Aggiornamento documentazione
- Fase 5: Documentazione tesi
- Fase 6: Presentazione (slide + demo)

**Quando usarlo:**

- Per tracking progressi
- Per non dimenticare step
- Prima della consegna tesi

---

### 3. Guide Workflow

#### 3.1 Workflow Completo TTS

**File:** [`tts_complete_workflow.md`](tts_complete_workflow.md)  
**Lunghezza:** ~600 righe  
**Contenuto:**

- Setup ambiente (edge-tts, parselmouth, librosa)
- Pipeline end-to-end (ViViT → TTS → Analysis)
- Scripts disponibili
- Comandi pratici
- Troubleshooting

**Quando usarlo:**

- Per replicare esperimenti
- Per onboarding nuovi membri
- Per debugging

#### 3.2 Summary Tesi

**File:** [`tts_thesis_summary.md`](tts_thesis_summary.md)  
**Lunghezza:** ~400 righe  
**Contenuto:**

- Tabelle LaTeX ready-to-use
- Formule statistiche
- Grafici con caption
- Contributi chiave

**Quando usarlo:**

- Per copy-paste in LaTeX
- Per sezioni risultati/appendici

---

### 4. Guide Tecniche

#### 4.1 Qwen-VL Inference

**File:** [`qwen_vl_inference_guide.md`](qwen_vl_inference_guide.md)  
**Contenuto:** Setup e uso Qwen-VL per visual question answering

#### 4.2 Audio Explainability

**File:** [`tts_audio_explainability_guide.md`](tts_audio_explainability_guide.md)  
**Contenuto:** Framework validazione quantitativa TTS

#### 4.3 ViViT Training/Testing

**File:**

- [`vivit_training_testing_guide.md`](vivit_training_testing_guide.md) (7 classi)
- [`vivit_training_testing_guide_2classes.md`](vivit_training_testing_guide_2classes.md) (2 classi)

**Contenuto:** Training, testing, evaluation ViViT classifier

---

## 📊 RISULTATI E DATI

### Grid Search Results

**Directory:** [`../results/grid_search/`](../results/grid_search/)

**File principali:**

- `README.md` - Documentazione esperimento
- `conservative/` - 10 audio baseline
- `moderate/` - 10 audio ottimizzati
- `aggressive/` - 10 audio massimi
- `analysis/config_comparison_boxplots.png` - Grafici boxplot
- `analysis/effect_sizes_comparison.png` - Grafici Cohen's d
- `analysis/configuration_comparison.csv` - Statistiche

### Dataset Completo Results

**Directory:** [`../results/analysis/`](../results/analysis/)

**File principali:**

- `audio_analysis_results.csv` - 200 campioni features acustiche
- `emotion_comparison_plots.png` - Boxplot Positive vs Negative
- `statistical_report.txt` - T-test, Cohen's d, normality tests

---

## 💻 CODICE SORGENTE

### TTS Generation

**File:** [`../src/tts/`](../src/tts/)

- `emotion_mapper.py` - ✅ Parametri MODERATE (aggiornati)
- `tts_generator.py` - Edge-TTS wrapper
- `text_templates.py` - Template testo + cleaning

### Analysis

**File:** [`../src/analysis/`](../src/analysis/)

- `prosody_grid_search.py` - Script grid search completo
- `run_analysis.py` - Analisi dataset completo
- `audio_comparison.py` - Comparazione audio
- `statistical_tests.py` - T-test, Cohen's d

### Explainability

**File:** [`../src/explainability/audio/`](../src/explainability/audio/)

- `acoustic_analyzer.py` - Feature extraction (Praat + Librosa)

---

## 🎓 PER LA TESI

### Capitoli da Scrivere

| Capitolo            | File Sorgente                                                                      | Sezioni Chiave                                     |
| ------------------- | ---------------------------------------------------------------------------------- | -------------------------------------------------- |
| **1. Introduzione** | `tts_detailed_analysis_report.md` § 1                                              | Contesto, Motivazione, Obiettivi                   |
| **2. Architettura** | `tts_detailed_analysis_report.md` § 2                                              | Pipeline, Componenti                               |
| **3. Dataset**      | `tts_detailed_analysis_report.md` § 3                                              | Ground truth, ViViT predictions, Imbalance         |
| **4. Metodologia**  | `tts_prosody_optimization_report.md` § 2                                           | Grid search, Analisi acustica                      |
| **5. Risultati**    | `tts_detailed_analysis_report.md` § 4 + `tts_prosody_optimization_report.md` § 3-4 | Pitch p<0.001, Cohen's d, Moderate wins            |
| **6. Discussione**  | `tts_detailed_analysis_report.md` § 5-6                                            | Limitazioni, Confronto letteratura, ViViT accuracy |
| **7. Conclusioni**  | `tts_detailed_analysis_report.md` § 13                                             | Contributi, Lavori futuri                          |

### Tabelle LaTeX

**Fonte:** `tts_thesis_summary.md`

Tabelle ready-to-use:

- Parametri prosodici (Conservative vs Moderate)
- Risultati statistici (Pitch, Energy)
- Confronto configurazioni grid search
- Confusion matrix ViViT

### Grafici

**Fonte:** `results/grid_search/analysis/` + `results/analysis/`

Grafici disponibili:

- Boxplot comparison (3 configs)
- Effect sizes comparison
- Emotion comparison plots (Positive vs Negative)

---

## 📝 WORKFLOW CONSEGNA TESI

### 1. Prima Stesura (1-2 settimane)

- [ ] Scrivere capitoli 1-3 da `tts_detailed_analysis_report.md`
- [ ] Inserire tabelle LaTeX da `tts_thesis_summary.md`
- [ ] Includere grafici da `results/`

### 2. Revisione (1 settimana)

- [ ] Controllare coerenza tra capitoli
- [ ] Verificare citazioni
- [ ] Aggiornare bibliografia

### 3. Finalizzazione (3 giorni)

- [ ] Formattazione finale LaTeX/Word
- [ ] Controllo grammatica/sintassi
- [ ] Generazione PDF
- [ ] Stampa

### 4. Presentazione (1 settimana)

- [ ] Creare slide da `PROSODY_OPTIMIZATION_SUMMARY.md`
- [ ] Preparare demo audio
- [ ] Provare timing (15-20 min)
- [ ] Anticipare domande

---

## 🔗 LINKS UTILI

### Repository

- GitHub: [Ignazio-Emanuele-Picciche/Improved_EmoSign_Thesis](https://github.com/Ignazio-Emanuele-Picciche/Improved_EmoSign_Thesis)
- Branch: `dev`

### Tools

- Edge-TTS: https://github.com/rany2/edge-tts
- Praat-Parselmouth: https://parselmouth.readthedocs.io/
- Librosa: https://librosa.org/

### Letteratura

- Scherer (1986): Vocal affect expression
- Burkhardt et al. (2005): Emotional TTS
- Cohen (1988): Statistical power analysis (Cohen's d)

---

## 📞 CONTATTI & SUPPORTO

**Problemi con:**

- Codice: controllare `../src/README.md` (se esiste)
- Dati: verificare `../results/README.md`
- Setup: seguire `tts_complete_workflow.md`

**Domande frequenti:**

- "Come rigenero gli audio?" → `PROSODY_OPTIMIZATION_CHECKLIST.md` Fase 3
- "Come re-run analisi?" → `tts_complete_workflow.md` § Comandi
- "Quale config usare?" → `PROSODY_OPTIMIZATION_SUMMARY.md` → MODERATE
- "Perché risultati identici?" → `COMPARISON_CONSERVATIVE_VS_MODERATE.txt` § 4

---

---

## 🆕 NUOVA PIPELINE: VIDEO-TO-TEXT (Sign Language)

### Quick Start

**File:** [`QUICKSTART_SIGN_TO_TEXT.md`](QUICKSTART_SIGN_TO_TEXT.md)  
**Lunghezza:** ~300 righe  
**Contenuto:**

- Risposta alla domanda: "Ha senso usare sign-language-translator?"
- Strategia raccomandata (feature extraction + nostro modello)
- Primi passi immediati (installazione + test)
- Architettura proposta (Seq2Seq Transformer)
- Timeline semplificata (3.5 mesi)

**Quando usarlo:**

- Per capire se usare la libreria (YES, ma solo per landmarks)
- Per iniziare subito con i test
- Per overview architettura sistema

### Roadmap Completa

**File:** [`VIDEO_TO_TEXT_PIPELINE_ROADMAP.md`](VIDEO_TO_TEXT_PIPELINE_ROADMAP.md)  
**Lunghezza:** ~1200 righe (35 pagine)  
**Contenuto:**

- **FASE 1:** Setup e Analisi (2 settimane)
- **FASE 2:** Feature Extraction Pipeline (2 settimane)
- **FASE 3:** Modello Sign-to-Text (4 settimane)
- **FASE 4:** Training e Validazione (3 settimane)
- **FASE 5:** Pipeline End-to-End (2 settimane)
- **FASE 6:** Ottimizzazione e Deployment (2 settimane)
- Alternative (Plan B, Transfer Learning, Rule-based)
- Metriche di successo (BLEU, WER)
- Stack tecnologico completo
- Rischi e mitigazioni
- Collegamento con tesi (capitoli, contributi scientifici)

**Quando usarlo:**

- Per pianificazione dettagliata progetto
- Per timeline e milestone
- Per scrivere proposal/introduzione tesi
- Per discussioni con supervisore

### Script di Test

**File:** [`../test_sign_language_extraction.py`](../test_sign_language_extraction.py)  
**Contenuto:**

- Test feature extraction su singolo video
- Batch processing su dataset sample
- Salvataggio landmarks in formato .npz
- Visualizzazione 3D landmarks
- Statistiche e validazione output

**Comandi:**

```bash
# Test singolo video
python test_sign_language_extraction.py \
    --mode single \
    --video_path data/raw/ASLLRP/videos/83664512.mp4

# Test batch (5 samples)
python test_sign_language_extraction.py \
    --mode batch \
    --n_samples 5
```

### Analisi Libreria sign-language-translator

**Punti chiave:**

✅ **POSSIAMO USARE:**

- MediaPipeLandmarksModel (estrazione landmarks 3D/2D)
- Video processing utilities
- Framework di riferimento per architettura

❌ **NON POSSIAMO USARE:**

- Sign-to-Text diretto (NON IMPLEMENTATO)
- Modelli pre-trained ASL→Text (NON DISPONIBILI)

**Conclusione:** Usare SLT come **tool di feature extraction**, non come soluzione completa. Il modello Sign-to-Text lo implementiamo noi (Seq2Seq Transformer).

### Dataset Disponibile

**Ground Truth:** `data/processed/golden_label_sentiment.csv`

- 202 video ASL annotati
- Colonne: video_name, caption, emotion
- Perfetto per training/validation Seq2Seq

### Integrazione con Progetto Esistente

```
Pipeline Completa:
Video ASL → Landmarks → Testo → Emozione → Audio TTS
            (SLT)      (NUOVO)  (ViViT)   (Edge-TTS)
```

**Contributi scientifici:**

1. Modello Seq2Seq per ASL-to-Text
2. Pipeline multi-modale completa
3. Analisi consistenza emozioni cross-modal
4. Benchmark su dataset annotato custom

---

**Ultima revisione:** 1 Novembre 2025  
**Autore:** Ignazio Emanuele Picciche + GitHub Copilot  
**Progetto:** Improved EmoSign Thesis

```

```
