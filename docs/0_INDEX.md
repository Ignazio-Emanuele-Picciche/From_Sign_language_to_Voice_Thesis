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

**Ultima revisione:** 25 Ottobre 2025  
**Autore:** Ignazio Emanuele Picciche  
**Progetto:** Improved EmoSign Thesis
