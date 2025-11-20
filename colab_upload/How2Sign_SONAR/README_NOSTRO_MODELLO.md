# 📘 Architettura del Modello di Tesi (SignHiera + SONAR Alignment)

Questo documento descrive l'architettura tecnica della pipeline di **Sign Language Translation (SLT)** sviluppata per la tesi.
L'obiettivo è tradurre video di lingua dei segni americana (ASL) in testo inglese, sfruttando la potenza dello spazio semantico multilingue di **SONAR**.

---

## 1. Panoramica Concettuale

Il nostro approccio si basa sul **Cross-Modal Alignment** (Allineamento tra modalità diverse).
Invece di addestrare un traduttore da zero (che richiederebbe milioni di video), insegniamo a un Encoder Video a "parlare la lingua" di SONAR.

**L'idea chiave:**
SONAR possiede uno "spazio vettoriale" dove frasi con lo stesso significato (in lingue diverse) finiscono vicine.
Noi addestriamo il nostro modello visivo a proiettare il video **nello stesso punto** in cui si troverebbe la frase inglese corrispondente.

---

## 2. Architettura Dettagliata

La pipeline è composta da 4 blocchi principali:

### A. Feature Extractor (Offline)

- **Modello**: **SignHiera** (ViT-Base pre-addestrato su Kinetics-400/ImageNet).
- **Ruolo**: Trasforma i pixel grezzi del video in vettori matematici ricchi di significato visivo.
- **Output**: Una sequenza di vettori `(T, 768)`, dove `T` è il numero di frame e `768` è la dimensione delle feature.
- **Nota**: Questo passaggio viene fatto una volta sola (`extract_features_signhiera.py`) per velocizzare il training.

### B. The Projector / Encoder (Trainable) 🧠

- **Cos'è**: È l'unica parte che addestriamo davvero.
- **Struttura**: Un **MLP (Multi-Layer Perceptron)** o un Transformer leggero.
  - Input: 768 (SignHiera)
  - Hidden: 512
  - Output: 1024 (Spazio SONAR)
- **Funzionamento**:
  1.  Prende le feature video.
  2.  Fa una media temporale (Pooling) per ottenere un singolo vettore per tutto il video.
  3.  Lo proietta nello spazio a 1024 dimensioni.
- **Obiettivo**: Produrre un vettore che sia indistinguibile da un embedding testuale di SONAR.

### C. SONAR Text Embedder (The Teacher / Oracle) 🎓

- **Stato**: Congelato (Frozen).
- **Ruolo**: Durante il training, legge la traduzione inglese corretta (Ground Truth) e ci dice: _"Ecco il vettore bersaglio che dovresti produrre"_.
- **Output**: Target Embedding `(1024)`.

### D. SONAR Text Decoder (The Judge) ⚖️

- **Stato**: Congelato (Frozen).
- **Ruolo**: Prende il vettore prodotto dal nostro Encoder e lo trasforma in testo leggibile.
- **Perché è congelato?**: È stato addestrato da Meta su miliardi di frasi. È robustissimo. Se provassimo a modificarlo con i nostri pochi dati, rischieremmo di "romperlo" (Catastrophic Forgetting).

---

## 3. La Strategia di Training (Loss Function)

Come insegniamo all'Encoder a imitare il Maestro? Usiamo una **Loss Composta** innovativa:

$$ \mathcal{L}_{total} = \mathcal{L}_{cosine} + \lambda \cdot \mathcal{L}\_{magnitude} $$

1.  **Cosine Loss (Direzione)**:
    - Misura l'angolo tra il vettore Video e il vettore Testo.
    - Assicura che il **significato** sia lo stesso (es. "Gatto" e non "Cane").
2.  **Magnitude Loss (Scala)**:
    - Misura la differenza di lunghezza (norma) dei vettori.
    - **Cruciale**: Il decoder SONAR si aspetta vettori con una "energia" specifica (Norma ~32). Se i nostri vettori sono troppo deboli (Norma ~1), il decoder genera silenzio o errori. Questa loss corregge questo problema.

---

## 4. Confronto con lo Stato dell'Arte (Paper SSVP-SLT)

Il paper di riferimento è _"SSVP-SLT: Self-Supervised Video Pretraining for Sign Language Translation"_ (ACL 2024).
Ecco come ci posizioniamo rispetto a loro:

| Caratteristica    | Paper SSVP-SLT (State of the Art)                                                 | Nostro Modello (Tesi "Efficiente")                                                   |
| :---------------- | :-------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------- |
| **Encoder Video** | **SSVP** (ViT Huge pre-addestrato su YouTube-ASL). Molto potente ma pesantissimo. | **SignHiera** (ViT Base pre-addestrato su azioni generiche). Più leggero e standard. |
| **Training**      | **2 Stadi**: <br>1. Alignment (come noi)<br>2. End-to-End (sbloccano il decoder)  | **1 Stadio**: <br>1. Alignment (Encoder Fine-Tuning)                                 |
| **Libreria**      | `fairseq` (Complessa, ricerca pura)                                               | `sonar-space` (API moderna, inferenza facile)                                        |
| **Risorse**       | Cluster di GPU A100                                                               | Singola GPU T4 (Google Colab)                                                        |
| **Obiettivo**     | Massimizzare il BLEU a ogni costo                                                 | Dimostrare la fattibilità con risorse limitate                                       |

### Perché il nostro approccio è valido per una tesi?

1.  **Replica la metodologia corretta**: Usiamo lo stesso principio di allineamento latente (Knowledge Distillation) del paper.
2.  **Efficienza**: Dimostriamo che si possono ottenere risultati di traduzione senza dover pre-addestrare modelli enormi su milioni di video (SSVP).
3.  **Innovazione**: L'uso della **Magnitude Loss** esplicita è un accorgimento tecnico specifico che abbiamo introdotto per far funzionare il training con l'API `sonar-space`.

---

## 5. Flusso dei Dati (Pipeline)

1.  **Video** (`.mp4`) ➔ **SignHiera** ➔ **Features** (`.npy`, 768-dim)
2.  **Features** ➔ **Nostro Encoder** ➔ **Predicted Embedding** (1024-dim)
3.  **Predicted Embedding** ➔ **SONAR Decoder** ➔ **Traduzione Finale** ("Hello world")

---

_Documento generato per la Tesi Magistrale "Improved EmoSign"._
_Autore: Ignazio Emanuele Piccichè & GitHub Copilot_
