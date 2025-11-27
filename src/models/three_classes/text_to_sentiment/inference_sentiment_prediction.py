"""
ANALISI DELLA DISTRIBUZIONE DEL SENTIMENT (VADER)
-------------------------------------------------

1. ANALISI DEI DATI QUANTITATIVA
Dai grafici di distribuzione emergono due pattern distinti ma correlati tra i due dataset:
   - Dominanza della classe 'NEUTRAL': Entrambi i dataset mostrano una forte preponderanza
     di etichette neutre (~63% per How2Sign, ~61% per ASLLRP).
   - Sbilanciamento Positivo/Negativo: C'è una sproporzione marcata tra le classi emotive.
     Il sentiment positivo è moderatamente rappresentato (28-33%), mentre il sentiment
     negativo è una "minority class" critica (solamente 4% in How2Sign e 11% in ASLLRP).

2. INTERPRETAZIONE DEL DOMINIO
   - How2Sign (Negative 4%): Essendo un dataset di video "instructional" (tutorial, spiegazioni),
     è naturale che il linguaggio sia prevalentemente fattuale (Neutro) o incoraggiante (Positivo).
     Manca quasi totalmente la conflittualità o la tristezza tipica di altri contesti.
   - ASLLRP (Negative 11%): Mostra una varianza emotiva leggermente superiore. Essendo basato
     su "utterances" (enunciati) e conversazioni, cattura una gamma più ampia di espressioni
     umane, inclusi disaccordo o negazione, ma rimane comunque fortemente sbilanciato.

3. IMPLICAZIONI PER IL MODELLO DI PREDIZIONE (Text-to-Emotion)
Addestrare un modello predittivo su questi dati "così come sono" comporta rischi specifici:
   A. Bias verso la Classe Maggioritaria: Il modello imparerà che scommettere su "NEUTRAL"
      è statisticamente la scelta più sicura. Otterremo un'alta "Accuracy" (es. 65%)
      semplicemente predicendo sempre Neutro, ma il modello sarà inutile per rilevare le emozioni.
   B. Scarsa capacità di rilevazione del Negativo (Low Recall): Con solo il 4% di esempi
      negativi in How2Sign, il modello non avrà abbastanza pattern per imparare cosa rende
      una frase "negativa". Rischiamo che i falsi negativi siano altissimi.
   C. Limitazioni di VADER su Traduzioni: VADER è ottimizzato per social media (inglese colloquiale).
      Applicarlo a traduzioni di Lingua dei Segni (che spesso sono grammaticalmente semplificate
      o "glossate") potrebbe aver appiattito alcune sfumature emotive verso il Neutro.

RACCOMANDAZIONI FUTURE:
Per il training, sarà necessario applicare tecniche di:
   - Data Augmentation (generare sinteticamente frasi negative).
   - Oversampling della classe minoritaria (ripescare gli esempi negativi più volte).
   - Utilizzo di metriche come F1-Score (macro) invece della semplice Accuracy per valutare le performance.
"""

"""
================================================================================
GLOBAL SENTIMENT VALIDATION: VADER (Rule-Based) vs. RoBERTa (Transformer)
================================================================================

AUTORE: [Tuo Nome]
DATA: Novembre 2025
CONTESTO: Analisi del sentiment su dataset di Lingua dei Segni (How2Sign + ASLLRP)

DESCRIZIONE GENERALE:
Questo script esegue un'analisi comparativa ("Inter-Model Agreement") su un Test Set
unificato per validare la robustezza delle etichette di sentiment.
Vengono caricati e fusi due dataset distinti per creare un unico corpus eterogeneo.
Le etichette generate da VADER vengono usate come "Proxy Ground Truth" per valutare
le performance di predizione del modello SOTA 'cardiffnlp/twitter-roberta-base'.

STRUTTURA DEL PROCESSO:
1.  DATA INGESTION & MERGE:
    Carica i file CSV elaborati dallo step precedente:
    - How2Sign (Instructional -> Dominio fattuale/positivo)
    - ASLLRP (Conversational -> Dominio misto)
    I due dataset vengono normalizzati e concatenati in un unico DataFrame globale.

2.  INFERENZA AI (RoBERTa):
    Utilizza il modello 'cardiffnlp/twitter-roberta-base-sentiment-latest'.
    Il modello (Transformer-based) analizza l'intero corpus unificato per gestire
    meglio le sfumature di contesto rispetto all'approccio lessicale di VADER.

3.  GENERAZIONE METRICHE GLOBALI:
    Calcola le metriche di similarità tra i due annotatori sull'intero set:
    - Weighted Accuracy: Accuratezza generale.
    - Balanced Accuracy: Fondamentale per monitorare la classe minoritaria (NEGATIVE).
    - Macro F1-Score: Per valutare la precisione media non pesata dalla frequenza.

OUTPUT GENERATI:
A.  Global CSV ('global_sentiment_comparison.csv'):
    File unico contenente: [ID] | [Testo] | [Fonte] | [Sentiment VADER] | [Sentiment RoBERTa]
    Include la colonna 'Source_Dataset' per tracciare l'origine del dato.

B.  Global Metrics Report ('global_metrics.txt'):
    Report statistico complessivo con breakdown delle performance e conteggi.

NOTE TECNICHE:
- L'approccio unificato garantisce che le metriche siano statisticamente più rilevanti,
  specialmente per le classi rare (Negative) che sono scarse nei singoli dataset.
- Classi mappate: POSITIVE, NEGATIVE, NEUTRAL.
================================================================================
"""


# --------------------------------------------------------------------------
# METRICA: COHEN'S KAPPA (Statistica di Accordo Inter-Annotatore)
# --------------------------------------------------------------------------
# DESCRIZIONE:
# Il Cohen's Kappa misura l'affidabilità dell'accordo tra due valutatori
# (in questo caso VADER e RoBERTa) escludendo la probabilità che siano
# d'accordo puramente per caso.
#
# PERCHÉ È FONDAMENTALE QUI:
# Il nostro dataset è fortemente sbilanciato verso la classe 'NEUTRAL' (60%+).
# Se entrambi i modelli predicessero sempre 'NEUTRAL' a caso, avrebbero un'alta
# "Accuracy" (60%) ma zero intelligenza. Il Kappa penalizza questo fenomeno.
# Un Kappa basso ci rivela che, al di là dei casi ovvi, i due modelli hanno
# logiche decisionali molto diverse (Lessicale vs Contestuale).
#
# SCALA DI INTERPRETAZIONE (Landis & Koch):
# <= 0.00 : Nessun accordo (o peggio del caso)
# 0.01 - 0.20 : Accordo Scarso
# 0.21 - 0.40 : Accordo Discreto
# 0.41 - 0.60 : Accordo Moderato (Tipico nel confronto Rule-based vs DL)
# 0.61 - 0.80 : Accordo Sostanziale
# 0.81 - 1.00 : Accordo Quasi Perfetto
# --------------------------------------------------------------------------


import pandas as pd
import torch
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    balanced_accuracy_score,
    confusion_matrix,  # <--- NUOVO
    cohen_kappa_score,  # <--- NUOVO
)
from tqdm import tqdm
import os

"""
================================================================================
GLOBAL SENTIMENT VALIDATION: VADER (Rule-Based) vs. RoBERTa (Transformer)
================================================================================
... (Mantieni pure la tua docstring aggiornata qui) ...
"""

# --- CONFIGURAZIONE ---
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 16
DEVICE = 0 if torch.cuda.is_available() else -1

print(f"Using device: {'GPU' if DEVICE == 0 else 'CPU'}")
print(f"Caricamento modello {MODEL_NAME}...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=MODEL_NAME,
    device=DEVICE,
    tokenizer=MODEL_NAME,
    max_length=512,
    truncation=True,
)


def map_roberta_label(label):
    return label.upper()


def load_and_normalize(file_path, id_col, text_col, gt_col, source_name):
    """
    Carica un CSV, rinomina le colonne per standardizzarle e aggiunge l'origine.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File non trovato: {file_path}")

    df = pd.read_csv(file_path)

    # Rinomina colonne per il merge
    df = df.rename(columns={id_col: "ID", text_col: "Text", gt_col: "Sentiment_GT"})

    # Seleziona solo colonne utili e aggiungi la fonte (per debug)
    df = df[["ID", "Text", "Sentiment_GT"]]
    df["Source_Dataset"] = source_name

    print(f" -> Caricato {source_name}: {len(df)} righe")
    return df


def run_global_analysis():
    # 1. CARICAMENTO E UNIONE DEI DATASET
    print("\n--- 1. Caricamento e Unione Dataset ---")

    try:
        # Carica How2Sign
        df_h2s = load_and_normalize(
            file_path="data/processed/text_to_sentiment/how2sign_sentiment_analyzed.csv",
            id_col="sentence_name",
            text_col="sentence",
            gt_col="sentiment",
            source_name="How2Sign",
        )

        # Carica ASLLRP
        df_asl = load_and_normalize(
            file_path="data/processed/text_to_sentiment/asllrp_sentiment_analyzed.csv",
            id_col="video_name",
            text_col="caption",
            gt_col="sentiment",
            source_name="ASLLRP",
        )

        # Unione (Concatenazione verticale)
        combined_df = pd.concat([df_h2s, df_asl], ignore_index=True)
        print(f"\n✅ DATASET UNIFICATO CREATO. Totale campioni: {len(combined_df)}")

    except Exception as e:
        print(f"Errore nel caricamento dei dati: {e}")
        return

    # 2. INFERENZA SU TUTTO IL DATASET
    print("\n--- 2. Esecuzione Inferenza RoBERTa (Global) ---")
    texts = combined_df["Text"].astype(str).tolist()
    predictions = []

    for i in tqdm(range(0, len(texts), BATCH_SIZE)):
        batch_texts = texts[i : i + BATCH_SIZE]
        results = sentiment_pipeline(batch_texts)
        for res in results:
            predictions.append(map_roberta_label(res["label"]))

    combined_df["Sentiment_Pred_RoBERTa"] = predictions

    # 3. SALVATAGGIO CSV UNICO
    output_dir = "src/models/three_classes/text_to_sentiment/"
    # Creiamo la cartella se non esiste
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, "global_sentiment_comparison.csv")
    combined_df.to_csv(output_csv, index=False)
    print(f"✅ File confronto salvato: {output_csv}")

    # 4. CALCOLO METRICHE GLOBALI E COMPARAZIONE
    print("\n--- 3. Calcolo Metriche Globali & Agreement ---")
    y_true = combined_df["Sentiment_GT"].str.upper()
    y_pred = combined_df["Sentiment_Pred_RoBERTa"]

    # Metriche Standard
    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    # Metriche Avanzate di Confronto
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=["NEGATIVE", "NEUTRAL", "POSITIVE"])

    report = classification_report(
        y_true, y_pred, target_names=["NEGATIVE", "NEUTRAL", "POSITIVE"]
    )

    # Creazione testo report
    metrics_text = (
        f"GLOBAL METRICS REPORT (Unified Test Set)\n"
        f"========================================\n"
        f"Model: {MODEL_NAME}\n"
        f"Sources Included: How2Sign + ASLLRP\n"
        f"Total Samples: {len(combined_df)}\n\n"
        f"GLOBAL PERFORMANCE:\n"
        f"-----------------\n"
        f"Accuracy (Standard):        {acc:.4f}\n"
        f"Balanced Accuracy:          {balanced_acc:.4f}\n"
        f"Cohen's Kappa (Agreement):  {kappa:.4f}\n"
        f"F1 Score (Weighted):        {f1_weighted:.4f}\n\n"
        f"CONFUSION MATRIX (Rows=VADER, Cols=RoBERTa):\n"
        f"-----------------\n"
        f"Labels: [NEGATIVE, NEUTRAL, POSITIVE]\n"
        f"{cm}\n\n"
        f"DETAILED CLASS REPORT:\n"
        f"-----------------\n"
        f"{report}\n\n"
        f"DATASET BREAKDOWN:\n"
        f"{combined_df['Source_Dataset'].value_counts().to_string()}\n"
    )

    # Salvataggio Metriche TXT
    output_txt = os.path.join(output_dir, "global_metrics.txt")
    with open(output_txt, "w") as f:
        f.write(metrics_text)

    print(f"✅ Report metriche globali salvato: {output_txt}")
    print("-" * 30)
    print(metrics_text)

    # 5. SALVATAGGIO DEI DISACCORDI (Nuova Sezione)
    print("\n--- 4. Estrazione Disaccordi ---")
    disagreements = combined_df[
        combined_df["Sentiment_GT"].str.upper() != combined_df["Sentiment_Pred_RoBERTa"]
    ]

    output_disagreements = os.path.join(
        output_dir, "global_sentiment_disagreements.csv"
    )
    disagreements.to_csv(output_disagreements, index=False)
    print(
        f"✅ File disaccordi salvato ({len(disagreements)} righe): {output_disagreements}"
    )


if __name__ == "__main__":
    run_global_analysis()
