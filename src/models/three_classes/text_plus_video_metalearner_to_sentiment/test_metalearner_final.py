"""
================================================================================================================
TEST FINALE DEL MULTI-LEARNER (INFERENZA SU GOLDEN SET)
================================================================================================================

DESCRIZIONE:
Questo script rappresenta l'ultimo passo della pipeline di ricerca. Valuta le prestazioni del "Meta-Learner"
(il modello di fusione addestrato nello step precedente) sui dati di TEST reali (Golden Set), che il modello
non ha mai visto durante il training.

FLUSSO DI LAVORO:
1.  Carica le feature (probabilità Softmax) estratte precedentemente dal Test Set:
    - Video (LSTM): results/lstm_probs_for_fusion_test.csv
    - Testo (RoBERTa): results/roberta_features_test_set.csv
2.  Esegue il MERGE dei due dataset basandosi sulla chiave univoca 'video_name'.
3.  Carica il modello di fusione addestrato (es. Decision Tree) e l'encoder delle etichette.
4.  Genera le predizioni finali combinando le informazioni video e testo.
5.  Calcola le metriche definitive (Accuracy, Classification Report, Matrice di Confusione).

INPUT RICHIESTI:
- I due file CSV di features del test set (generati dagli script di estrazione).
- Il modello salvato in 'models/metalearner/metalearner_decision_tree.joblib'.

OUTPUT:
- Report testuale delle performance a video.
- Immagine della Matrice di Confusione finale salvata in 'reports/figures/metalearner/'.
================================================================================================================
"""

import pandas as pd
import os
import sys
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# --- PERCORSI FILE (Assoluti e Sicuri) ---
# Input: Features estratte dal Test Set
LSTM_TEST = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/data/test/lstm_features_test_set.csv"
ROBERTA_TEST = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/data/test/roberta_features_test_set.csv"

# Input: Modello e Encoder addestrati
MODEL_PATH = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/models/metalearner/metalearner_random_forest.joblib"
ENCODER_PATH = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/models/metalearner/label_encoder.joblib"
OUTPUT_PLOT = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/figures/metalearner/final_test_confusion_matrix.png"

OUTPUT_LOG_FILE = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/metalearner_final_test_log.txt"


# --- CLASSE LOGGER ---
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    # --- ATTIVAZIONE LOGGING ---
    os.makedirs(os.path.dirname(OUTPUT_LOG_FILE), exist_ok=True)
    sys.stdout = Logger(OUTPUT_LOG_FILE)

    print("=" * 60)
    print("=== TEST FINALE MULTI-LEARNER (LATE FUSION) ===")
    print(f"Log salvato in: {OUTPUT_LOG_FILE}")
    print("=" * 60)
    print(f"Modello usato: {MODEL_PATH}")

    # 1. Caricamento Features
    if not os.path.exists(LSTM_TEST):
        print(f"❌ Manca il file LSTM: {LSTM_TEST}")
        return
    if not os.path.exists(ROBERTA_TEST):
        print(f"❌ Manca il file RoBERTa: {ROBERTA_TEST}")
        return

    df_lstm = pd.read_csv(LSTM_TEST)
    df_roberta = pd.read_csv(ROBERTA_TEST)

    # 2. Merge (Inner Join)
    df = pd.merge(df_lstm, df_roberta, on="video_name", suffixes=("_lstm", "_roberta"))
    print(f"✅ Dati uniti. Campioni di test trovati: {len(df)}")

    # Gestione colonna label
    if "true_label_lstm" in df.columns:
        df["label"] = df["true_label_lstm"]
    elif "true_label" in df.columns:
        df["label"] = df["true_label"]

    # 3. Preparazione X e y
    feature_cols = [c for c in df.columns if "lstm_prob" in c or "roberta_prob" in c]
    print(f"Features usate ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols]

    # 4. Caricamento Modello & Encoder
    try:
        clf = joblib.load(MODEL_PATH)
        le = joblib.load(ENCODER_PATH)
    except FileNotFoundError:
        print("❌ Errore: Modello o Encoder non trovati. Esegui prima il training!")
        return

    # Encoding Label Reali (String -> Int)
    y_true = le.transform(df["label"])

    # 5. Predizione
    print("\nEsecuzione predizione...")
    y_pred = clf.predict(X)

    # 6. Risultati e Metriche Estese
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_w = f1_score(y_true, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_true, y_pred)

    print("\n" + "=" * 40)
    print(f"🏆 RISULTATI FINALI SUL TEST SET")
    print("=" * 40)
    print(f"  Accuracy Standard:     {acc:.4f}")
    print(f"  Balanced Accuracy:     {bal_acc:.4f} (Weighted Accuracy)")
    print(f"  Weighted F1-Score:     {f1_w:.4f}")
    print(f"  Cohen's Kappa:         {kappa:.4f}")

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=le.classes_))

    # Matrice di Confusione
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap="Greens", ax=ax, values_format="d")
    plt.title("Final Multimodal Confusion Matrix (Test Set)")

    os.makedirs(os.path.dirname(OUTPUT_PLOT), exist_ok=True)
    plt.savefig(OUTPUT_PLOT)
    print(f"\n📊 Matrice salvata in: {OUTPUT_PLOT}")


if __name__ == "__main__":
    main()
