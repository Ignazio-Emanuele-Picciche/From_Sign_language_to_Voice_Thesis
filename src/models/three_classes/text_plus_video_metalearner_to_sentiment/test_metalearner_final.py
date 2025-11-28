import pandas as pd
import os
import sys
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# --- PERCORSI (Assoluti) ---
LSTM_TEST = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/results/lstm_probs_for_fusion_test.csv"
ROBERTA_TEST = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/results/roberta_features_test_set.csv"
MODEL_PATH = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/models/metalearner/metalearner_decision_tree.joblib"
ENCODER_PATH = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/models/metalearner/label_encoder.joblib"
OUTPUT_PLOT = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/figures/metalearner/final_test_confusion_matrix.png"


def main():
    print("=== TEST FINALE MULTI-LEARNER (LATE FUSION) ===")
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
    # Selezioniamo le stesse colonne usate nel training
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

    # 6. Risultati
    acc = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 40)
    print(f"🏆 ACCURACY FINALE SUL TEST SET: {acc:.4f}")
    print("=" * 40)

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
