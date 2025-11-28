"""
================================================================================================================
TRAINING DEL MULTI-LEARNER (LATE FUSION / STACKING STRATEGY)
================================================================================================================

DESCRIZIONE:
Questo script allena il Meta-Learner che fonde le probabilità di LSTM (Video) e RoBERTa (Testo).
Tutto l'output della console viene salvato automaticamente in un file di log.

PERCORSI FILE:
- Input Features: src/models/three_classes/text_plus_video_metalearner_to_sentiment/
- Output Modello: models/metalearner/
- Output Log: src/models/three_classes/text_plus_video_metalearner_to_sentiment/metalearner_training_log.txt
================================================================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import LabelEncoder

# --- SETUP BASE_DIR ---
# Calcola la root del progetto risalendo di 3 livelli da questo file (src/models/metalearner/)
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, BASE_DIR)

# --- CONFIGURAZIONE PERCORSI BLINDATA ---
# Usiamo os.path.join per essere sicuri che i percorsi siano sempre corretti,
# indipendentemente da dove lanci lo script.

LSTM_FEATURES_PATH = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/lstm_features_train_set.csv"

ROBERTA_FEATURES_PATH = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/roberta_features_train_set.csv"

OUTPUT_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "three_classes",
    "text_plus_video_metalearner_to_sentiment",
    "models",
    "metalearner",
)

OUTPUT_PLOTS_DIR = os.path.join(BASE_DIR, "reports", "figures", "metalearner")

OUTPUT_LOG_FILE = os.path.join(
    BASE_DIR,
    "src",
    "models",
    "three_classes",
    "text_plus_video_metalearner_to_sentiment",
    "metalearner_training_log.txt",
)


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


def save_confusion_matrix(y_true, y_pred, classes, model_name):
    """Genera e salva la matrice di confusione come immagine."""
    os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap="Blues", ax=ax, values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")

    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    save_path = os.path.join(OUTPUT_PLOTS_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"    📊 Matrice di confusione salvata in: {save_path}")


def evaluate_model(model, X_test, y_test, model_name, class_names):
    print(f"\n--- Valutazione Modello: {model_name} ---")

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    f1_w = f1_score(y_test, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_test, y_pred)

    print(f"  Accuracy Standard:     {acc:.4f}")
    print(f"  Balanced Accuracy:     {bal_acc:.4f}")
    print(f"  Weighted F1-Score:     {f1_w:.4f}")
    print(f"  Cohen's Kappa:         {kappa:.4f}")

    print("\n  > Classification Report Detagliato:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    save_confusion_matrix(y_test, y_pred, class_names, model_name)

    return acc, bal_acc, f1_w, kappa


def load_and_merge_data():
    print("--- 1. Caricamento e Merge dei Dati ---")
    print(
        f"  Cercando features in:\n   - {LSTM_FEATURES_PATH}\n   - {ROBERTA_FEATURES_PATH}"
    )

    if not os.path.exists(LSTM_FEATURES_PATH):
        raise FileNotFoundError(f"❌ File non trovato: {LSTM_FEATURES_PATH}")
    if not os.path.exists(ROBERTA_FEATURES_PATH):
        raise FileNotFoundError(f"❌ File non trovato: {ROBERTA_FEATURES_PATH}")

    df_lstm = pd.read_csv(LSTM_FEATURES_PATH)
    df_roberta = pd.read_csv(ROBERTA_FEATURES_PATH)

    # Merge Inner su video_name
    merged_df = pd.merge(
        df_lstm, df_roberta, on="video_name", suffixes=("_lstm", "_roberta")
    )

    # Label Normalization
    if "true_label_lstm" in merged_df.columns:
        merged_df["label"] = merged_df["true_label_lstm"]
    elif "true_label" in merged_df.columns:
        merged_df["label"] = merged_df["true_label"]

    print(f"✅ Merge completato. Shape finale: {merged_df.shape}")
    return merged_df


def main():
    # --- SETUP LOGGING ---
    os.makedirs(os.path.dirname(OUTPUT_LOG_FILE), exist_ok=True)
    sys.stdout = Logger(OUTPUT_LOG_FILE)

    print("=" * 60)
    print(f"LOG ATTIVATO: Output salvato in:\n{OUTPUT_LOG_FILE}")
    print("=" * 60)

    # 1. Preparazione Dati
    try:
        df = load_and_merge_data()
    except FileNotFoundError as e:
        print(e)
        return

    feature_cols = [c for c in df.columns if "lstm_prob" in c or "roberta_prob" in c]
    print(f"\nFeature utilizzate ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols]
    y = df["label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = list(le.classes_)
    print(f"Classi codificate: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # ---------------------------------------------------------
    # MODELLO A: Decision Tree
    # ---------------------------------------------------------
    tree_clf = DecisionTreeClassifier(
        max_depth=5, random_state=42, class_weight="balanced"
    )
    tree_clf.fit(X_train, y_train)

    print("\n  > [Decision Tree] Feature Importance:")
    for name, imp in zip(feature_cols, tree_clf.feature_importances_):
        if imp > 0.01:
            print(f"    {name}: {imp:.4f}")

    metrics_tree = evaluate_model(tree_clf, X_val, y_val, "Decision Tree", class_names)

    # ---------------------------------------------------------
    # MODELLO B: Logistic Regression
    # ---------------------------------------------------------
    log_clf = LogisticRegression(
        random_state=42, max_iter=1000, class_weight="balanced"
    )
    log_clf.fit(X_train, y_train)

    print("\n  > [Logistic Regression] Pesi (Influenza media):")
    avg_coefs = np.mean(np.abs(log_clf.coef_), axis=0)
    for name, coef in zip(feature_cols, avg_coefs):
        print(f"    {name}: {coef:.4f}")

    metrics_log = evaluate_model(
        log_clf, X_val, y_val, "Logistic Regression", class_names
    )

    # ---------------------------------------------------------
    # SELEZIONE VINCITORE
    # ---------------------------------------------------------
    t_f1 = metrics_tree[2]
    l_f1 = metrics_log[2]

    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("CONFRONTO FINALE (Metric: Weighted F1)")
    print(f"Decision Tree:       {t_f1:.4f}")
    print(f"Logistic Regression: {l_f1:.4f}")

    if l_f1 >= t_f1:
        print(f"🏆 VINCITORE: Logistic Regression")
        best_model = log_clf
        best_name = "logistic_regression"
    else:
        print(f"🏆 VINCITORE: Decision Tree")
        best_model = tree_clf
        best_name = "decision_tree"
    print("=" * 60)

    # Salvataggio con percorso ASSOLUTO
    model_path = os.path.join(OUTPUT_MODEL_DIR, f"metalearner_{best_name}.joblib")
    encoder_path = os.path.join(OUTPUT_MODEL_DIR, "label_encoder.joblib")

    joblib.dump(best_model, model_path)
    joblib.dump(le, encoder_path)

    print(f"\n✅ Modello salvato in: {model_path}")
    print(f"✅ Encoder salvato in: {encoder_path}")


if __name__ == "__main__":
    main()
