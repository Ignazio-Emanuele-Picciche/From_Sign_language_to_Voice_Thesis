"""
================================================================================================================
TRAINING DEL MULTI-LEARNER (LATE FUSION / STACKING STRATEGY)
================================================================================================================

DESCRIZIONE:
Questo script allena 4 diversi Meta-Learner (Decision Tree, Logistic Regression, Random Forest, SVM)
sulle probabilità fornite da LSTM e RoBERTa.
Confronta le prestazioni (Weighted F1) e salva solo il modello migliore.

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

# Importiamo i nuovi modelli
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, BASE_DIR)

# --- CONFIGURAZIONE PERCORSI BLINDATA ---
# Usiamo os.path.join per essere sicuri che i percorsi siano sempre corretti,
# indipendentemente da dove lanci lo script.

LSTM_FEATURES_PATH = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/data/train/lstm_features_train_set.csv"

ROBERTA_FEATURES_PATH = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/data/train/roberta_features_train_set.csv"

OUTPUT_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "three_classes",
    "text_plus_video_metalearner_to_sentiment",
    "models",
    "metalearner",
)

OUTPUT_PLOTS_DIR = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/figures/metalearner"

OUTPUT_LOG_FILE = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/metalearner_training_log.txt"


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

    if "true_label_lstm" in merged_df.columns:
        merged_df["label"] = merged_df["true_label_lstm"]
    elif "true_label" in merged_df.columns:
        merged_df["label"] = merged_df["true_label"]

    print(f"✅ Merge completato. Shape finale: {merged_df.shape}")
    return merged_df


# --- FUNZIONI DI TRAINING ---


def train_decision_tree(X_train, y_train):
    print("\nTraining Decision Tree...")
    clf = DecisionTreeClassifier(max_depth=5, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf


def train_logistic_regression(X_train, y_train):
    print("\nTraining Logistic Regression...")
    clf = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf


def train_random_forest(X_train, y_train):
    print("\nTraining Random Forest...")
    # n_estimators=100 è un buon default. Max depth limita l'overfitting.
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    return clf


def train_svm(X_train, y_train):
    print("\nTraining SVM (Support Vector Machine)...")
    # Kernel RBF è standard per dati non lineari. probability=True serve se volessimo le prob in output.
    clf = SVC(
        kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=42
    )
    clf.fit(X_train, y_train)
    return clf


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

    # 2. Addestramento e Valutazione dei 4 Modelli
    models_results = {}
    trained_models = {}

    # --- MODELLO A: Decision Tree ---
    model_dt = train_decision_tree(X_train, y_train)
    _, _, f1_dt, _ = evaluate_model(
        model_dt, X_val, y_val, "Decision Tree", class_names
    )
    models_results["decision_tree"] = f1_dt
    trained_models["decision_tree"] = model_dt

    # --- MODELLO B: Logistic Regression ---
    model_lr = train_logistic_regression(X_train, y_train)
    _, _, f1_lr, _ = evaluate_model(
        model_lr, X_val, y_val, "Logistic Regression", class_names
    )
    models_results["logistic_regression"] = f1_lr
    trained_models["logistic_regression"] = model_lr

    # --- MODELLO C: Random Forest ---
    model_rf = train_random_forest(X_train, y_train)
    _, _, f1_rf, _ = evaluate_model(
        model_rf, X_val, y_val, "Random Forest", class_names
    )
    models_results["random_forest"] = f1_rf
    trained_models["random_forest"] = model_rf

    # --- MODELLO D: SVM ---
    model_svm = train_svm(X_train, y_train)
    _, _, f1_svm, _ = evaluate_model(model_svm, X_val, y_val, "SVM", class_names)
    models_results["svm"] = f1_svm
    trained_models["svm"] = model_svm

    # 3. Selezione Vincitore
    print("\n" + "=" * 60)
    print("CONFRONTO FINALE (Metric: Weighted F1)")
    for name, score in models_results.items():
        print(f"{name.replace('_', ' ').title():<25}: {score:.4f}")

    # Trova la chiave col valore massimo
    best_name = max(models_results, key=models_results.get)
    best_score = models_results[best_name]
    best_model = trained_models[best_name]

    print("-" * 60)
    print(f"🏆 VINCITORE: {best_name.replace('_', ' ').title()} (F1: {best_score:.4f})")
    print("=" * 60)

    # 4. Salvataggio
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

    model_path = os.path.join(OUTPUT_MODEL_DIR, f"metalearner_{best_name}.joblib")
    encoder_path = os.path.join(OUTPUT_MODEL_DIR, "label_encoder.joblib")

    joblib.dump(best_model, model_path)
    joblib.dump(le, encoder_path)

    print(f"\n✅ Modello salvato in: {model_path}")
    print(f"✅ Encoder salvato in: {encoder_path}")


if __name__ == "__main__":
    main()
