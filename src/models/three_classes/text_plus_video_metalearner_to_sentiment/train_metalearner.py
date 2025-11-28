"""
================================================================================================================
TRAINING DEL MULTI-LEARNER (LATE FUSION / STACKING STRATEGY)
================================================================================================================

DESCRIZIONE:
Questo script allena il Meta-Learner che fonde le probabilità di LSTM (Video) e RoBERTa (Testo).

METRICHE CALCOLATE:
1.  Accuracy Standard
2.  Balanced Accuracy (Weighted Accuracy)
3.  Weighted F1-Score (Metrica decisiva)
4.  Cohen's Kappa (Affidabilità statistica)
5.  Classification Report (Precision/Recall per classe)
6.  Matrice di Confusione (Salvata come PNG)

OUTPUT:
- Modello migliore (.joblib)
- Label Encoder (.joblib)
- Grafici Matrici di Confusione (.png)
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

# --- SETUP ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, BASE_DIR)

# Percorsi
LSTM_FEATURES_PATH = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/lstm_features_train_set.csv"
ROBERTA_FEATURES_PATH = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/roberta_features_train_set.csv"
OUTPUT_MODEL_DIR = "models/metalearner"
OUTPUT_PLOTS_DIR = "reports/figures/metalearner"


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
    """
    Esegue predizioni e calcola tutte le metriche.
    """
    print(f"\n--- Valutazione Modello: {model_name} ---")

    y_pred = model.predict(X_test)

    # Metriche Scalari
    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)  # Weighted Accuracy
    f1_w = f1_score(y_test, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_test, y_pred)

    print(f"  Accuracy Standard:     {acc:.4f}")
    print(f"  Balanced Accuracy:     {bal_acc:.4f}")
    print(f"  Weighted F1-Score:     {f1_w:.4f}")
    print(f"  Cohen's Kappa:         {kappa:.4f}")

    print("\n  > Classification Report Detagliato:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    # Plot
    save_confusion_matrix(y_test, y_pred, class_names, model_name)

    return acc, bal_acc, f1_w, kappa


def load_and_merge_data():
    print("--- 1. Caricamento e Merge dei Dati ---")
    if not os.path.exists(LSTM_FEATURES_PATH) or not os.path.exists(
        ROBERTA_FEATURES_PATH
    ):
        raise FileNotFoundError(
            "❌ Mancano i file delle features! Esegui prima gli script di estrazione."
        )

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
    # 1. Preparazione Dati
    df = load_and_merge_data()

    feature_cols = [c for c in df.columns if "lstm_prob" in c or "roberta_prob" in c]
    print(f"\nFeature utilizzate ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols]
    y = df["label"]

    # Encoding Label
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = list(le.classes_)
    print(f"Classi codificate: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # Split Train/Val (Stratified è cruciale per mantenere le proporzioni delle classi)
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

    # Feature Importance (Tree)
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

    # Analisi Pesi (Logistic)
    print("\n  > [Logistic Regression] Pesi (Influenza media):")
    avg_coefs = np.mean(np.abs(log_clf.coef_), axis=0)
    for name, coef in zip(feature_cols, avg_coefs):
        print(f"    {name}: {coef:.4f}")

    metrics_log = evaluate_model(
        log_clf, X_val, y_val, "Logistic Regression", class_names
    )

    # ---------------------------------------------------------
    # SELEZIONE VINCITORE (Basata su Weighted F1)
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

    # Salvataggio
    model_path = os.path.join(OUTPUT_MODEL_DIR, f"metalearner_{best_name}.joblib")
    joblib.dump(best_model, model_path)
    joblib.dump(le, os.path.join(OUTPUT_MODEL_DIR, "label_encoder.joblib"))

    print(f"\n✅ Modello salvato in: {model_path}")


if __name__ == "__main__":
    main()
