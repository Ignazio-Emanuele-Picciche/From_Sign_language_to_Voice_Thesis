"""
================================================================================================================
TRAINING DEL MULTI-LEARNER CON GRID SEARCH (OTTIMIZZAZIONE PARAMETRI)
================================================================================================================

DESCRIZIONE:
Questo script allena 4 Meta-Learner (DT, LR, RF, SVM) ma invece di usare parametri fissi,
cerca la combinazione migliore per ognuno usando GridSearchCV.

PERCHÉ LO FACCIAMO:
La Random Forest andava in overfitting (troppo complessa). Cercando i parametri ottimali
(es. riducendo max_depth) possiamo renderla più stabile e potente sul Test Set.

OUTPUT:
- Stampa i migliori parametri per ogni modello.
- Salva il modello vincitore assoluto ottimizzato.
================================================================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
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
OUTPUT_MODEL_DIR = "src/models/three_classes/text_plus_video_metalearner_to_sentiment/models/metalearner"
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


def load_and_merge_data():
    print("--- 1. Caricamento e Merge dei Dati ---")
    if not os.path.exists(LSTM_FEATURES_PATH) or not os.path.exists(
        ROBERTA_FEATURES_PATH
    ):
        raise FileNotFoundError("❌ File features non trovati!")

    df_lstm = pd.read_csv(LSTM_FEATURES_PATH)
    df_roberta = pd.read_csv(ROBERTA_FEATURES_PATH)

    merged_df = pd.merge(
        df_lstm, df_roberta, on="video_name", suffixes=("_lstm", "_roberta")
    )

    if "true_label_lstm" in merged_df.columns:
        merged_df["label"] = merged_df["true_label_lstm"]
    elif "true_label" in merged_df.columns:
        merged_df["label"] = merged_df["true_label"]

    print(f"✅ Merge completato. Shape: {merged_df.shape}")
    return merged_df


def run_grid_search(model, param_grid, X_train, y_train, model_name):
    print(f"\n🔍 Tuning {model_name}...")
    # cv=5 significa 5-Fold Cross Validation (molto robusto)
    # scoring='f1_weighted' ottimizza direttamente la metrica che ci interessa
    grid = GridSearchCV(
        model, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    print(f"   ✅ Migliori parametri: {grid.best_params_}")
    print(f"   ✅ Miglior Score CV (F1): {grid.best_score_:.4f}")

    return grid.best_estimator_


def evaluate_model(model, X_test, y_test, model_name, class_names):
    print(f"\n--- Valutazione Finale {model_name} (sul Validation Set) ---")
    y_pred = model.predict(X_test)

    f1_w = f1_score(y_test, y_pred, average="weighted")
    print(f"  Weighted F1-Score:     {f1_w:.4f}")

    return f1_w


def main():
    os.makedirs(os.path.dirname(OUTPUT_LOG_FILE), exist_ok=True)
    sys.stdout = Logger(OUTPUT_LOG_FILE)

    print("=" * 60)
    print("TRAINING CON GRID SEARCH (HYPERPARAMETER TUNING)")
    print("=" * 60)

    # 1. Dati
    df = load_and_merge_data()
    feature_cols = [c for c in df.columns if "lstm_prob" in c or "roberta_prob" in c]
    X = df[feature_cols]
    y = df["label"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    class_names = list(le.classes_)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 2. Definizione Griglie di Ricerca
    # Qui definiamo quali parametri provare per ogni modello

    # A. Decision Tree
    dt_params = {
        "max_depth": [3, 4, 5, 6, 8],
        "min_samples_leaf": [1, 5, 10],  # Evita regole troppo specifiche
        "criterion": ["gini", "entropy"],
    }

    # B. Logistic Regression
    lr_params = {
        "C": [0.01, 0.1, 1, 10, 100],  # Regolarizzazione (C basso = più semplice)
        "solver": ["lbfgs", "liblinear"],
    }

    # C. Random Forest (Focus su ridurre overfitting)
    rf_params = {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 4, 5, 6],  # Teniamolo basso! Prima era 8 o None
        "min_samples_leaf": [2, 5, 10],  # Più alto = meno overfitting
        "max_features": ["sqrt", "log2"],
    }

    # D. SVM
    svm_params = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.1, 0.01],
        "kernel": ["rbf", "linear"],
    }

    # 3. Esecuzione Grid Search
    models_config = [
        (
            "Decision Tree",
            DecisionTreeClassifier(class_weight="balanced", random_state=42),
            dt_params,
        ),
        (
            "Logistic Regression",
            LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
            lr_params,
        ),
        (
            "Random Forest",
            RandomForestClassifier(class_weight="balanced", random_state=42),
            rf_params,
        ),
        (
            "SVM",
            SVC(class_weight="balanced", probability=True, random_state=42),
            svm_params,
        ),
    ]

    results = {}
    best_estimators = {}

    for name, model, params in models_config:
        # Trova il modello migliore
        best_model = run_grid_search(model, params, X_train, y_train, name)
        best_estimators[name] = best_model

        # Valuta sul validation set tenuto da parte
        score = evaluate_model(best_model, X_val, y_val, name, class_names)
        results[name] = score

    # 4. Selezione Vincitore
    print("\n" + "=" * 60)
    print("CONFRONTO FINALE (Post-Tuning)")
    for name, score in results.items():
        print(f"{name:<25}: {score:.4f}")

    best_name = max(results, key=results.get)
    best_score = results[best_name]
    best_model_final = best_estimators[best_name]

    print("-" * 60)
    print(f"🏆 VINCITORE: {best_name} (F1: {best_score:.4f})")
    print("=" * 60)

    # 5. Salvataggio
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    # Salviamo con un nome generico o specifico? Usiamo specifico per chiarezza.
    # Ma per lo script di test finale, dovrai aggiornare il path se cambia il vincitore.
    model_filename = f"metalearner_{best_name.lower().replace(' ', '_')}_tuned.joblib"
    model_path = os.path.join(OUTPUT_MODEL_DIR, model_filename)
    encoder_path = os.path.join(OUTPUT_MODEL_DIR, "label_encoder.joblib")

    joblib.dump(best_model_final, model_path)
    joblib.dump(le, encoder_path)

    print(f"\n✅ Modello OTTIMIZZATO salvato in: {model_path}")
    print(
        f"⚠️  RICORDA: Aggiorna MODEL_PATH in 'test_metalearner_final.py' con questo nuovo file!"
    )


if __name__ == "__main__":
    main()
