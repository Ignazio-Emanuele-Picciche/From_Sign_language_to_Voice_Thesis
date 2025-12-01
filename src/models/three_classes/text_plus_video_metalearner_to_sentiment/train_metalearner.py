"""
================================================================================================================
TRAINING DEL MULTI-LEARNER CON GRID SEARCH + VALIDATION SET ESPLICITO
================================================================================================================

DESCRIZIONE:
Questo script allena 4 Meta-Learner usando GridSearchCV per trovare i migliori iperparametri.
Invece di splittare il training set, usa un VALIDATION SET dedicato per la valutazione finale.

FLUSSO:
1. Carica TRAIN SET (LSTM + RoBERTa) -> Usa 5-Fold CV per il Tuning.
2. Carica VALIDATION SET (LSTM + RoBERTa) -> Usa questo per decretare il vincitore.
3. Salva il modello migliore.

PERCORSI FILE:
- Train: data/train/...
- Val:   data/val/...
================================================================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
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

# --- PERCORSI ---
PATH_PREFIX = "src/models/three_classes/text_plus_video_metalearner_to_sentiment"

# TRAIN PATHS
LSTM_TRAIN_PATH = os.path.join(PATH_PREFIX, "data/train/lstm_features_train_set.csv")
ROBERTA_TRAIN_PATH = os.path.join(
    PATH_PREFIX, "data/train/roberta_features_train_set.csv"
)

# VAL PATHS (Nuovi)
LSTM_VAL_PATH = os.path.join(PATH_PREFIX, "data/val/lstm_features_val_set.csv")
ROBERTA_VAL_PATH = os.path.join(PATH_PREFIX, "data/val/roberta_features_val_set.csv")

# OUTPUTS
OUTPUT_MODEL_DIR = os.path.join(PATH_PREFIX, "models", "metalearner")
OUTPUT_PLOTS_DIR = os.path.join(PATH_PREFIX, "reports", "figures", "metalearner_tuning")
OUTPUT_LOG_FILE = os.path.join(PATH_PREFIX, "metalearner_gridsearch_log.txt")


# --- LOGGER ---
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


def load_and_merge(lstm_path, roberta_path, dataset_name="Dataset"):
    print(f"--- Caricamento {dataset_name} ---")
    if not os.path.exists(lstm_path) or not os.path.exists(roberta_path):
        raise FileNotFoundError(
            f"❌ File non trovati per {dataset_name}:\n {lstm_path}\n {roberta_path}"
        )

    df_lstm = pd.read_csv(lstm_path)
    df_roberta = pd.read_csv(roberta_path)

    merged_df = pd.merge(
        df_lstm, df_roberta, on="video_name", suffixes=("_lstm", "_roberta")
    )

    if "true_label_lstm" in merged_df.columns:
        merged_df["label"] = merged_df["true_label_lstm"]
    elif "true_label" in merged_df.columns:
        merged_df["label"] = merged_df["true_label"]

    print(f"✅ {dataset_name} caricato. Shape: {merged_df.shape}")
    return merged_df


def run_grid_search(model, param_grid, X_train, y_train, model_name):
    print(f"\n🔍 Tuning {model_name} (5-Fold CV su Train)...")
    grid = GridSearchCV(
        model, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)

    print(f"   ✅ Migliori parametri: {grid.best_params_}")
    print(f"   ✅ Miglior Score CV (Train): {grid.best_score_:.4f}")

    return grid.best_estimator_


def log_feature_importance(model, feature_names, model_name):
    """
    Estrae e stampa l'importanza delle feature o i coefficienti per la tesi.
    """
    print(f"\n📊 Analisi Pesi/Importanza per {model_name}:")

    # 1. Modelli basati su alberi (Decision Tree, Random Forest)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        # Ordiniamo per importanza decrescente per leggibilità
        indices = np.argsort(importances)[::-1]

        for i in indices:
            print(f"   {feature_names[i]:<30}: {importances[i]:.4f}")

    # 2. Modelli lineari (Logistic Regression, SVM Lineare)
    elif hasattr(model, "coef_"):
        # coef_ è (n_classes, n_features). Facciamo la media assoluta per l'impatto globale.
        avg_coefs = np.mean(np.abs(model.coef_), axis=0)
        indices = np.argsort(avg_coefs)[::-1]

        print("   (Media assoluta dei coefficienti su tutte le classi)")
        for i in indices:
            print(f"   {feature_names[i]:<30}: {avg_coefs[i]:.4f}")

    # 3. Altri (es. SVM RBF)
    else:
        print("   ⚠️  Pesi diretti non disponibili (es. Kernel non lineare).")


def evaluate_model(model, X_val, y_val, model_name, class_names):
    print(f"\n--- Valutazione {model_name} su VALIDATION SET ---")
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    f1_w = f1_score(y_val, y_pred, average="weighted")
    kappa = cohen_kappa_score(y_val, y_pred)

    print(f"  Accuracy:              {acc:.4f}")
    print(f"  Weighted F1-Score:     {f1_w:.4f}")
    print(f"  Cohen Kappa Score:     {kappa:.4f}")

    return f1_w, kappa


def main():
    os.makedirs(os.path.dirname(OUTPUT_LOG_FILE), exist_ok=True)
    sys.stdout = Logger(OUTPUT_LOG_FILE)

    print("=" * 60)
    print("TRAINING GRID SEARCH + VALIDATION SET ESPLICITO")
    print("=" * 60)

    # 1. Caricamento Dati
    df_train = load_and_merge(LSTM_TRAIN_PATH, ROBERTA_TRAIN_PATH, "TRAIN SET")
    df_val = load_and_merge(LSTM_VAL_PATH, ROBERTA_VAL_PATH, "VAL SET")

    # 2. Preparazione X, y
    feature_cols = [
        c for c in df_train.columns if "lstm_prob" in c or "roberta_prob" in c
    ]
    print(f"\nFeature utilizzate ({len(feature_cols)}): {feature_cols}")

    X_train = df_train[feature_cols]
    y_train_raw = df_train["label"]

    X_val = df_val[feature_cols]
    y_val_raw = df_val["label"]

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_val = le.transform(y_val_raw)

    class_names = list(le.classes_)
    print(f"Classi codificate: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # --------------------------------------------------------
    # DEFINIZIONE GRIGLIE PARAMETRI (AGGIORNATE)
    # --------------------------------------------------------

    # A. Decision Tree (Va bene così, forse stringiamo un po' per precisione)
    dt_params = {
        "max_depth": [3, 4, 5, 6],  # Tolto 8 visto che ha scelto 4
        "min_samples_leaf": [1, 3, 5, 10],  # Aggiunto 3 per granularità
        "criterion": ["gini", "entropy"],
    }

    # B. Logistic Regression (Allarghiamo C verso l'alto)
    lr_params = {
        "C": [
            1,
            10,
            100,
            200,
            500,
        ],  # Tolti i valori piccolissimi (0.01), aggiunti alti
        "solver": ["lbfgs", "liblinear"],
    }

    # C. Random Forest (Diamo un po' più di libertà ma non troppa)
    rf_params = {
        "n_estimators": [100, 200, 300],  # Aumentiamo leggermente
        "max_depth": [2, 4, 6, 8, 10],  # Spingiamo fino a 10 (prima si fermava a 6)
        "min_samples_leaf": [2, 4, 8, 10],
        "max_features": ["sqrt", "log2"],
    }

    # D. SVM (Allarghiamo C)
    svm_params = {
        "C": [1, 10, 100, 200],  # Tolto 0.1
        "gamma": ["scale", "auto", 0.1, 0.01],
        "kernel": ["rbf", "linear"],  # RBF vince quasi sempre sui dati non lineari
    }

    # 4. Esecuzione
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
        # Tuning
        best_model = run_grid_search(model, params, X_train, y_train, name)
        best_estimators[name] = best_model

        # LOGGING PESI (NUOVO)
        log_feature_importance(best_model, feature_cols, name)

        # Valutazione
        score, kappa = evaluate_model(best_model, X_val, y_val, name, class_names)
        results[name] = (score, kappa)

    # 5. Selezione Vincitore
    print("\n" + "=" * 60)
    print("CONFRONTO FINALE (Basato su Validation Set)")
    for name, (score, kappa) in results.items():
        print(f"{name:<25}: F1-Score: {score:.4f}, Kappa: {kappa:.4f}")

    best_name = max(results, key=lambda x: results[x][0])
    best_score, best_kappa = results[best_name]
    best_model_final = best_estimators[best_name]

    print("-" * 60)
    print(
        f"🏆 VINCITORE: {best_name} (F1 Val: {best_score:.4f}, Kappa: {best_kappa:.4f})"
    )
    print("=" * 60)

    # Salvataggio
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
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
