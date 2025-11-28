"""
================================================================================================================
TRAINING DEL MULTI-LEARNER (LATE FUSION / STACKING STRATEGY)
================================================================================================================

DESCRIZIONE:
Questo script implementa il livello finale dell'architettura Multimodale: il "Meta-Learner".
Utilizza una tecnica di Late Fusion (Stacking) per combinare le predizioni di due modelli unimodali
specializzati:
1.  Modello LSTM (Video): Addestrato su landmarks facciali/corporei (MediaPipe/OpenPose).
2.  Modello RoBERTa (Testo): Fine-tunato sulle caption/trascrizioni.

OBIETTIVO:
Costruire un classificatore "arbitro" che prenda in input i vettori di probabilità (Softmax)
dei due modelli base e impari a decidere l'emozione finale. Il meta-learner impara, ad esempio,
a fidarsi più del video quando il testo è ambiguo (o neutro) e viceversa.

INPUT DATI (Feature Engineering):
I dati di input non sono i dati grezzi (video o testo), ma le "confidence scores" estratte
dal Training Set tramite gli script di estrazione dedicati:
- Input X (6 Features):
    [P_neg_LSTM, P_neu_LSTM, P_pos_LSTM, P_neg_RoBERTa, P_neu_RoBERTa, P_pos_RoBERTa]
- Target y:
    Label reale (Emotion Ground Truth)

METODOLOGIA:
1.  Caricamento e Merge dei CSV di features (join su 'video_name').
2.  Split Train/Validation (80/20) interno per valutare il meta-learner.
3.  Addestramento e confronto di due approcci:
    - Logistic Regression: Lineare, interpretabile (i coefficienti indicano il peso delle modalità).
    - Decision Tree: Non lineare, capace di imparare regole condizionali complesse.
4.  Salvataggio del modello migliore per l'inferenza finale sul Test Set.

FILE GENERATI:
- models/metalearner/metalearner_[tipo].joblib: Il modello addestrato.
- models/metalearner/label_encoder.joblib: Encoder per le classi target.
================================================================================================================
"""

import pandas as pd
import numpy as np
import os
import sys
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --- SETUP ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, BASE_DIR)

# Percorsi ai file generati negli step precedenti
LSTM_FEATURES_PATH = "./lstm_features_train_set.csv"
ROBERTA_FEATURES_PATH = "./roberta_features_train_set.csv"
OUTPUT_MODEL_DIR = "models/metalearner"


def load_and_merge_data():
    """Carica i due CSV e li unisce usando video_name come chiave."""
    print("--- 1. Caricamento e Merge dei Dati ---")

    if not os.path.exists(LSTM_FEATURES_PATH) or not os.path.exists(
        ROBERTA_FEATURES_PATH
    ):
        raise FileNotFoundError(
            "❌ Mancano i file delle features! Esegui prima gli script di estrazione."
        )

    df_lstm = pd.read_csv(LSTM_FEATURES_PATH)
    df_roberta = pd.read_csv(ROBERTA_FEATURES_PATH)

    print(f"  LSTM shape: {df_lstm.shape}")
    print(f"  RoBERTa shape: {df_roberta.shape}")

    # Merge Inner: Teniamo solo i video presenti in entrambi i dataset
    # Suffissi per evitare conflitti sulla colonna true_label
    merged_df = pd.merge(
        df_lstm, df_roberta, on="video_name", suffixes=("_lstm", "_roberta")
    )

    # Verifica coerenza etichette (Opzionale ma consigliato)
    # Se le etichette sono stringhe, normalizziamole
    if "true_label_lstm" in merged_df.columns:
        merged_df["label"] = merged_df["true_label_lstm"]
    elif "true_label" in merged_df.columns:
        merged_df["label"] = merged_df["true_label"]

    print(f"✅ Merge completato. Shape finale: {merged_df.shape}")
    return merged_df


def train_decision_tree(X_train, X_test, y_train, y_test, feature_names):
    print("\n--- Modello A: Decision Tree ---")
    # Max depth limitata per evitare overfitting (visto che abbiamo poche feature)
    clf = DecisionTreeClassifier(max_depth=4, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"  Accuracy: {acc:.4f}")
    print("  Report:")
    print(classification_report(y_test, y_pred))

    # Feature Importance
    print("  Importanza Feature:")
    for name, importance in zip(feature_names, clf.feature_importances_):
        print(f"    {name}: {importance:.4f}")

    return clf, acc


def train_logistic_regression(X_train, X_test, y_train, y_test, feature_names):
    print("\n--- Modello B: Logistic Regression ---")
    clf = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"  Accuracy: {acc:.4f}")
    print("  Report:")
    print(classification_report(y_test, y_pred))

    # Analisi dei coefficienti (chi comanda?)
    print("  Coefficienti (Pesi):")
    # I coefficienti sono (n_classes, n_features). Facciamo la media assoluta per capire l'importanza globale
    avg_coefs = np.mean(np.abs(clf.coef_), axis=0)
    for name, coef in zip(feature_names, avg_coefs):
        print(f"    {name}: {coef:.4f}")

    return clf, acc


def main():
    # 1. Preparazione Dati
    df = load_and_merge_data()

    # Definiamo le feature (X) e il target (y)
    # Cerchiamo le colonne che iniziano con 'lstm_prob' o 'roberta_prob'
    feature_cols = [c for c in df.columns if "lstm_prob" in c or "roberta_prob" in c]
    print(f"\nFeature usate ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols]
    y = df[
        "label"
    ]  # Assumiamo che la colonna label sia stringa (NEGATIVE, NEUTRAL, POSITIVE)

    # Encoding delle label (String -> Int)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Classi trovate: {le.classes_}")

    # Split Train/Validation per il Metalearner
    # IMPORTANTE: Questo è un 'internal validation'.
    # Non stiamo ancora toccando il Golden Test Set reale.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 2. Addestramento e Confronto
    tree_model, tree_acc = train_decision_tree(
        X_train, X_val, y_train, y_val, feature_cols
    )
    log_model, log_acc = train_logistic_regression(
        X_train, X_val, y_train, y_val, feature_cols
    )

    # 3. Selezione e Salvataggio Vincitore
    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)

    best_model = None
    best_name = ""

    print("\n--- Risultato Finale ---")
    if log_acc >= tree_acc:
        print(f"🏆 Vince Logistic Regression ({log_acc:.4f} vs {tree_acc:.4f})")
        best_model = log_model
        best_name = "logistic_regression"
    else:
        print(f"🏆 Vince Decision Tree ({tree_acc:.4f} vs {log_acc:.4f})")
        best_model = tree_model
        best_name = "decision_tree"

    # Salvataggio con Joblib
    model_path = os.path.join(OUTPUT_MODEL_DIR, f"metalearner_{best_name}.joblib")
    joblib.dump(best_model, model_path)
    # Salviamo anche il LabelEncoder per decodificare le predizioni dopo
    joblib.dump(le, os.path.join(OUTPUT_MODEL_DIR, "label_encoder.joblib"))

    print(f"✅ Modello salvato in: {model_path}")
    print(
        f"ℹ️  Nota: Per usarlo sul test set, dovrai estrarre le features dal test set e caricarle qui."
    )


if __name__ == "__main__":
    main()
