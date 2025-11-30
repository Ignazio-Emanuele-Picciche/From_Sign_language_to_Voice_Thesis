import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
)


def calcola_metriche_excel(file_path):
    try:
        # 1. Caricamento del file Excel
        # Assicurati che il file si trovi nella stessa cartella o fornisci il percorso completo
        df = pd.read_excel(file_path)

        # Stampa le prime righe per debug (opzionale)
        print(f"File caricato con successo. Righe totali: {len(df)}")

        # 2. Definizione dei nomi delle colonne
        # Modifica queste stringhe se i nomi nel tuo Excel sono leggermente diversi (es. spazi extra)
        col_pred = "emotion PREDICTION"
        col_gt = "emotion GT"

        # Verifica che le colonne esistano
        if col_pred not in df.columns or col_gt not in df.columns:
            print(
                f"ERRORE: Colonne non trovate. Colonne disponibili: {df.columns.tolist()}"
            )
            return

        # 3. Pulizia dei dati
        # Rimuoviamo righe che potrebbero avere valori nulli nelle colonne di interesse
        df_clean = df.dropna(subset=[col_pred, col_gt])

        # Convertiamo tutto in stringa per evitare errori se ci sono numeri misti a testo
        y_pred = df_clean[col_pred].astype(str).str.upper()
        y_true = df_clean[col_gt].astype(str).str.upper()

        # 4. Calcolo delle Metriche

        # Accuracy Standard
        acc = accuracy_score(y_true, y_pred)

        # Weighted F1 Score
        # average='weighted': calcola le metriche per ogni etichetta e trova la media
        # ponderata in base al supporto (il numero di istanze vere per ogni etichetta).
        f1_w = f1_score(y_true, y_pred, average="weighted")

        # [OPZIONALE] Balanced Accuracy
        # Utile se le classi sono sbilanciate (es. molti "neutral" e pochi "angry")
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        # Cohen's Kappa Score
        kappa = cohen_kappa_score(y_true, y_pred)

        # 5. Output dei risultati
        print("-" * 30)
        print("RISULTATI DEL CALCOLO")
        print("-" * 30)
        print(f"Accuracy (Standard):  {acc:.4f} ({acc*100:.2f}%)")
        print(f"Weighted F1 Score:    {f1_w:.4f} ({f1_w*100:.2f}%)")
        print(f"Balanced Accuracy:    {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
        print(f"Cohen's Kappa Score:  {kappa:.4f}")
        print("-" * 30)

        # Report dettagliato per ogni classe (opzionale)
        print("\nReport Dettagliato per Classe:")
        print(classification_report(y_true, y_pred))

    except FileNotFoundError:
        print("Errore: File non trovato. Controlla il percorso o il nome del file.")
    except Exception as e:
        print(f"Si è verificato un errore imprevisto: {e}")


# --- ESECUZIONE ---
# Sostituisci 'tuo_file.xlsx' con il nome reale del tuo file Excel
nome_file_excel = (
    "src/models/three_classes/text_to_sentiment/modello_zero_shot/gemini_3_0_pro.xlsx"
)
calcola_metriche_excel(nome_file_excel)
