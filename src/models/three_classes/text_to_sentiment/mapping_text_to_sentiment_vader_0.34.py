import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# --- CONFIGURAZIONE ---
THRESHOLD = 0.34

# Inizializziamo l'analizzatore VADER
analyzer = SentimentIntensityAnalyzer()


def get_vader_sentiment(text):
    """
    Calcola il sentiment usando VADER.
    Restituisce POSITIVE, NEGATIVE o NEUTRAL basandosi sulla soglia.
    """
    if not isinstance(text, str):
        text = str(text)  # Gestione di eventuali valori non stringa (NaN o numeri)

    scores = analyzer.polarity_scores(text)
    compound = scores["compound"]

    if compound >= THRESHOLD:
        return "POSITIVE"
    elif compound <= -THRESHOLD:  # Nota il meno: -0.34
        return "NEGATIVE"
    else:
        return "NEUTRAL"


def process_utterances():
    filename = "data/processed/utterances_with_translations.csv"
    if not os.path.exists(filename):
        print(f"⚠️  File {filename} non trovato.")
        return

    print(f"Elaborazione di {filename}...")

    # Caricamento CSV standard
    df = pd.read_csv(filename)

    # Verifica che la colonna esista
    if "caption" not in df.columns:
        print(f"Errore: Colonna 'caption' non trovata in {filename}")
        return

    # Applichiamo VADER
    df["sentiment"] = df["caption"].apply(get_vader_sentiment)

    # Rinominiamo le colonne per l'output richiesto
    # Assumiamo che ci sia una colonna ID, se non c'è usiamo l'indice
    if "video_name" not in df.columns:
        # Se il file ha un'altra colonna ID (es. 'id'), cambiala qui.
        # Altrimenti creiamo un ID fittizio basato sull'indice.
        df["video_name"] = df.index

    df_output = df.rename(columns={"TRANSLATION": "caption"})

    # Selezioniamo solo le colonne richieste
    final_df = df_output[["video_name", "caption", "sentiment"]]

    # Salvataggio
    output_name = "data/processed/text_to_sentiment/asllrp_sentiment_analyzed.csv"
    final_df.to_csv(output_name, index=False)
    print(f"✅ Salvato: {output_name}")


def process_how2sign():
    filename = "data/raw/test/how2sign_test.csv"
    if not os.path.exists(filename):
        print(f"⚠️  File {filename} non trovato.")
        return

    print(f"Elaborazione di {filename}...")

    # Caricamento CSV con separatore TAB
    df = pd.read_csv(filename, sep="\t")

    # Verifica colonne
    if "SENTENCE" not in df.columns:
        print(f"Errore: Colonna 'SENTENCE' non trovata in {filename}")
        return

    # Applichiamo VADER
    df["sentiment"] = df["SENTENCE"].apply(get_vader_sentiment)

    # Gestione colonna ID
    # Solitamente How2Sign ha 'SENTENCE_NAME' o 'VIDEO_ID'
    # Cerchiamo di normalizzare il nome per l'output
    if "SENTENCE_NAME" in df.columns:
        df = df.rename(columns={"SENTENCE_NAME": "sentence_name"})
    elif "sentence_name" not in df.columns:
        df["sentence_name"] = df.index  # Fallback se manca l'ID

    # Rinominiamo SENTENCE in sentence (minuscolo) per l'output se necessario
    df = df.rename(columns={"SENTENCE": "sentence"})

    # Selezioniamo solo le colonne richieste
    final_df = df[["sentence_name", "sentence", "sentiment"]]

    # Salvataggio
    output_name = "data/processed/text_to_sentiment/how2sign_sentiment_analyzed.csv"
    final_df.to_csv(output_name, index=False)
    print(f"✅ Salvato: {output_name}")


# --- ESECUZIONE ---
if __name__ == "__main__":
    process_utterances()
    print("-" * 30)
    process_how2sign()
