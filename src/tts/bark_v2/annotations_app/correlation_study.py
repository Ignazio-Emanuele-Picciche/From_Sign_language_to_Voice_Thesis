import pandas as pd
import numpy as np


# --- FUNZIONE DI SUPPORTO PER CATEGORIE ---
def get_polarity(score):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    return "Neutral"


# 1. CARICAMENTO DATI
# Sostituisci con i nomi reali dei tuoi file
file1 = "src/tts/bark_v2/annotations_app/daniele_human_annotations.csv"
file2 = "src/tts/bark_v2/annotations_app/luca_human_annotations.csv"  # Metti qui il nome del secondo file

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)


# --- ANALISI 1: AUDIO VS TESTO (Per singolo annotatore - Es. Daniele) ---
def analyze_audio_text_coherence(df, annotator_name):
    print(f"\n--- ANALISI COERENZA AUDIO-TESTO: {annotator_name} ---")

    # Separiamo Audio e Testo
    df_audio = df[df["presentation_mode"] == "AUDIO_ONLY"][
        ["video_name", "human_rating", "original_sentiment"]
    ].copy()
    df_text = df[df["presentation_mode"] == "TEXT_ONLY"][
        ["video_name", "human_rating", "original_sentiment"]
    ].copy()

    # Uniamo sulla base del video_name per confrontare lo stesso video
    merged = pd.merge(df_audio, df_text, on="video_name", suffixes=("_audio", "_text"))

    # 1. Correlazione Totale (Pearson)
    corr_total = merged["human_rating_audio"].corr(merged["human_rating_text"])
    print(f"Correlazione Totale (Pearson): {corr_total:.2f}")

    # 2. Analisi per Categorie (Basata sul 'original_sentiment' come riferimento)
    merged["category"] = merged["original_sentiment_audio"].apply(get_polarity)

    print("\nCoerenza % (Agreement) divisa per categoria originale:")
    for cat in ["Negative", "Neutral", "Positive"]:
        subset = merged[merged["category"] == cat]
        if len(subset) == 0:
            print(f"- {cat}: Nessun dato")
            continue

        # Calcoliamo la coerenza di "Segno" (se entrambi dicono Positivo, anche se voti diversi es: 2 e 3)
        # Oppure "Esatta": subset['human_rating_audio'] == subset['human_rating_text']
        # Qui facciamo coerenza di polarità che è più utile
        coherence = np.where(
            subset["human_rating_audio"].apply(get_polarity)
            == subset["human_rating_text"].apply(get_polarity),
            1,
            0,
        )
        acc_pct = coherence.mean() * 100
        print(f"- {cat} (n={len(subset)}): {acc_pct:.1f}% di accordo sulla polarità")


# Eseguiamo su Daniele
analyze_audio_text_coherence(df1, "Daniele")


# --- ANALISI 2: CONFRONTO TRA DUE ANNOTATORI ---
print(f"\n--- ANALISI INTER-ANNOTATORE (Daniele vs Altro) ---")
# Uniamo i due dataframe completi su video_name e presentation_mode
df_inter = pd.merge(
    df1, df2, on=["video_name", "presentation_mode"], suffixes=("_dan", "_ann2")
)

# Correlazione Totale
corr_inter = df_inter["human_rating_dan"].corr(df_inter["human_rating_ann2"])
print(f"Correlazione tra annotatori (Globale): {corr_inter:.2f}")


# --- ANALISI 3: COERENZA CON ORIGINAL_SENTIMENT (Solo TEXT_ONLY) ---
def analyze_ground_truth_accuracy(df, annotator_name):
    print(f"\n--- ACCURATEZZA VS ORIGINAL SENTIMENT (Text Only): {annotator_name} ---")

    df_text = df[df["presentation_mode"] == "TEXT_ONLY"].copy()

    # Correlazione pura
    corr_gt = df_text["human_rating"].corr(df_text["original_sentiment"])
    print(f"Correlazione con Original Sentiment: {corr_gt:.2f}")

    # Percentuale di accordo sulla polarità
    agreement = np.where(
        df_text["human_rating"].apply(get_polarity)
        == df_text["original_sentiment"].apply(get_polarity),
        1,
        0,
    )
    print(f"Accuratezza Polarità (Coerenza %): {agreement.mean() * 100:.1f}%")


analyze_ground_truth_accuracy(df1, "Daniele")
analyze_ground_truth_accuracy(df2, "Secondo Annotatore")
