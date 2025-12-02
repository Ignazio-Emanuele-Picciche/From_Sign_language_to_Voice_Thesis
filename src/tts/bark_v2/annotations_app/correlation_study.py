import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

# 1. CARICAMENTO DATI
# Sostituisci con i nomi reali dei tuoi file
file1 = "src/tts/bark_v2/annotations_app/daniele_human_annotations.csv"
file2 = "src/tts/bark_v2/annotations_app/luca_human_annotations.csv"  # Metti qui il nome del secondo file

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)


# Funzione helper per calcolare il Kappa Pesato
def get_weighted_kappa(y1, y2):
    # weights='linear' è ideale per scale ordinali come la tua (-3 a 3)
    # weights='quadratic' penalizzerebbe ancora di più gli errori grandi
    return cohen_kappa_score(y1, y2, weights="linear")


# --- ANALISI 1: COERENZA INTERNA (AUDIO vs TESTO) ---
def analyze_intra_coherence(df, annotator_name):
    print(f"\n--- ANALISI COERENZA AUDIO vs TESTO: {annotator_name} ---")

    # Separiamo e allineiamo Audio e Testo per lo stesso video
    df_audio = df[df["presentation_mode"] == "AUDIO_ONLY"][
        ["video_name", "human_rating"]
    ]
    df_text = df[df["presentation_mode"] == "TEXT_ONLY"][["video_name", "human_rating"]]

    # Uniamo su video_name
    merged = pd.merge(df_audio, df_text, on="video_name", suffixes=("_audio", "_text"))

    if len(merged) == 0:
        print("Errore: Non ci sono video corrispondenti con entrambe le modalità.")
        return

    kappa = get_weighted_kappa(
        merged["human_rating_audio"], merged["human_rating_text"]
    )

    print(f"Coppie analizzate: {len(merged)}")
    print(f"Weighted Cohen's Kappa: {kappa:.3f}")


# Eseguiamo per entrambi gli annotatori
analyze_intra_coherence(df1, "Daniele")
analyze_intra_coherence(df2, "Luca")


# --- ANALISI 2: AGREEMENT TRA ANNOTATORI (INTER-RATER) ---
print(f"\n--- ANALISI AGREEMENT TRA ANNOTATORI (Daniele vs Luca) ---")

# Uniamo i due annotatori assicurandoci di confrontare lo stesso video nello stesso modo
df_inter = pd.merge(
    df1, df2, on=["video_name", "presentation_mode"], suffixes=("_dan", "_luca")
)

# Calcolo globale
kappa_global = get_weighted_kappa(
    df_inter["human_rating_dan"], df_inter["human_rating_luca"]
)
print(f"Weighted Kappa Globale (Audio + Testo): {kappa_global:.3f}")

# Calcolo diviso per modalità (Opzionale, ma utile)
df_inter_audio = df_inter[df_inter["presentation_mode"] == "AUDIO_ONLY"]
df_inter_text = df_inter[df_inter["presentation_mode"] == "TEXT_ONLY"]

k_audio = get_weighted_kappa(
    df_inter_audio["human_rating_dan"], df_inter_audio["human_rating_luca"]
)
k_text = get_weighted_kappa(
    df_inter_text["human_rating_dan"], df_inter_text["human_rating_luca"]
)

print(f" -> Kappa solo Audio: {k_audio:.3f}")
print(f" -> Kappa solo Testo: {k_text:.3f}")


# --- NUOVA FUNZIONE ANALISI 3: VALIDITÀ VS ORIGINAL SENTIMENT ---
def analyze_vs_ground_truth(df, annotator_name):
    print(f"\n--- VALIDITÀ VS ORIGINAL SENTIMENT (Text Only): {annotator_name} ---")

    # Filtriamo solo Text Only
    df_text = df[df["presentation_mode"] == "TEXT_ONLY"]

    # 1. Correlazione di Pearson (Linearità)
    # Utile per capire se al variare di uno varia anche l'altro
    pearson = df_text["human_rating"].corr(
        df_text["original_sentiment"], method="pearson"
    )

    # 2. Correlazione di Spearman (Rango)
    # Più robusta per dati ordinali come i tuoi (-3 a +3), valuta se l'ordinamento è rispettato
    spearman = df_text["human_rating"].corr(
        df_text["original_sentiment"], method="spearman"
    )

    # 3. Mean Absolute Error (MAE)
    # Ti dice l'errore medio in "punti". Es: 0.5 significa che sbagliano in media di mezzo voto.
    mae = mean_absolute_error(df_text["original_sentiment"], df_text["human_rating"])

    print(f"Correlazione Pearson (Trend Lineare): {pearson:.3f}")
    print(f"Correlazione Spearman (Trend Rango):  {spearman:.3f}")
    print(f"Errore Medio Assoluto (MAE):          {mae:.3f} punti")

    # Interpretazione al volo
    if pearson > 0.7:
        print("-> Ottima validità: l'annotatore segue fedelmente il Gold Standard.")
    elif pearson > 0.4:
        print("-> Validità moderata: c'è una relazione ma con del rumore.")
    else:
        print(
            "-> Bassa validità: l'annotatore sembra valutare cose diverse dal Gold Standard."
        )


analyze_vs_ground_truth(df1, "Daniele")
analyze_vs_ground_truth(df2, "Luca")
