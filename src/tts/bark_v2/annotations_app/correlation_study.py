import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_absolute_error

# --- CONFIGURAZIONE ---
# Sostituisci con i tuoi file reali
FILE_DANIELE = "src/tts/bark_v2/annotations_app/daniele_human_annotations.csv"
FILE_LUCA = "src/tts/bark_v2/annotations_app/luca_human_annotations.csv"

# Impostiamo stile grafico
sns.set_theme(style="whitegrid")

# --- FUNZIONI DI CALCOLO ---


def get_weighted_kappa(y1, y2):
    """Calcola il Kappa di Cohen con pesi lineari (ideale per scale ordinali -3 a +3)"""
    return cohen_kappa_score(y1, y2, weights="linear")


def print_metrics_section(title, metrics):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"{k:<40}: {v}")


# --- 1. CARICAMENTO DATI ---
df1 = pd.read_csv(FILE_DANIELE)
df2 = pd.read_csv(FILE_LUCA)
df1["Annotator"] = "Daniele"
df2["Annotator"] = "Luca"

# --- 2. CALCOLO INDICI DI AGREEMENT (NUMERICI) ---

# A) Coerenza Interna (Audio vs Testo) per Annotatore
intra_results = {}
for name, df in [("Daniele", df1), ("Luca", df2)]:
    audio = df[df["presentation_mode"] == "AUDIO_ONLY"].set_index("video_name")[
        "human_rating"
    ]
    text = df[df["presentation_mode"] == "TEXT_ONLY"].set_index("video_name")[
        "human_rating"
    ]

    # Allineiamo i dati
    common = audio.index.intersection(text.index)
    if len(common) > 0:
        k = get_weighted_kappa(audio[common], text[common])
        intra_results[f"Kappa Audio-Text ({name})"] = f"{k:.3f}"
    else:
        intra_results[f"Kappa Audio-Text ({name})"] = "N/A (Nessun video comune)"

print_metrics_section("COERENZA INTERNA (Intra-rater)", intra_results)


# B) Agreement tra Annotatori (Daniele vs Luca)
inter_results = {}
# Merge su video e modalità
merged = pd.merge(
    df1, df2, on=["video_name", "presentation_mode"], suffixes=("_dan", "_luca")
)

# Kappa Globale
k_global = get_weighted_kappa(merged["human_rating_dan"], merged["human_rating_luca"])
inter_results["Weighted Kappa (Globale)"] = f"{k_global:.3f}"

# Kappa per Modalità
for mode in ["AUDIO_ONLY", "TEXT_ONLY"]:
    subset = merged[merged["presentation_mode"] == mode]
    k_mode = get_weighted_kappa(subset["human_rating_dan"], subset["human_rating_luca"])
    inter_results[f"Weighted Kappa ({mode})"] = f"{k_mode:.3f}"

print_metrics_section("AGREEMENT TRA ANNOTATORI (Inter-rater)", inter_results)


# C) Accuratezza vs Ground Truth (Solo TEXT_ONLY)
validity_results = {}
for name, df in [("Daniele", df1), ("Luca", df2)]:
    # Filtro Text Only
    df_text = df[df["presentation_mode"] == "TEXT_ONLY"]

    # Metriche
    k_gt = get_weighted_kappa(df_text["human_rating"], df_text["original_sentiment"])
    mae = mean_absolute_error(df_text["original_sentiment"], df_text["human_rating"])
    corr = df_text["human_rating"].corr(df_text["original_sentiment"])

    validity_results[f"--- {name} ---"] = ""
    validity_results[f"Kappa vs GT ({name})"] = f"{k_gt:.3f}"
    validity_results[f"Pearson Corr ({name})"] = f"{corr:.3f}"
    validity_results[f"MAE Error ({name})"] = f"{mae:.3f} punti"

print_metrics_section("VALIDITÀ VS GOLD STANDARD (Text Only)", validity_results)


# --- 3. GENERAZIONE GRAFICI ---

# Prepariamo dataset unico per i grafici (Solo Text Only per confronto con GT)
df_plot = pd.concat(
    [
        df1[df1["presentation_mode"] == "TEXT_ONLY"],
        df2[df2["presentation_mode"] == "TEXT_ONLY"],
    ]
)

# GRAFICO 1: Bias Trend (Lineplot)
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=df_plot,
    x="original_sentiment",
    y="human_rating",
    hue="Annotator",
    style="Annotator",
    markers=True,
    dashes=False,
    err_style="bars",
    linewidth=2.5,
    markersize=9,
)
plt.plot([-3, 3], [-3, 3], color="gray", linestyle="--", label="Perfect Match")
plt.title("Trend del Bias: Gli annotatori seguono gli estremi?", fontsize=14)
plt.xlabel("Original Sentiment (Ground Truth)")
plt.ylabel("Voto Medio Assegnato")
plt.legend()
plt.tight_layout()
plt.savefig("src/tts/bark_v2/annotations_app/results/1_bias_trend.png")
print("\n[Grafico salvato]: 1_bias_trend.png")

# GRAFICO 2: Matrici di Confusione (Heatmap)
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
classes = range(-3, 4)  # Da -3 a +3

for i, annotator in enumerate(["Daniele", "Luca"]):
    data = df_plot[df_plot["Annotator"] == annotator]
    cm = confusion_matrix(
        data["original_sentiment"], data["human_rating"], labels=classes
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=classes,
        yticklabels=classes,
        ax=axes[i],
    )
    axes[i].set_title(f"{annotator}", fontsize=14)
    axes[i].set_xlabel("Voto Assegnato")
    if i == 0:
        axes[i].set_ylabel("Voto Reale (GT)")

plt.suptitle("Matrici di Confusione (Text Only)", fontsize=16)
plt.tight_layout()
plt.savefig("src/tts/bark_v2/annotations_app/results/2_confusion_matrix.png")
print("[Grafico salvato]: 2_confusion_matrix.png")

# GRAFICO 3: Distribuzione Errori (Boxplot)
df_plot["error"] = df_plot["human_rating"] - df_plot["original_sentiment"]
plt.figure(figsize=(8, 6))
sns.boxplot(x="Annotator", y="error", data=df_plot, palette="Set2", showmeans=True)
plt.axhline(0, color="red", linestyle="--", alpha=0.7)
plt.title("Distribuzione degli Errori (Human - GT)", fontsize=14)
plt.ylabel("Errore (Punti)")
plt.text(
    0.5,
    -2.8,
    "Sotto 0 = Sottostima (Voto più basso del reale)\nSopra 0 = Sovrastima (Voto più alto del reale)",
    ha="center",
    fontsize=9,
    bbox=dict(facecolor="white", alpha=0.8),
)
plt.tight_layout()
plt.savefig("src/tts/bark_v2/annotations_app/results/3_error_distribution.png")
print("[Grafico salvato]: 3_error_distribution.png")

print("\nAnalisi Completata.")
