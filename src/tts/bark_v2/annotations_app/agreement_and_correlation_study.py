import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score, confusion_matrix, mean_absolute_error

# --- CONFIGURAZIONE ---
FILE_DANIELE = "src/tts/bark_v2/annotations_app/daniele_human_annotations.csv"
FILE_LUCA = "src/tts/bark_v2/annotations_app/luca_human_annotations.csv"

# Impostiamo stile grafico
sns.set_theme(style="whitegrid")

# --- FUNZIONI DI CALCOLO ---


def get_metrics_bundle(y1, y2, prefix=""):
    """Calcola pacchetto completo di metriche."""
    # Gestione liste vuote per evitare crash
    if len(y1) == 0 or len(y2) == 0:
        return {f"{prefix}Status": "Insufficient Data"}

    kappa_unweighted = cohen_kappa_score(y1, y2, weights=None)
    kappa_weighted = cohen_kappa_score(y1, y2, weights="linear")
    pearson = y1.corr(y2, method="pearson")
    spearman = y1.corr(y2, method="spearman")

    return {
        f"{prefix}Kappa (Unweighted)": f"{kappa_unweighted:.3f}",
        f"{prefix}Kappa (Weighted Linear)": f"{kappa_weighted:.3f}",
        f"{prefix}Pearson Corr": f"{pearson:.3f}",
        f"{prefix}Spearman Corr": f"{spearman:.3f}",
    }


def print_metrics_section(title, metrics_dict):
    print(f"\n{'='*70}")
    print(f" {title}")
    print(f"{'='*70}")
    for k, v in metrics_dict.items():
        print(f"{k:<45}: {v}")


# --- 1. CARICAMENTO DATI ---
df1 = pd.read_csv(FILE_DANIELE)
df2 = pd.read_csv(FILE_LUCA)
df1["Annotator"] = "Daniele"
df2["Annotator"] = "Luca"

# --- 2. FILTRAGGIO INTELLIGENTE (Solo Audio Bad) ---

# Identifichiamo i video corrotti (segnalati da ALMENO UNO dei due)
bad_videos_daniele = df1[df1["is_audio_bad"] == True]["video_name"].unique()
bad_videos_luca = df2[df2["is_audio_bad"] == True]["video_name"].unique()
all_bad_videos = set(bad_videos_daniele).union(set(bad_videos_luca))

print(f"Dataset Totale Iniziale: Daniele={len(df1)}, Luca={len(df2)}")
print(f"Video con audio corrotto identificati: {len(all_bad_videos)}")

# FUNZIONE DI PULIZIA:
# Rimuoviamo la riga SOLO SE:
# 1. Il video_name è nella lista dei cattivi
# 2. E la modalità è AUDIO_ONLY
# (Lasciamo intatte le righe TEXT_ONLY anche se l'audio è rotto)


def clean_dataset(df, bad_list):
    # Maschera: True se da cancellare
    to_drop = (df["video_name"].isin(bad_list)) & (
        df["presentation_mode"] == "AUDIO_ONLY"
    )
    return df[~to_drop].copy()  # Ritorniamo l'inverso (quelli da tenere)


df1 = clean_dataset(df1, all_bad_videos)
df2 = clean_dataset(df2, all_bad_videos)

print(f"Dataset Filtrato (Text preservato): Daniele={len(df1)}, Luca={len(df2)}")


# --- 3. CALCOLO METRICHE ---

# A) Coerenza Interna (Audio vs Testo)
intra_results = {}
for name, df in [("Daniele", df1), ("Luca", df2)]:
    audio = df[df["presentation_mode"] == "AUDIO_ONLY"].set_index("video_name")[
        "human_rating"
    ]
    text = df[df["presentation_mode"] == "TEXT_ONLY"].set_index("video_name")[
        "human_rating"
    ]

    # Intersezione (Nota: qui i video bad audio verranno esclusi automaticamente perché mancano nel df Audio)
    common = audio.index.intersection(text.index)

    if len(common) > 0:
        metrics = get_metrics_bundle(audio[common], text[common])
        intra_results[f"--- {name} (n={len(common)}) ---"] = ""
        intra_results.update(metrics)
    else:
        intra_results[f"--- {name} ---"] = "N/A (No overlap)"

print_metrics_section("COERENZA INTERNA (Audio vs Testo)", intra_results)


# B) Agreement Inter-Annotatore (Daniele vs Luca)
inter_results = {}
merged = pd.merge(
    df1, df2, on=["video_name", "presentation_mode"], suffixes=("_dan", "_luca")
)

# Globale
inter_results["--- GLOBALE (Audio + Testo) ---"] = ""
inter_results.update(
    get_metrics_bundle(merged["human_rating_dan"], merged["human_rating_luca"])
)

# Per Modalità
for mode in ["AUDIO_ONLY", "TEXT_ONLY"]:
    subset = merged[merged["presentation_mode"] == mode]
    inter_results[f"--- {mode} (n={len(subset)}) ---"] = ""
    inter_results.update(
        get_metrics_bundle(subset["human_rating_dan"], subset["human_rating_luca"])
    )

print_metrics_section("AGREEMENT INTER-ANNOTATORE (Daniele vs Luca)", inter_results)


# C) Validità vs Ground Truth (Solo TEXT_ONLY)
validity_results = {}
for name, df in [("Daniele", df1), ("Luca", df2)]:
    df_text = df[df["presentation_mode"] == "TEXT_ONLY"]

    validity_results[f"--- {name} (Text Only, n={len(df_text)}) ---"] = ""
    validity_results.update(
        get_metrics_bundle(df_text["human_rating"], df_text["original_sentiment"])
    )

    mae = mean_absolute_error(df_text["original_sentiment"], df_text["human_rating"])
    validity_results["MAE (Mean Absolute Error)"] = f"{mae:.3f} punti"

print_metrics_section("VALIDITÀ VS GOLD STANDARD (Ground Truth)", validity_results)


# --- 4. GENERAZIONE GRAFICI ---

# Dataset unico per grafici (Text Only)
df_plot = pd.concat(
    [
        df1[df1["presentation_mode"] == "TEXT_ONLY"],
        df2[df2["presentation_mode"] == "TEXT_ONLY"],
    ]
)

# --- GRAFICO 1: BIAS TREND ---
plt.figure(figsize=(10, 6))

agg_data = (
    df_plot.groupby(["Annotator", "original_sentiment"])["human_rating"]
    .agg(["mean", "count", "std"])
    .reset_index()
)
agg_data["ci"] = 1.96 * agg_data["std"] / np.sqrt(agg_data["count"])

styles = {
    "Daniele": {"color": "#4477AA", "marker": "o", "shift": -0.15},
    "Luca": {"color": "#CC6677", "marker": "s", "shift": +0.15},
}

for annotator in ["Daniele", "Luca"]:
    # FIX: Usiamo .copy() per evitare il SettingWithCopyWarning
    subset = agg_data[agg_data["Annotator"] == annotator].copy()
    style = styles[annotator]

    # Riempie i NaN con 0 (per i punti singoli dove std non è calcolabile)
    subset["ci"] = subset["ci"].fillna(0)

    plt.errorbar(
        x=subset["original_sentiment"] + style["shift"],
        y=subset["mean"],
        yerr=subset["ci"],
        label=annotator,
        fmt=f"-{style['marker']}",
        color=style["color"],
        linewidth=2.5,
        markersize=8,
        capsize=5,
        alpha=0.9,
    )

plt.plot(
    [-3, 3], [-3, 3], color="gray", linestyle=":", label="Perfect Match (GT)", zorder=0
)
plt.title("Trend del Bias: Gli annotatori sottostimano gli estremi", fontsize=14)
plt.xlabel("Sentiment Reale (Original Ground Truth)", fontsize=12)
plt.ylabel("Voto Assegnato (Media)", fontsize=12)
plt.legend(title="Annotatore", loc="upper left")
plt.grid(True, linestyle="--", alpha=0.6)
plt.xticks(range(-3, 4))
plt.yticks(range(-3, 4))
plt.tight_layout()
plt.savefig("src/tts/bark_v2/annotations_app/results/1_bias_trend.png")
print("\n[Grafico salvato]: 1_bias_trend.png")


# --- GRAFICO 2: MATRICI DI CONFUSIONE ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
classes = range(-3, 4)

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


# --- GRAFICO 3: MAE Comparison ---
mae_data = []
for name in ["Daniele", "Luca"]:
    sub_df = df_plot[df_plot["Annotator"] == name]
    if len(sub_df) > 0:
        val = mean_absolute_error(sub_df["original_sentiment"], sub_df["human_rating"])
        mae_data.append({"Annotator": name, "MAE": val})

df_mae = pd.DataFrame(mae_data)

plt.figure(figsize=(7, 6))
colors = ["#5DADE2", "#F5B041"]
bars = plt.bar(
    df_mae["Annotator"],
    df_mae["MAE"],
    color=colors,
    width=0.5,
    edgecolor="black",
    alpha=0.8,
)

for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 0.02,
        f"{yval:.3f}",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

plt.title("Errore Medio Assoluto (MAE)\n(Più basso è meglio)", fontsize=14)
plt.ylabel("Errore Medio (Punti)", fontsize=12)
if not df_mae.empty:
    plt.ylim(0, df_mae["MAE"].max() + 0.5)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("src/tts/bark_v2/annotations_app/results/3_mae_comparison.png")
print("[Grafico salvato]: 3_mae_comparison.png")

print("\nAnalisi Completata.")
