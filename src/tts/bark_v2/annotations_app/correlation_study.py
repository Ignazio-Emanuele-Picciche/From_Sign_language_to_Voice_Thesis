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


# --- 3. GENERAZIONE GRAFICI AGGIORNATA ---

# Prepariamo dataset unico per i grafici (Solo Text Only per confronto con GT)
df_plot = pd.concat(
    [
        df1[df1["presentation_mode"] == "TEXT_ONLY"],
        df2[df2["presentation_mode"] == "TEXT_ONLY"],
    ]
)

# --- GRAFICO 1: BIAS TREND (Metodo "Manuale" per allineamento perfetto) ---
# Questo metodo calcola prima le medie e poi usa plt.errorbar.
# Risolve definitivamente il problema dell'asse sfalsato e della sovrapposizione.

plt.figure(figsize=(10, 6))

# 1. Aggreghiamo i dati per calcolare Media e Intervallo di Confidenza (CI)
# Calcoliamo media e errore standard (sem) per ogni punto
agg_data = (
    df_plot.groupby(["Annotator", "original_sentiment"])["human_rating"]
    .agg(["mean", "count", "std"])
    .reset_index()
)
# Calcoliamo l'intervallo di confidenza al 95% (1.96 * std / sqrt(n))
agg_data["ci"] = 1.96 * agg_data["std"] / np.sqrt(agg_data["count"])

# 2. Impostiamo i colori e gli stili
styles = {
    "Daniele": {
        "color": "#4477AA",
        "marker": "o",
        "shift": -0.15,
    },  # Daniele spostato a sx
    "Luca": {"color": "#CC6677", "marker": "s", "shift": +0.15},  # Luca spostato a dx
}

# 3. Disegniamo manualmente le linee con errorbar
for annotator in ["Daniele", "Luca"]:
    subset = agg_data[agg_data["Annotator"] == annotator]
    style = styles[annotator]

    # Applichiamo lo shift (jitter) all'asse X per evitare sovrapposizioni
    x_shifted = subset["original_sentiment"] + style["shift"]

    plt.errorbar(
        x=x_shifted,
        y=subset["mean"],
        yerr=subset["ci"],
        label=annotator,
        fmt=f"-{style['marker']}",  # Linea + Marker
        color=style["color"],
        linewidth=2.5,
        markersize=8,
        capsize=5,  # Cappuccetti alle barre di errore
        alpha=0.9,
    )

# 4. Linea diagonale perfetta (Ground Truth)
# Ora che l'asse è numerico, questa linea sarà perfettamente allineata
plt.plot(
    [-3, 3], [-3, 3], color="gray", linestyle=":", label="Perfect Match (GT)", zorder=0
)

# 5. Rifiniture grafiche
plt.title("Trend del Bias: Gli annotatori sottostimano gli estremi", fontsize=14)
plt.xlabel("Sentiment Reale (Original Ground Truth)", fontsize=12)
plt.ylabel("Voto Assegnato (Media)", fontsize=12)
plt.legend(title="Annotatore", loc="upper left")
plt.grid(True, linestyle="--", alpha=0.6)

# Forziamo gli assi a mostrare tutti i numeri interi da -3 a 3
plt.xticks(range(-3, 4))
plt.yticks(range(-3, 4))

plt.tight_layout()
plt.savefig("src/tts/bark_v2/annotations_app/results/1_bias_trend.png")
print("\n[Grafico salvato]: 1_bias_trend.png (Versione corretta)")


# --- GRAFICO 2: MATRICI DI CONFUSIONE (Invariato) ---
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


# --- GRAFICO 3: CONFRONTO ERRORE MEDIO (MAE) - NUOVO ---
# Calcoliamo il MAE dinamicamente dai dati
mae_data = []
for name in ["Daniele", "Luca"]:
    sub_df = df_plot[df_plot["Annotator"] == name]
    val = mean_absolute_error(sub_df["original_sentiment"], sub_df["human_rating"])
    mae_data.append({"Annotator": name, "MAE": val})

df_mae = pd.DataFrame(mae_data)

plt.figure(figsize=(7, 6))
colors = ["#5DADE2", "#F5B041"]  # Blu e Arancione

bars = plt.bar(
    df_mae["Annotator"],
    df_mae["MAE"],
    color=colors,
    width=0.5,
    edgecolor="black",
    alpha=0.8,
)

# Aggiungiamo il numero sopra la barra
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

plt.title("Errore Medio Assoluto (MAE)", fontsize=14)
plt.ylabel("Errore Medio (Punti)", fontsize=12)
plt.ylim(0, df_mae["MAE"].max() + 0.5)  # Scala dinamica
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig("src/tts/bark_v2/annotations_app/results/3_mae_comparison.png")
print("[Grafico salvato]: 3_mae_comparison.png")

print("\nAnalisi Completata.")
