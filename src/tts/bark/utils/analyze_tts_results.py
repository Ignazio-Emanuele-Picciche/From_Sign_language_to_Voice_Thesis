"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        TTS ANALYZER - VALIDAZIONE QUANTITATIVA E QUALITATIVA                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 DESCRIZIONE:
    Strumento di analisi post-generazione progettato per validare l'output del
    modulo TTS. Fornisce metriche oggettive sulla qualità del processo e seleziona
    intelligente campioni per l'ascolto umano (Human Evaluation).

📊 METRICHE CALCOLATE:
    1. Copertura: Percentuale di video per cui è stato generato con successo un audio.
    2. Durata Media per Emozione: Verifica l'ipotesi che le emozioni negative
       (caratterizzate da pause e sospiri) producano audio mediamente più lunghi.
    3. Correlazione Confidenza-Durata: Indaga se una maggiore certezza del modello
       (che attiva più tag espressivi) corrisponde a una maggiore durata dell'audio.

🎧 CAMPIONAMENTO QUALITATIVO:
    Seleziona automaticamente i casi limite (Edge Cases) per l'ascolto manuale:
    - Top-2 High Confidence: Per verificare la massima espressività (risate/pianti).
    - Top-2 Low Confidence: Per verificare la gestione dell'incertezza (esitazioni).

📂 OUTPUT:
    - Report testuale a video.
    - File CSV dettagliato: reports/tts_generation_analysis.csv
"""

import os
import pandas as pd
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
AUDIO_DIR = os.path.join("src", "tts", "bark", "output_audio")
PREDICTIONS_FILE = os.path.join(
    "src", "tts", "bark", "final_predictions_with_captions.csv"
)


def get_audio_duration(file_path):
    try:
        sr, data = wavfile.read(file_path)
        return len(data) / sr
    except Exception:
        return 0.0


def analyze_results():
    print("=" * 60)
    print("ANALISI QUANTITATIVA GENERAZIONE TTS")
    print("=" * 60)

    # 1. Caricamento Dati
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"❌ File predizioni mancante: {PREDICTIONS_FILE}")
        return

    df = pd.read_csv(PREDICTIONS_FILE)
    total_videos = len(df)

    # 2. Scansione Audio Generati
    print(f"Scansione cartella audio: {AUDIO_DIR}...")
    if not os.path.exists(AUDIO_DIR):
        print("❌ Cartella audio non trovata.")
        return

    generated_files = os.listdir(AUDIO_DIR)
    # Mappa video_name -> file_path
    # Assumiamo formato: {video_name}_{emotion}.wav
    audio_map = {}
    for f in generated_files:
        if f.endswith(".wav"):
            # Ricostruisce video_name (un po' fragile se il nome contiene underscore, ma ci proviamo)
            # Strategia migliore: iterare sul DF e cercare il file
            pass

    # Aggiungi info audio al DataFrame
    durations = []
    file_exists = []

    for _, row in df.iterrows():
        v_name = str(row["video_name"]).replace("/", "_").replace("\\", "_")
        emotion = str(row["predicted_label"]).lower()
        if emotion not in ["positive", "negative", "neutral"]:
            emotion = "neutral"

        filename = f"{v_name}_{emotion}.wav"
        full_path = os.path.join(AUDIO_DIR, filename)

        if os.path.exists(full_path):
            file_exists.append(True)
            durations.append(get_audio_duration(full_path))
        else:
            file_exists.append(False)
            durations.append(0.0)

    df["audio_generated"] = file_exists
    df["duration"] = durations

    # 3. Statistiche Generali
    generated_count = df["audio_generated"].sum()
    print(f"\n📊 Copertura:")
    print(f"  Video Totali:      {total_videos}")
    print(
        f"  Audio Generati:    {generated_count} ({generated_count/total_videos*100:.1f}%)"
    )
    print(f"  Audio Mancanti:    {total_videos - generated_count}")

    if generated_count == 0:
        print("⚠️  Nessun audio trovato. Hai eseguito tts_generator.py?")
        return

    # 4. Analisi Durata per Emozione
    print(f"\n⏱️  Durata Media per Emozione (s):")
    stats = (
        df[df["audio_generated"]]
        .groupby("predicted_label")["duration"]
        .agg(["mean", "std", "count"])
    )
    print(stats)

    # Check: Negative dovrebbe essere più lungo (pause/sospiri)
    try:
        neg_mean = stats.loc["Negative", "mean"] if "Negative" in stats.index else 0
        pos_mean = stats.loc["Positive", "mean"] if "Positive" in stats.index else 0
        if neg_mean > pos_mean:
            print(
                "  ✅ Conferma Ipotesi: I video 'Negative' sono mediamente più lunghi (sospiri/pause)."
            )
    except:
        pass

    # 5. Correlazione Confidenza-Durata
    # Ipotesi: Alta confidenza = più tag (risate/sospiri) = durata maggiore?
    corr = df[df["audio_generated"]]["confidence"].corr(
        df[df["audio_generated"]]["duration"]
    )
    print(f"\n📉 Correlazione Confidenza/Durata: {corr:.4f}")

    # 6. SELEZIONE CAMPIONI PER ASCOLTO QUALITATIVO
    print("\n" + "=" * 60)
    print("CAMPIONI SELEZIONATI PER ANALISI QUALITATIVA (MANUALE)")
    print("Ascolta questi file per verificare l'espressività:")
    print("=" * 60)

    # Seleziona 2 High Confidence e 2 Low Confidence per ogni emozione
    emotions = ["Positive", "Negative", "Neutral"]

    for emo in emotions:
        subset = df[(df["predicted_label"] == emo) & (df["audio_generated"])]
        if subset.empty:
            continue

        print(f"\n--- {emo.upper()} ---")

        # High Confidence (Dovrebbe avere risate forti / sospiri)
        high_conf = subset.nlargest(2, "confidence")
        print(f"🔊 ALTA CONFIDENZA (Atteso: {emo} intenso, tag forti):")
        for _, row in high_conf.iterrows():
            fname = f"{str(row['video_name']).replace('/','_')}_{emo.lower()}.wav"
            print(f"  - File: {fname}")
            print(
                f"    Conf: {row['confidence']:.4f} | Caption: {str(row.get('caption', ''))[:50]}..."
            )

        # Low Confidence (Dovrebbe avere esitazioni / [clears throat])
        low_conf = subset.nsmallest(2, "confidence")
        print(f"🔊 BASSA CONFIDENZA (Atteso: Esitazioni, tono incerto):")
        for _, row in low_conf.iterrows():
            fname = f"{str(row['video_name']).replace('/','_')}_{emo.lower()}.wav"
            print(f"  - File: {fname}")
            print(f"    Conf: {row['confidence']:.4f}")

    # Salva report CSV
    report_path = os.path.join(BASE_DIR, "reports", "tts_generation_analysis.csv")
    df.to_csv(report_path, index=False)
    print(f"\n📄 Report completo salvato in: {report_path}")


if __name__ == "__main__":
    analyze_results()
