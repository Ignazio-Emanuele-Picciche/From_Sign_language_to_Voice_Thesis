"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      PROSODY ANALYZER - ANALISI SPETTRALE E DEL PITCH (F0)                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 DESCRIZIONE:
    Modulo di analisi del segnale audio (DSP) finalizzato alla dimostrazione
    scientifica dell'espressività emotiva.

    Confronta file audio generati con etichette emotive opposte (Positive vs Negative)
    estraendo feature acustiche oggettive per dimostrare che il modello TTS non
    sta solo leggendo il testo, ma sta modulando la voce.

🔬 METODOLOGIA:
    1. Estrazione F0 (Fondamentale): Utilizza l'algoritmo PYIN per tracciare
       il contorno dell'intonazione (Pitch Contour) nel tempo.
    2. Analisi Spettrale: Genera spettrogrammi log-mel per visualizzare l'energia
       e la formante della voce.
    3. Calcolo Variabilità: Misura la deviazione standard del pitch come proxy
       della dinamicità emotiva (voci felici tendono ad avere alta varianza).

🖼️ OUTPUT VISIVO:
    Genera grafici comparativi (Side-by-Side) salvati in `reports/figures/prosody_analysis`,
    ideali per l'inclusione nella tesi o nella presentazione finale.

📦 DIPENDENZE:
    - Librosa (Audio processing)
    - Matplotlib/Seaborn (Visualizzazione)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import random

# --- CONFIGURAZIONE ---
AUDIO_DIR = os.path.join("src", "tts", "bark", "best_audio")
OUTPUT_IMG_DIR = os.path.join("reports", "figures", "prosody_analysis")


def extract_pitch(audio_path):
    """Estrae la fondamentale (F0) usando l'algoritmo PYIN."""
    y, sr = librosa.load(audio_path)
    # Estrazione F0 (Pitch)
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7")
    )
    # Rimuovi i valori NaN (silenzio)
    times = librosa.times_like(f0, sr=sr)
    return times, f0, y, sr


def plot_comparison(pos_path, neg_path, save_name="prosody_comparison.png"):
    """Crea un grafico comparativo: Spettrogramma + Pitch Contour."""

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 8))

    # --- AUDIO POSITIVO ---
    times_p, f0_p, y_p, sr_p = extract_pitch(pos_path)
    D_p = librosa.amplitude_to_db(np.abs(librosa.stft(y_p)), ref=np.max)

    # 1. Spettrogramma Positive
    librosa.display.specshow(D_p, x_axis="time", y_axis="log", ax=ax[0, 0])
    ax[0, 0].set_title(f"Positive Emotion Spectrogram\n({os.path.basename(pos_path)})")
    ax[0, 0].set_xlabel("")

    # 2. Pitch Contour Positive
    ax[1, 0].plot(times_p, f0_p, label="F0 (Pitch)", color="cyan", linewidth=2)
    ax[1, 0].set_title("Pitch Contour (Intonazione)")
    ax[1, 0].set_ylabel("Hz")
    ax[1, 0].set_ylim(50, 400)  # Range voce umana tipico
    ax[1, 0].legend(loc="upper right")
    ax[1, 0].grid(True, alpha=0.3)

    # --- AUDIO NEGATIVO ---
    times_n, f0_n, y_n, sr_n = extract_pitch(neg_path)
    D_n = librosa.amplitude_to_db(np.abs(librosa.stft(y_n)), ref=np.max)

    # 3. Spettrogramma Negative
    librosa.display.specshow(D_n, x_axis="time", y_axis="log", ax=ax[0, 1])
    ax[0, 1].set_title(f"Negative Emotion Spectrogram\n({os.path.basename(neg_path)})")
    ax[0, 1].set_xlabel("")

    # 4. Pitch Contour Negative
    ax[1, 1].plot(times_n, f0_n, label="F0 (Pitch)", color="orange", linewidth=2)
    ax[1, 1].set_title("Pitch Contour (Intonazione)")
    ax[1, 1].set_ylabel("Hz")
    ax[1, 1].set_ylim(50, 400)
    ax[1, 1].legend(loc="upper right")
    ax[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
    save_path = os.path.join(OUTPUT_IMG_DIR, save_name)
    plt.savefig(save_path)
    print(f"✅ Grafico salvato: {save_path}")
    plt.close()

    # --- STATISTICHE ---
    # Calcoliamo la variabilità del pitch (deviazione standard) ignorando i NaNs
    std_p = np.nanstd(f0_p)
    std_n = np.nanstd(f0_n)
    mean_p = np.nanmean(f0_p)
    mean_n = np.nanmean(f0_n)

    print("\n📊 STATISTICHE PROSODICHE:")
    print(f"POSITIVE -> Pitch Medio: {mean_p:.1f} Hz | Variabilità (Std): {std_p:.1f}")
    print(f"NEGATIVE -> Pitch Medio: {mean_n:.1f} Hz | Variabilità (Std): {std_n:.1f}")

    if std_p > std_n:
        print("💡 CONCLUSIONE: L'audio Positivo è più dinamico (maggiore variabilità).")
    else:
        print("💡 CONCLUSIONE: L'audio Negativo è più monotono o simile al positivo.")


def find_files_by_emotion(emotion):
    """Trova file audio nella cartella per una data emozione."""
    files = [
        f
        for f in os.listdir(AUDIO_DIR)
        if f.endswith(".wav") and f"_{emotion.lower()}.wav" in f
    ]
    return [os.path.join(AUDIO_DIR, f) for f in files]


def main():
    print("Ricerca file audio...")
    pos_files = find_files_by_emotion("positive")
    neg_files = find_files_by_emotion("negative")

    if not pos_files or not neg_files:
        print(
            "❌ Errore: Non ho trovato abbastanza file audio (servono almeno 1 Pos e 1 Neg)."
        )
        print("   Esegui prima tts_generator.py!")
        return

    # Scegliamo due file a caso (o prendi i primi)
    # L'ideale sarebbe confrontare lo stesso speaker, ma qui prendiamo a campione
    # p_file = pos_files[0]
    # n_file = neg_files[0]
    # video_name_p = "30438"
    # video_name_n = "254424"
    video_name_p = "30438"
    video_name_n = "26827321"
    p_file = f"{AUDIO_DIR}/{video_name_p}.mp4_positive.wav"
    n_file = f"{AUDIO_DIR}/{video_name_n}.mp4_negative.wav"

    print(f"Confronto:\n 1. {os.path.basename(p_file)}\n 2. {os.path.basename(n_file)}")

    plot_comparison(p_file, n_file)


if __name__ == "__main__":
    main()
