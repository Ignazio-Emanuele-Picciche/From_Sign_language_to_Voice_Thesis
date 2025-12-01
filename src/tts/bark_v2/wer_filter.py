"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              WER FILTERING - QUALITY ASSURANCE (PAPER METHOD)                ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 RIFERIMENTO PAPER:
    "Evaluating Text-to-Speech Synthesis from a Large Discrete Token-based SLM"
    Sezione 3.2 (Intelligibilità) e 4.5 (Listening Tests Filtering).

⚙️ METODOLOGIA:
    1. ASR Model: Whisper Base
    2. Normalizzazione: Lowercase + Rimozione Punteggiatura
    3. Soglia Cut-off: WER > 0.1
"""

import os
import pandas as pd
import whisper
import jiwer
import re
import shutil
from tqdm import tqdm

# --- CONFIGURAZIONE ---
AUDIO_DIR = "src/tts/bark_v2/output_audio_emosign"  # Cartella con i tuoi wav generati
CSV_FILE = "data/processed/golden_test_set.csv"
DISCARD_DIR = "src/tts/bark_v2/output_audio_discarded"  # Dove spostare i file brutti
WER_THRESHOLD = 0.1  # Soglia del paper (0.1 = 10%)

# Carichiamo il modello "base" come specificato nel paper
# Nota: Puoi usare "small" o "medium" se vuoi essere più gentile,
# ma il paper usa "base".
WHISPER_MODEL_SIZE = "base"


def normalize_text(text):
    """
    Normalizzazione come descritto nel paper:
    - Rimozione punteggiatura
    - Lowercase
    """
    if not isinstance(text, str):
        return ""
    # Rimuove tutto ciò che non è parola o spazio
    text = re.sub(r"[^\w\s]", "", text)
    # Converte in minuscolo
    return text.lower().strip()


def calculate_wer(reference, hypothesis):
    """Calcola il Word Error Rate."""
    if not reference or not hypothesis:
        return 1.0  # Errore massimo se stringhe vuote
    return jiwer.wer(reference, hypothesis)


def main():
    # 1. Setup
    print(f"📥 Caricamento modello Whisper '{WHISPER_MODEL_SIZE}'...")
    model = whisper.load_model(WHISPER_MODEL_SIZE)

    os.makedirs(DISCARD_DIR, exist_ok=True)

    # Carica dati originali per avere le Caption (Ground Truth)
    df_truth = pd.read_csv(CSV_FILE)
    # Creiamo un dizionario {video_name: caption} per accesso veloce
    # Puliamo il nome video per matchare i file (rimuoviamo .mp4)
    truth_map = {}
    for _, row in df_truth.iterrows():
        clean_name = str(row["video_name"]).replace(".mp4", "")
        truth_map[clean_name] = row["caption"]

    # 2. Scansione File Audio
    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    print(f"🔍 Trovati {len(audio_files)} file audio da analizzare.")

    results = []
    passed_count = 0

    print("🚀 Avvio calcolo WER...")
    for filename in tqdm(audio_files):
        file_path = os.path.join(AUDIO_DIR, filename)

        # Estrarre l'ID video dal nome file complesso generato da Bark
        # Esempio: "83664512_positive_score2.wav" -> "83664512"
        # Assumiamo che l'ID sia la prima parte prima del primo underscore
        video_id = filename.split("_")[0]

        if video_id not in truth_map:
            print(f"⚠️ Caption non trovata per {filename} (ID: {video_id})")
            continue

        # 3. Trascrizione (ASR)
        # Transcribe restituisce un dizionario, prendiamo il testo
        result = model.transcribe(file_path)
        hypothesis_text = result["text"]

        ground_truth_text = truth_map[video_id]

        # 4. Normalizzazione
        norm_ref = normalize_text(ground_truth_text)
        norm_hyp = normalize_text(hypothesis_text)

        # 5. Calcolo WER
        wer_score = calculate_wer(norm_ref, norm_hyp)

        # 6. Decisione
        status = "KEEP" if wer_score <= WER_THRESHOLD else "DISCARD"

        if status == "KEEP":
            passed_count += 1
        else:
            # Spostiamo fisicamente il file? (Decommenta se vuoi spostarli subito)
            shutil.move(file_path, os.path.join(DISCARD_DIR, filename))

        results.append(
            {
                "filename": filename,
                "video_id": video_id,
                "original_caption": ground_truth_text,
                "whisper_transcript": hypothesis_text,
                "norm_ref": norm_ref,
                "norm_hyp": norm_hyp,
                "WER": round(wer_score, 4),
                "STATUS": status,
            }
        )

    # 7. Salvataggio Report
    df_results = pd.DataFrame(results)
    report_path = "wer_analysis_report.csv"
    df_results.to_csv(report_path, index=False)

    # Statistiche Finali
    total = len(results)
    pass_rate = (passed_count / total) * 100 if total > 0 else 0

    print("\n" + "=" * 50)
    print("📊 RISULTATI ANALISI WER")
    print("=" * 50)
    print(f"Totale Audio Analizzati: {total}")
    print(f"✅ Passati (WER <= {WER_THRESHOLD}): {passed_count}")
    print(f"❌ Scartati (WER > {WER_THRESHOLD}): {total - passed_count}")
    print(f"📈 Pass Rate: {pass_rate:.2f}%")
    print(f"📄 Report salvato in: {report_path}")
    print("=" * 50)

    # Istruzioni per l'utente
    if total - passed_count > 0:
        print("\n💡 CONSIGLIO: Ispeziona 'wer_analysis_report.csv'.")
        print("I file con STATUS='DISCARD' sono quelli dove l'emozione")
        print("ha probabilmente reso il parlato incomprensibile.")


if __name__ == "__main__":
    main()
