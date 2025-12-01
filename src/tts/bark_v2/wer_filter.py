"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              WER FILTERING - QUALITY ASSURANCE (PAPER METHOD)                ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 RIFERIMENTO PAPER:
    "Evaluating Text-to-Speech Synthesis from a Large Discrete Token-based SLM"
    ArXiv:2405.09768 [cite: 1]

⚙️ LOGICA:
    1. Carica CSV originale (con colonne 'caption' e 'Sentiment').
    2. Trascrive audio generati usando Whisper 'base'[cite: 139].
    3. Calcola WER normalizzato (lowercase, no punteggiatura)[cite: 222].
    4. Filtra: Se WER > 0.1 (10%), marca come DISCARD.
    5. Output: CSV con Sentiment incluso per analisi statistica.
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
CSV_FILE = os.path.join("data", "processed", "golden_test_set.csv")
REPORT_FILE = "src/tts/bark_v2/wer_analysis_report.csv"
WER_THRESHOLD = 0.1  # Soglia rigorosa del paper
WHISPER_MODEL_SIZE = "base"  # Modello usato nel paper [cite: 139]


def normalize_text(text):
    """
    Normalizzazione testo come da metodologia paper[cite: 222]:
    - Rimozione punteggiatura
    - Lowercase
    """
    if not isinstance(text, str):
        return ""
    # Rimuove tutto ciò che non è alfanumerico o spazio
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower().strip()


def calculate_wer(reference, hypothesis):
    """Calcola il Word Error Rate."""
    if not reference:
        return 1.0  # Errore totale se caption vuota
    if not hypothesis:
        return 1.0  # Errore totale se audio muto
    return jiwer.wer(reference, hypothesis)


def main():
    print(f"📥 Caricamento modello Whisper '{WHISPER_MODEL_SIZE}'...")
    model = whisper.load_model(WHISPER_MODEL_SIZE)

    # 1. Caricamento Dati Originali (Ground Truth)
    if not os.path.exists(CSV_FILE):
        print(f"❌ Errore: File dati non trovato: {CSV_FILE}")
        return

    df_truth = pd.read_csv(CSV_FILE)

    # Creiamo un dizionario lookup veloce: VideoID -> {Caption, Sentiment}
    truth_map = {}
    for _, row in df_truth.iterrows():
        # Puliamo il nome video (rimuoviamo estensione .mp4)
        clean_name = str(row["video_name"]).replace(".mp4", "").strip()

        truth_map[clean_name] = {
            "caption": row["caption"],
            "Sentiment": row["Sentiment"],  # <--- SALVIAMO IL SENTIMENT
        }

    # 2. Scansione File Audio Generati
    if not os.path.exists(AUDIO_DIR):
        print(f"❌ Cartella audio non trovata: {AUDIO_DIR}")
        return

    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]
    print(f"🔍 Trovati {len(audio_files)} file audio da analizzare.")

    results = []
    passed_count = 0

    print("🚀 Avvio analisi WER e Sentiment...")
    for filename in tqdm(audio_files):
        file_path = os.path.join(AUDIO_DIR, filename)

        # Parsing ID dal nome file (es: "83664512_positive_score2.wav" -> "83664512")
        video_id = filename.split("_")[0]

        if video_id not in truth_map:
            # Succede se hai generato audio per file non nel CSV attuale
            continue

        data = truth_map[video_id]
        ground_truth_text = data["caption"]
        sentiment_val = data["Sentiment"]

        # 3. Trascrizione (ASR)
        try:
            # Whisper processa l'audio
            asr_result = model.transcribe(file_path)
            hypothesis_text = asr_result["text"]
        except Exception as e:
            print(f"⚠️ Errore trascrizione {filename}: {e}")
            hypothesis_text = ""

        # 4. Normalizzazione [cite: 222]
        norm_ref = normalize_text(ground_truth_text)
        norm_hyp = normalize_text(hypothesis_text)

        # 5. Calcolo WER
        wer_score = calculate_wer(norm_ref, norm_hyp)

        # 6. Applicazione Soglia Paper
        # Se WER <= 0.1 (10%), il file è buono. Altrimenti è scartato.
        status = "KEEP" if wer_score <= WER_THRESHOLD else "DISCARD"

        if status == "KEEP":
            passed_count += 1

        results.append(
            {
                "video_name": video_id + ".mp4",  # Ripristiniamo formato originale
                "filename": filename,
                "Sentiment": sentiment_val,  # <--- COLONNA RICHIESTA
                "original_caption": ground_truth_text,
                "whisper_transcript": hypothesis_text,
                "WER": round(wer_score, 4),
                "STATUS": status,
            }
        )

    # 7. Salvataggio Report
    if not results:
        print("⚠️ Nessun risultato da salvare.")
        return

    df_results = pd.DataFrame(results)

    # Ordiniamo per Status (DISCARD prima) e poi per WER decrescente
    df_results = df_results.sort_values(by=["STATUS", "WER"], ascending=[True, False])

    df_results.to_csv(REPORT_FILE, index=False)

    # 8. Statistiche Finali
    total = len(results)
    pass_rate = (passed_count / total) * 100 if total > 0 else 0

    print("\n" + "=" * 60)
    print("📊 REPORT QUALITÀ BARK (Metodo Paper ArXiv:2405.09768)")
    print("=" * 60)
    print(f"Totale Audio: {total}")
    print(f"✅ KEEP (WER <= 0.1):    {passed_count}")
    print(f"❌ DISCARD (WER > 0.1):  {total - passed_count}")
    print(f"📈 Tasso Accettazione:   {pass_rate:.2f}%")
    print("-" * 60)

    # Analisi rapida per Sentiment (se ci sono dati)
    print("🔍 Analisi Scarti per Sentiment:")
    try:
        discarded_df = df_results[df_results["STATUS"] == "DISCARD"]
        print(discarded_df["Sentiment"].value_counts().sort_index())
    except:
        pass

    print(f"\n📄 Dettagli salvati in: {REPORT_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
