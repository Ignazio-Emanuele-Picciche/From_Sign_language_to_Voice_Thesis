"""
╔══════════════════════════════════════════════════════════════════════════════╗
║            MERGE CAPTIONS - DATA INTEGRATION & ENRICHMENT                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 DESCRIZIONE:
    Modulo di pre-processing critico per la pipeline TTS.
    Questo script risolve il disallineamento tra i risultati numerici del Meta-Learner
    (che contengono solo probabilità e label) e i dati testuali originali (trascrizioni).

    Esegue un'operazione di JOIN relazionale tra il file delle predizioni e il
    Golden Test Set originale, utilizzando il `video_name` come chiave primaria.

🎯 OBIETTIVO:
    Garantire che il generatore vocale (Bark) riceva non solo l'emozione da simulare,
    ma anche il testo corretto da pronunciare (Caption), evitando l'uso eccessivo
    di template di fallback generici.

🔄 FLUSSO DATI:
    Input A: final_metalearner_predictions_for_tts.csv (Output del classificatore)
    Input B: golden_test_set.csv (Ground Truth con trascrizioni)

    Processo: Left Join su 'video_name' -> Gestione valori nulli (fillna)

    Output: final_predictions_with_captions.csv (Dataset completo pronto per TTS)
"""

import pandas as pd
import os
import sys

# --- SETUP PERCORSI ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)

# File Input 1: Predizioni del Meta-Learner (senza caption o con caption vuote)
PREDICTIONS_FILE = os.path.join(
    BASE_DIR,
    "src",
    "models",
    "three_classes",
    "text_plus_video_metalearner_to_sentiment",
    "results",
    "final_metalearner_predictions_for_tts.csv",
)

# File Input 2: Golden Test Set originale (con le caption corrette)
GOLDEN_TEST_FILE = os.path.join(BASE_DIR, "data", "processed", "golden_test_set.csv")

# File Output: Predizioni arricchite
OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "src",
    "tts",
    "bark",
    "final_predictions_with_captions.csv",
)


def main():
    print("=" * 60)
    print("MERGE CAPTION TOOL")
    print("=" * 60)

    # 1. Caricamento File
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"❌ File predizioni non trovato: {PREDICTIONS_FILE}")
        return
    if not os.path.exists(GOLDEN_TEST_FILE):
        print(f"❌ File Golden Test non trovato: {GOLDEN_TEST_FILE}")
        return

    print("Caricamento dataset...")
    df_pred = pd.read_csv(PREDICTIONS_FILE)
    df_golden = pd.read_csv(GOLDEN_TEST_FILE)

    print(f"  Predizioni: {len(df_pred)} righe")
    print(f"  Golden Set: {len(df_golden)} righe")

    # 2. Pulizia Nomi Video (Cruciale per il Join)
    # A volte i video name hanno estensioni diverse o path parziali. Normalizziamo.
    # Assumiamo che video_name sia la chiave univoca.

    # Rimuoviamo eventuali spazi bianchi
    df_pred["video_name"] = df_pred["video_name"].astype(str).str.strip()
    df_golden["video_name"] = df_golden["video_name"].astype(str).str.strip()

    # 3. Esecuzione Merge
    print("\nEsecuzione Merge...")

    # Selezioniamo solo le colonne utili dal Golden Set per evitare duplicati
    cols_to_merge = ["video_name", "caption"]

    # Facciamo un LEFT JOIN: manteniamo tutte le predizioni, aggiungiamo caption dove possibile
    merged_df = pd.merge(
        df_pred,
        df_golden[cols_to_merge],
        on="video_name",
        how="left",
        suffixes=(
            "_old",
            "",
        ),  # Se c'era già una colonna caption, diventerà caption_old
    )

    # 4. Gestione Caption
    # Se 'caption' (quella nuova dal golden) è vuota, proviamo a tenere quella vecchia (se c'era)
    if "caption_old" in merged_df.columns:
        merged_df["caption"] = merged_df["caption"].fillna(merged_df["caption_old"])
        merged_df.drop(columns=["caption_old"], inplace=True)

    # Conteggio successi
    captions_found = merged_df["caption"].notna().sum()
    captions_missing = merged_df["caption"].isna().sum()

    print(f"  Caption trovate: {captions_found}")
    print(f"  Caption mancanti: {captions_missing}")

    if captions_missing > 0:
        print("⚠️  Attenzione: Alcuni video non hanno caption nel Golden Set.")
        # Ispezione dei mancanti (primi 5)
        print(
            "  Esempi di video senza caption:",
            merged_df[merged_df["caption"].isna()]["video_name"].head(5).tolist(),
        )

    # 5. Salvataggio
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    merged_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 60)
    print(f"✅ FILE GENERATO: {OUTPUT_FILE}")
    print("   Ora aggiorna tts_generator.py per usare questo file!")
    print("=" * 60)


if __name__ == "__main__":
    main()
