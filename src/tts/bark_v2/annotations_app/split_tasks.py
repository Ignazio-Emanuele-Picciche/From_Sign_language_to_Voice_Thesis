import pandas as pd
import os
from sklearn.utils import shuffle

# --- CONFIGURAZIONE ---
INPUT_FILE = "src/tts/bark_v2/wer_analysis_report.csv"  # Il report del filtro WER
OUTPUT_DIR = "src/tts/bark_v2/assignments"


def split_dataset():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Errore: {INPUT_FILE} non trovato.")
        return

    # 1. Carica e Filtra
    df = pd.read_csv(INPUT_FILE)
    df_keep = df[df["STATUS"] == "KEEP"].copy()

    print(f"Totale campioni validi (KEEP): {len(df_keep)}")

    # 2. Mischia casualmente (ma con seed fisso per riproducibilità)
    df_shuffled = shuffle(df_keep, random_state=42)

    # 3. Divisione a metà
    mid_index = len(df_shuffled) // 2

    set_luca = df_shuffled.iloc[:mid_index]
    set_daniele = df_shuffled.iloc[mid_index:]

    # 4. Salvataggio
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    path_luca = os.path.join(OUTPUT_DIR, "task_luca.csv")
    path_daniele = os.path.join(OUTPUT_DIR, "task_daniele.csv")

    set_luca.to_csv(path_luca, index=False)
    set_daniele.to_csv(path_daniele, index=False)

    print(f"✅ Fatto! Compiti assegnati:")
    print(f"   👤 Luca:    {len(set_luca)} video -> {path_luca}")
    print(f"   👤 Daniele: {len(set_daniele)} video -> {path_daniele}")


if __name__ == "__main__":
    split_dataset()
