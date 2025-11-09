"""
Preparazione Dataset How2Sign
==============================

Converte il dataset How2Sign (31k train, 1.7k val) con landmarks OpenPose
nel formato compatibile con il modello Sign-to-Text.

Usage:
    python src/sign_to_text/data_preparation/prepare_how2sign_dataset.py
    python src/sign_to_text/data_preparation/prepare_how2sign_dataset.py --analyze_only
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from collections import Counter
import re


def load_openpose_landmarks(json_dir):
    """
    Carica landmarks OpenPose da directory JSON.

    OpenPose format: 1 JSON per frame
    - pose_keypoints_2d: 25 keypoints x 3 coords (x, y, confidence) = 75 values
    - hand_left_keypoints_2d: 21 keypoints x 3 = 63 values
    - hand_right_keypoints_2d: 21 keypoints x 3 = 63 values
    - face_keypoints_2d: 70 keypoints x 3 = 210 values

    Total: 135 keypoints x 3 = 405 values per frame

    Args:
        json_dir: Directory con i JSON OpenPose (1 file per frame)

    Returns:
        landmarks: numpy array (n_frames, 405) - tutti i landmarks flatten
        None se directory non esiste o vuota
    """
    json_dir = Path(json_dir)

    if not json_dir.exists():
        return None

    # Trova tutti i JSON (ordinati per numero frame)
    json_files = sorted(json_dir.glob("*.json"))

    if len(json_files) == 0:
        return None

    frames_data = []

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # OpenPose output: {"people": [...]}
            if "people" not in data or len(data["people"]) == 0:
                # Nessuna persona rilevata -> frame vuoto (tutti zero)
                frame_landmarks = np.zeros(405, dtype=np.float32)
            else:
                person = data["people"][0]  # Prima persona rilevata

                # Estrai keypoints (già flatten in lista)
                pose = person.get("pose_keypoints_2d", [0] * 75)
                hand_left = person.get("hand_left_keypoints_2d", [0] * 63)
                hand_right = person.get("hand_right_keypoints_2d", [0] * 63)
                face = person.get("face_keypoints_2d", [0] * 210)

                # Concatena tutti i keypoints
                frame_landmarks = np.array(
                    pose + hand_left + hand_right + face, dtype=np.float32
                )

            frames_data.append(frame_landmarks)

        except Exception as e:
            # Se errore, usa frame vuoto
            frame_landmarks = np.zeros(405, dtype=np.float32)
            frames_data.append(frame_landmarks)

    # Stack tutti i frame
    landmarks = np.stack(frames_data, axis=0)  # (n_frames, 405)

    return landmarks


def analyze_how2sign_dataset(
    train_xlsx="data/raw/train/how2sign_train.xlsx",
    val_xlsx="data/raw/val/how2sign_val.xlsx",
    train_openpose_dir="data/raw/train/openpose_output_train/json",
    val_openpose_dir="data/raw/val/openpose_output_val/json",
    output_dir="results/how2sign_analysis",
):
    """Analizza dataset How2Sign e verifica landmarks disponibili."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("📊 ANALISI DATASET HOW2SIGN")
    print("=" * 80)

    # 1. Carica Excel
    print(f"\n1️⃣  Caricamento dataset...")
    df_train = pd.read_excel(train_xlsx)
    df_val = pd.read_excel(val_xlsx)

    print(f"   ✓ Train: {len(df_train)} samples")
    print(f"   ✓ Val:   {len(df_val)} samples")
    print(f"   ✓ Colonne: {df_train.columns.tolist()}")

    # 2. Analisi caption
    print(f"\n2️⃣  Analisi caption...")

    def count_words(text):
        if pd.isna(text):
            return 0
        return len(str(text).split())

    df_train["num_words"] = df_train["SENTENCE"].apply(count_words)
    df_val["num_words"] = df_val["SENTENCE"].apply(count_words)

    print(f"\n   📏 Lunghezza caption TRAIN:")
    print(f"      - Min:    {df_train['num_words'].min()}")
    print(f"      - Max:    {df_train['num_words'].max()}")
    print(f"      - Mean:   {df_train['num_words'].mean():.2f}")
    print(f"      - Median: {df_train['num_words'].median():.0f}")

    print(f"\n   📏 Lunghezza caption VAL:")
    print(f"      - Min:    {df_val['num_words'].min()}")
    print(f"      - Max:    {df_val['num_words'].max()}")
    print(f"      - Mean:   {df_val['num_words'].mean():.2f}")
    print(f"      - Median: {df_val['num_words'].median():.0f}")

    # 3. Vocabulary
    print(f"\n3️⃣  Analisi vocabulary...")

    def tokenize(text):
        if pd.isna(text):
            return []
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", "", text)
        return text.split()

    all_words = []
    for sentence in df_train["SENTENCE"]:
        all_words.extend(tokenize(sentence))

    word_counts = Counter(all_words)

    print(f"   📚 Vocabulary:")
    print(f"      - Unique words: {len(word_counts)}")
    print(f"      - Total words:  {len(all_words)}")
    print(f"      - Top 20:")
    for word, count in word_counts.most_common(20):
        print(f"         '{word}': {count}")

    # 4. Check landmarks OpenPose
    print(f"\n4️⃣  Verifica landmarks OpenPose...")

    train_openpose_dir = Path(train_openpose_dir)
    val_openpose_dir = Path(val_openpose_dir)

    # Filtra NaN
    df_train_valid = df_train[~df_train["SENTENCE_NAME"].isna()].copy()

    # Sample check (primi 100 samples)
    sample_train = df_train_valid.head(100)
    available_landmarks = 0
    missing_landmarks = 0

    for idx, row in sample_train.iterrows():
        sentence_name = str(row["SENTENCE_NAME"])
        landmark_dir = train_openpose_dir / sentence_name

        if landmark_dir.exists() and len(list(landmark_dir.glob("*.json"))) > 0:
            available_landmarks += 1
        else:
            missing_landmarks += 1

    print(f"   📂 Sample check (primi 100 train):")
    print(f"      - Landmarks disponibili: {available_landmarks}")
    print(f"      - Landmarks mancanti:     {missing_landmarks}")

    # 5. Test caricamento landmark
    print(f"\n5️⃣  Test caricamento landmark...")

    # Usa solo samples validi
    df_train_valid = df_train[~df_train["SENTENCE_NAME"].isna()].copy()

    if len(df_train_valid) > 0:
        first_sample = df_train_valid.iloc[0]
        landmark_dir = train_openpose_dir / str(first_sample["SENTENCE_NAME"])

        if landmark_dir.exists():
            landmarks = load_openpose_landmarks(landmark_dir)

            if landmarks is not None:
                print(f"   ✅ Landmark caricati con successo!")
                print(f"      - Sample: {first_sample['SENTENCE_NAME']}")
                print(f"      - Shape: {landmarks.shape}")
                print(f"      - Frames: {landmarks.shape[0]}")
                print(
                    f"      - Features: {landmarks.shape[1]} (OpenPose: 135 keypoints x 3)"
                )
                print(f"      - Caption: '{first_sample['SENTENCE'][:80]}...'")
            else:
                print(f"   ❌ Errore caricamento landmark")
        else:
            print(f"   ❌ Directory landmark non trovata: {landmark_dir}")
    else:
        print(f"   ❌ Nessun sample valido con SENTENCE_NAME")

    # 6. Salva statistiche
    print(f"\n6️⃣  Salvataggio statistiche...")

    stats = {
        "train_size": int(len(df_train)),
        "val_size": int(len(df_val)),
        "total_words": int(len(all_words)),
        "unique_words": int(len(word_counts)),
        "avg_words_per_caption_train": float(df_train["num_words"].mean()),
        "median_words_per_caption_train": float(df_train["num_words"].median()),
        "min_words_train": int(df_train["num_words"].min()),
        "max_words_train": int(df_train["num_words"].max()),
        "avg_words_per_caption_val": float(df_val["num_words"].mean()),
        "landmarks_available_sample": int(available_landmarks),
        "landmarks_missing_sample": int(missing_landmarks),
    }

    with open(output_dir / "how2sign_statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"   ✓ Statistiche: {output_dir / 'how2sign_statistics.json'}")

    print(f"\n" + "=" * 80)
    print(f"✅ ANALISI COMPLETATA!")
    print(f"=" * 80)
    print(f"\n🎯 How2Sign è PRONTO per training!")
    print(f"   - Train: {len(df_train)} samples")
    print(f"   - Val:   {len(df_val)} samples")
    print(f"   - Landmarks: OpenPose 135 keypoints (405 features)")
    print(f"   - Vocab: {len(word_counts)} unique words")

    return df_train, df_val, stats


def prepare_how2sign_splits(
    train_xlsx="data/raw/train/how2sign_train.xlsx",
    val_xlsx="data/raw/val/how2sign_val.xlsx",
    output_dir="results/how2sign_splits",
):
    """
    Prepara CSV splits compatibili con SignLanguageDataset.

    Output CSV format:
    - video_name: SENTENCE_NAME (path relativo a landmarks)
    - caption: SENTENCE
    - source_collection: "How2Sign"
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("📝 PREPARAZIONE SPLITS HOW2SIGN")
    print("=" * 80)

    # Carica Excel
    print(f"\n1️⃣  Caricamento dataset...")
    df_train = pd.read_excel(train_xlsx)
    df_val = pd.read_excel(val_xlsx)

    print(f"   ✓ Train: {len(df_train)}")
    print(f"   ✓ Val:   {len(df_val)}")

    # Converti formato
    print(f"\n2️⃣  Conversione formato...")

    def convert_to_dataset_format(df, split_name):
        """Converti Excel How2Sign -> CSV compatibile con SignLanguageDataset."""

        df_converted = pd.DataFrame(
            {
                "video_name": df["SENTENCE_NAME"],  # Es: --7E2sU6zP4_10-5-rgb_front
                "caption": df["SENTENCE"],
                "Source collection": "How2Sign",
                "video_id": df["VIDEO_ID"],
                "sentence_id": df["SENTENCE_ID"],
                "start_time": df["START"],
                "end_time": df["END"],
            }
        )

        # Filtra caption valide
        df_converted = df_converted[~df_converted["caption"].isna()].copy()

        print(f"   ✓ {split_name}: {len(df_converted)} samples validi")

        return df_converted

    train_converted = convert_to_dataset_format(df_train, "Train")
    val_converted = convert_to_dataset_format(df_val, "Val")

    # Salva CSV
    print(f"\n3️⃣  Salvataggio splits...")

    train_converted.to_csv(output_dir / "train_split.csv", index=False)
    val_converted.to_csv(output_dir / "val_split.csv", index=False)

    print(f"   ✓ Train: {output_dir / 'train_split.csv'}")
    print(f"   ✓ Val:   {output_dir / 'val_split.csv'}")

    # Info summary
    print(f"\n📊 Summary:")
    print(f"   - Train samples: {len(train_converted)}")
    print(f"   - Val samples:   {len(val_converted)}")
    print(f"   - Formato compatibile con SignLanguageDataset")
    print(f"   - Landmarks: data/raw/train/openpose_output_train/json/")
    print(f"   - Landmarks: data/raw/val/openpose_output_val/json/")

    print(f"\n✅ Splits pronti per training!")
    print(f"\n")

    return train_converted, val_converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara dataset How2Sign")
    parser.add_argument(
        "--analyze_only", action="store_true", help="Solo analisi, senza creare splits"
    )
    parser.add_argument(
        "--train_xlsx",
        type=str,
        default="data/raw/train/how2sign_train.xlsx",
    )
    parser.add_argument(
        "--val_xlsx",
        type=str,
        default="data/raw/val/how2sign_val.xlsx",
    )

    args = parser.parse_args()

    # Analisi dataset
    analyze_how2sign_dataset(
        train_xlsx=args.train_xlsx,
        val_xlsx=args.val_xlsx,
    )

    # Crea splits (se non solo analisi)
    if not args.analyze_only:
        prepare_how2sign_splits(
            train_xlsx=args.train_xlsx,
            val_xlsx=args.val_xlsx,
        )
