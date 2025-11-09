"""
Estrazione Landmarks MediaPipe - Batch Processing
=================================================

Estrae landmarks da tutti i video del dataset utterances usando MediaPipe
tramite sign-language-translator.

Usage:
    python src/sign_to_text/data_preparation/extract_landmarks_mediapipe.py --resume
    python src/sign_to_text/data_preparation/extract_landmarks_mediapipe.py --video_list results/utterances_analysis/train_split.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import sign_language_translator as slt
import json


def extract_landmarks_batch(
    csv_path="data/processed/utterances_with_translations.csv",
    video_dir="data/raw/ASLLRP/batch_utterance_video_v3_1",
    output_dir="data/processed/sign_language_landmarks",
    resume=True,
    max_videos=None,
):
    """
    Estrae landmarks MediaPipe da batch di video.

    Args:
        csv_path: Path al CSV con lista video
        video_dir: Directory con video MP4
        output_dir: Directory output landmarks
        resume: Se True, salta video già processati
        max_videos: Max numero video da processare (None = tutti)
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_dir = Path(video_dir)

    # Carica lista video
    print(f"\n📂 Caricamento dataset...")
    df = pd.read_csv(csv_path)

    # Filtra solo caption valide
    df = df[~df["caption"].isna()].copy()
    print(f"   ✓ {len(df)} utterances con caption valida")

    if max_videos:
        df = df.head(max_videos)
        print(f"   ⚠️  Limitato a {max_videos} video per test")

    # Check video esistenti
    df["video_path"] = df["video_name"].apply(lambda x: video_dir / x)
    df["exists"] = df["video_path"].apply(lambda p: p.exists())

    df_valid = df[df["exists"]].copy()
    print(f"   ✓ {len(df_valid)} video esistenti")

    # Resume: salta già processati
    if resume:
        df_valid["landmark_path"] = df_valid["video_name"].apply(
            lambda x: output_dir / f"{Path(x).stem}_landmarks.npz"
        )
        df_valid["processed"] = df_valid["landmark_path"].apply(lambda p: p.exists())

        n_processed = df_valid["processed"].sum()
        df_todo = df_valid[~df_valid["processed"]].copy()

        print(f"\n♻️  Resume mode:")
        print(f"   - Già processati: {n_processed}")
        print(f"   - Da processare:  {len(df_todo)}")
    else:
        df_todo = df_valid

    if len(df_todo) == 0:
        print("\n✅ Tutti i video sono già stati processati!")
        return

    # Batch processing
    print(f"\n🎬 Inizio estrazione landmarks...")
    print(f"   Video: {len(df_todo)}")
    print(f"   Output: {output_dir}")

    # Inizializza modello MediaPipe (UNA VOLTA sola)
    print(f"\n🔧 Caricamento modello MediaPipe...")
    model = slt.models.MediaPipeLandmarksModel()
    print(f"   ✓ Modello caricato")

    successes = []
    failures = []

    start_time = time.time()

    progress_bar = tqdm(
        df_todo.iterrows(), total=len(df_todo), desc="Estrazione landmarks"
    )

    for idx, row in progress_bar:
        video_path = row["video_path"]
        video_name = row["video_name"]
        stem = Path(video_name).stem

        output_path = output_dir / f"{stem}_landmarks.npz"

        try:
            # Carica video
            video = slt.Video(str(video_path))

            # Estrai landmarks (MediaPipe holistic) - 3D world coordinates
            landmarks_3d_tensor = model.embed(
                video.iter_frames(), landmark_type="world"
            )

            if landmarks_3d_tensor is None or len(landmarks_3d_tensor) == 0:
                raise ValueError("Landmarks non estratti")

            # Converti tensor a numpy
            landmarks_3d = np.array(landmarks_3d_tensor)

            # Reshape landmarks: (n_frames, 375) -> (n_frames, 75, 5)
            n_frames = landmarks_3d.shape[0]
            landmarks_reshaped = landmarks_3d.reshape(n_frames, 75, 5)

            # Flatten per formato compatibile
            n_frames, n_landmarks, n_coords = landmarks_reshaped.shape
            landmarks_flat = landmarks_reshaped.reshape(n_frames, -1)  # (n_frames, 375)

            # Metadata
            metadata = {
                "video_name": video_name,
                "caption": row["caption"],
                "source_collection": row["Source collection"],
                "n_frames": int(n_frames),
                "fps": float(video.fps) if hasattr(video, "fps") else None,
                "duration_sec": float(video.duration),
                "landmark_shape": landmarks_flat.shape,
                "extraction_timestamp": time.time(),
            }

            # Salva
            np.savez_compressed(
                output_path,
                landmarks_3d=landmarks_flat.astype(np.float32),
                metadata=metadata,
            )

            successes.append(
                {
                    "video_name": video_name,
                    "n_frames": n_frames,
                    "file_size_kb": output_path.stat().st_size / 1024,
                }
            )

            progress_bar.set_postfix(
                {"Success": len(successes), "Failed": len(failures), "Frames": n_frames}
            )

        except Exception as e:
            failures.append({"video_name": video_name, "error": str(e)})

            progress_bar.set_postfix(
                {
                    "Success": len(successes),
                    "Failed": len(failures),
                    "Error": str(e)[:30],
                }
            )

            continue

    elapsed = time.time() - start_time

    # Summary report
    print(f"\n" + "=" * 80)
    print(f"✅ ESTRAZIONE COMPLETATA!")
    print(f"=" * 80)
    print(f"\n📊 Statistiche:")
    print(f"   - Successi:  {len(successes)} ({len(successes)/len(df_todo)*100:.1f}%)")
    print(f"   - Fallimenti: {len(failures)} ({len(failures)/len(df_todo)*100:.1f}%)")
    print(f"   - Tempo totale: {elapsed/60:.1f} min")
    print(f"   - Tempo medio:  {elapsed/len(df_todo):.2f} sec/video")

    if successes:
        avg_frames = np.mean([s["n_frames"] for s in successes])
        avg_size = np.mean([s["file_size_kb"] for s in successes])
        total_size = sum([s["file_size_kb"] for s in successes])

        print(f"\n   📏 Landmarks:")
        print(f"      - Frames medi: {avg_frames:.1f}")
        print(f"      - File size medio: {avg_size:.1f} KB")
        print(f"      - Totale: {total_size/1024:.1f} MB")

    # Salva report
    report = {
        "total_videos": int(len(df_todo)),
        "successes": int(len(successes)),
        "failures": int(len(failures)),
        "elapsed_sec": float(elapsed),
        "avg_time_per_video": float(elapsed / len(df_todo)),
        "success_details": successes,
        "failure_details": failures,
    }

    report_path = output_dir / "extraction_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Report: {report_path}")

    if failures:
        failures_path = output_dir / "failed_videos.txt"
        with open(failures_path, "w") as f:
            for fail in failures:
                f.write(f"{fail['video_name']}\t{fail['error']}\n")
        print(f"⚠️  Failed videos: {failures_path}")

    print(f"\n🚀 Prossimo step: Training Seq2Seq model")
    print(f"   Landmarks: {output_dir}")
    print(f"   Splits: results/utterances_analysis/train_split.csv")
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estrai landmarks MediaPipe da video ASL"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/processed/utterances_with_translations.csv",
    )
    parser.add_argument(
        "--video_dir", type=str, default="data/raw/ASLLRP/batch_utterance_video_v3_1"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/processed/sign_language_landmarks"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Salta video già processati"
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Max video da processare (per test)",
    )

    args = parser.parse_args()

    extract_landmarks_batch(
        csv_path=args.csv_path,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        resume=args.resume,
        max_videos=args.max_videos,
    )
