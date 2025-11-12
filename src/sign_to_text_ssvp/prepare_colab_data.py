#!/usr/bin/env python3
"""
Prepare How2Sign data for Google Colab upload
Creates TSV manifests and organizes videos for SONAR feature extraction
"""

import os
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_VIDEO_DIR = BASE_DIR / "data" / "raw"
CSV_DIR = BASE_DIR / "data" / "processed" / "how2sign"
OUTPUT_DIR = BASE_DIR / "colab_upload" / "How2Sign_SONAR"

# Video directories
TRAIN_VIDEOS = RAW_VIDEO_DIR / "train" / "raw_videos_front_train"
VAL_VIDEOS = RAW_VIDEO_DIR / "val" / "raw_videos_front_val"
TEST_VIDEOS = RAW_VIDEO_DIR / "test" / "raw_videos_front_test"

# CSV files
TRAIN_CSV = CSV_DIR / "how2sign_realigned_train.csv"
VAL_CSV = CSV_DIR / "how2sign_realigned_val.csv"
TEST_CSV = CSV_DIR / "how2sign_realigned_test.csv"

# Output directories
MANIFEST_DIR = OUTPUT_DIR / "manifests"
VIDEO_OUT_DIR = OUTPUT_DIR / "videos"


def create_tsv_manifest(csv_path, video_dir, output_tsv, split_name):
    """
    Create TSV manifest from How2Sign CSV
    Format: id, video_path, n_frames, translation
    """
    print(f"\n📝 Creating {split_name} manifest...")

    # Read CSV
    df = pd.read_csv(csv_path)
    print(f"   Loaded {len(df)} samples from {csv_path.name}")

    # Prepare TSV data
    tsv_data = []
    missing_videos = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
        video_id = row["SENTENCE_NAME"]
        video_filename = f"{video_id}.mp4"
        video_path = video_dir / video_filename

        # Check if video exists
        if not video_path.exists():
            missing_videos.append(video_filename)
            continue

        # Get translation
        translation = row["SENTENCE"]

        # Create TSV row
        tsv_data.append(
            {
                "id": video_id,
                "video_path": f"videos/{split_name}/{video_filename}",
                "n_frames": 0,  # Will be calculated during extraction
                "translation": translation,
            }
        )

    # Create DataFrame
    tsv_df = pd.DataFrame(tsv_data)

    # Save TSV
    output_tsv.parent.mkdir(parents=True, exist_ok=True)
    tsv_df.to_csv(output_tsv, sep="\t", index=False)

    print(f"   ✅ Saved {len(tsv_df)} samples to {output_tsv.name}")

    if missing_videos:
        print(f"   ⚠️  Missing {len(missing_videos)} videos")

    return tsv_df, missing_videos


def create_video_symlinks(tsv_df, video_dir, output_video_dir, split_name):
    """
    Create symlinks to videos (or copy if needed)
    """
    print(f"\n🔗 Creating video symlinks for {split_name}...")

    output_split_dir = output_video_dir / split_name
    output_split_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    for _, row in tqdm(
        tsv_df.iterrows(), total=len(tsv_df), desc=f"Symlinking {split_name}"
    ):
        video_id = row["id"]
        video_filename = f"{video_id}.mp4"

        src = video_dir / video_filename
        dst = output_split_dir / video_filename

        if src.exists() and not dst.exists():
            try:
                # Create symlink (faster than copy)
                os.symlink(src.absolute(), dst)
                created += 1
            except OSError:
                # If symlink fails, copy file
                shutil.copy2(src, dst)
                created += 1

    print(f"   ✅ Created {created} symlinks in {output_split_dir}")

    return created


def main():
    print("🚀 Preparing How2Sign data for Google Colab...")
    print(f"📂 Output directory: {OUTPUT_DIR}")

    # Create output directories
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process each split
    splits = [
        ("train", TRAIN_CSV, TRAIN_VIDEOS),
        ("val", VAL_CSV, VAL_VIDEOS),
        ("test", TEST_CSV, TEST_VIDEOS),
    ]

    total_samples = 0
    total_videos = 0

    for split_name, csv_path, video_dir in splits:
        print(f"\n{'='*60}")
        print(f"Processing {split_name.upper()} split")
        print(f"{'='*60}")

        # Create TSV manifest
        output_tsv = MANIFEST_DIR / f"{split_name}.tsv"
        tsv_df, missing = create_tsv_manifest(
            csv_path, video_dir, output_tsv, split_name
        )

        # Create video symlinks
        num_videos = create_video_symlinks(tsv_df, video_dir, VIDEO_OUT_DIR, split_name)

        total_samples += len(tsv_df)
        total_videos += num_videos

    # Summary
    print(f"\n{'='*60}")
    print(f"✅ PREPARATION COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Total samples:  {total_samples}")
    print(f"🎥 Total videos:   {total_videos}")
    print(f"📂 Output:         {OUTPUT_DIR}")
    print(f"\n📋 Files created:")
    print(f"   - manifests/train.tsv")
    print(f"   - manifests/val.tsv")
    print(f"   - manifests/test.tsv")
    print(
        f"   - videos/train/ ({len([f for f in (VIDEO_OUT_DIR/'train').glob('*.mp4')])} videos)"
    )
    print(
        f"   - videos/val/ ({len([f for f in (VIDEO_OUT_DIR/'val').glob('*.mp4')])} videos)"
    )
    print(
        f"   - videos/test/ ({len([f for f in (VIDEO_OUT_DIR/'test').glob('*.mp4')])} videos)"
    )
    print(f"\n🚀 Next steps:")
    print(f"   1. Upload {OUTPUT_DIR} to Google Drive as 'How2Sign_SONAR/'")
    print(f"   2. Open Colab notebook: SONAR_Feature_Extraction_Colab.ipynb")
    print(f"   3. Run all cells to extract features")
    print(f"\n⏱️  Estimated Colab time: 8-11 hours (with T4 GPU)")


if __name__ == "__main__":
    main()
