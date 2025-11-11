"""
Prepare How2Sign Dataset for SSVP-SLT
======================================

Convert How2Sign dataset from your format to SSVP-SLT expected format.

Your format:
    - CSV with columns: video_name, caption, split, duration, n_frames
    - Videos in: data/raw/how2sign/videos/

SSVP-SLT format:
    - TSV manifest with columns: id, duration, text
    - Videos organized by split: data/how2sign_ssvp/clips/{split}/

Usage:
    python prepare_how2sign_for_ssvp.py \\
        --input_csv results/how2sign_splits/train_split.csv \\
        --video_dir data/raw/how2sign/videos \\
        --output_dir data/how2sign_ssvp \\
        --split train
"""

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import pandas as pd


def get_video_duration(video_path: Path) -> Optional[float]:
    """
    Get video duration using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        Duration in seconds, or None if failed
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        print(f"⚠️  Failed to get duration for {video_path.name}: {e}")
        return None


def find_video_file(video_name: str, video_dir: Path) -> Optional[Path]:
    """
    Find video file in directory (handles various extensions).

    Args:
        video_name: Video name (with or without extension)
        video_dir: Directory containing videos

    Returns:
        Path to video file, or None if not found
    """
    # Remove extension if present
    video_base = Path(video_name).stem

    # Try common video extensions
    extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"]

    for ext in extensions:
        video_path = video_dir / f"{video_base}{ext}"
        if video_path.exists():
            return video_path

    # Try searching recursively
    for ext in extensions:
        matches = list(video_dir.rglob(f"{video_base}{ext}"))
        if matches:
            return matches[0]

    return None


def prepare_split(
    input_csv: Path,
    video_dir: Path,
    output_dir: Path,
    split: str,
    copy_videos: bool = False,
    max_samples: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Prepare single split (train/val/test) for SSVP-SLT.

    Args:
        input_csv: Path to input CSV file
        video_dir: Directory containing source videos
        output_dir: Output directory for SSVP-SLT format
        split: Split name (train/val/test)
        copy_videos: If True, copy videos; if False, create symlinks
        max_samples: Maximum number of samples to process (for testing)

    Returns:
        Tuple of (successful_count, failed_count)
    """
    print(f"\n{'=' * 80}")
    print(f"📊 Processing {split.upper()} split")
    print(f"{'=' * 80}")

    # Read input CSV
    print(f"\n1️⃣  Reading CSV: {input_csv}")
    df = pd.read_csv(input_csv)

    if max_samples:
        df = df.head(max_samples)
        print(f"   ⚠️  Limited to {max_samples} samples for testing")

    print(f"   ✓ Found {len(df)} samples")

    # Create output directories
    clips_dir = output_dir / "clips" / split
    clips_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = output_dir / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n2️⃣  Output directories:")
    print(f"   Clips:    {clips_dir}")
    print(f"   Manifest: {manifest_dir}")

    # Prepare manifest data
    print(f"\n3️⃣  Processing videos...")
    manifest_data = []
    success_count = 0
    failed_count = 0
    failed_videos = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"   Processing {split}"):
        video_name = row["video_name"]
        caption = row["caption"]

        # Skip rows with missing video_name or caption
        if pd.isna(video_name) or pd.isna(caption):
            failed_count += 1
            continue

        # Find source video
        source_video = find_video_file(video_name, video_dir)

        if source_video is None:
            failed_count += 1
            failed_videos.append(video_name)
            continue

        # Get video extension
        video_ext = source_video.suffix

        # Destination video path
        dest_video = clips_dir / f"{video_name}{video_ext}"

        # Copy or symlink video
        try:
            if copy_videos:
                if not dest_video.exists():
                    shutil.copy2(source_video, dest_video)
            else:
                if not dest_video.exists():
                    # Create relative symlink
                    rel_source = Path(os.path.relpath(source_video, dest_video.parent))
                    dest_video.symlink_to(rel_source)
        except Exception as e:
            print(f"\n   ⚠️  Failed to copy/link {video_name}: {e}")
            failed_count += 1
            failed_videos.append(video_name)
            continue

        # Get duration (from CSV if available, otherwise probe)
        if "duration" in row and pd.notna(row["duration"]):
            duration = float(row["duration"])
        else:
            duration = get_video_duration(dest_video)
            if duration is None:
                # Use a default duration
                duration = 5.0

        # Add to manifest
        manifest_data.append(
            {"id": video_name, "duration": round(duration, 2), "text": caption}
        )

        success_count += 1

    # Write manifest TSV
    manifest_file = manifest_dir / f"{split}.tsv"
    print(f"\n4️⃣  Writing manifest: {manifest_file}")

    with open(manifest_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "duration", "text"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(manifest_data)

    print(f"   ✓ Wrote {len(manifest_data)} entries")

    # Summary
    print(f"\n📊 {split.upper()} Summary:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failed:  {failed_count}")

    if failed_videos:
        print(f"\n   Failed videos (first 10):")
        for video in failed_videos[:10]:
            print(f"      - {video}")
        if len(failed_videos) > 10:
            print(f"      ... and {len(failed_videos) - 10} more")

    return success_count, failed_count


def create_config_file(output_dir: Path, splits: List[str]):
    """Create config file with dataset info."""
    config = {
        "dataset": "How2Sign",
        "format": "SSVP-SLT",
        "splits": splits,
        "structure": {"clips": "clips/{split}/", "manifest": "manifest/{split}.tsv"},
        "manifest_format": {"columns": ["id", "duration", "text"], "delimiter": "\\t"},
    }

    config_file = output_dir / "dataset_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Created config: {config_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare How2Sign dataset for SSVP-SLT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare train split
  python prepare_how2sign_for_ssvp.py \\
      --input_csv results/how2sign_splits/train_split.csv \\
      --video_dir data/raw/how2sign/videos \\
      --output_dir data/how2sign_ssvp \\
      --split train
  
  # Prepare all splits
  python prepare_how2sign_for_ssvp.py \\
      --input_csv results/how2sign_splits/train_split.csv \\
      --video_dir data/raw/how2sign/videos \\
      --output_dir data/how2sign_ssvp \\
      --split train
  
  # Quick test with 100 samples
  python prepare_how2sign_for_ssvp.py \\
      --input_csv results/how2sign_splits/train_split.csv \\
      --video_dir data/raw/how2sign/videos \\
      --output_dir data/how2sign_ssvp \\
      --split train \\
      --max_samples 100
        """,
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input CSV file with columns: video_name, caption, split",
    )

    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing source videos",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/how2sign_ssvp",
        help="Output directory for SSVP-SLT format (default: data/how2sign_ssvp)",
    )

    parser.add_argument(
        "--split",
        type=str,
        required=True,
        choices=["train", "val", "test"],
        help="Split to prepare (train/val/test)",
    )

    parser.add_argument(
        "--copy", action="store_true", help="Copy videos instead of creating symlinks"
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )

    args = parser.parse_args()

    # Convert to Path objects
    input_csv = Path(args.input_csv)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    # Validate inputs
    if not input_csv.exists():
        print(f"❌ Input CSV not found: {input_csv}")
        return 1

    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        return 1

    print("=" * 80)
    print("🎬 HOW2SIGN → SSVP-SLT FORMAT CONVERTER")
    print("=" * 80)
    print(f"\n📂 Paths:")
    print(f"   Input CSV:  {input_csv}")
    print(f"   Videos:     {video_dir}")
    print(f"   Output:     {output_dir}")
    print(f"\n⚙️  Settings:")
    print(f"   Split:      {args.split}")
    print(f"   Copy files: {args.copy}")
    if args.max_samples:
        print(f"   Max samples: {args.max_samples}")

    # Process split
    success, failed = prepare_split(
        input_csv=input_csv,
        video_dir=video_dir,
        output_dir=output_dir,
        split=args.split,
        copy_videos=args.copy,
        max_samples=args.max_samples,
    )

    # Create config
    create_config_file(output_dir, [args.split])

    # Final summary
    print("\n" + "=" * 80)
    print("✅ CONVERSION COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Total: {success} videos prepared")

    if failed > 0:
        print(f"⚠️  {failed} videos failed")

    print(f"\n📍 Output structure:")
    print(f"   {output_dir}/")
    print(f"   ├── clips/{args.split}/     (video files)")
    print(f"   ├── manifest/{args.split}.tsv  (annotations)")
    print(f"   └── dataset_config.json  (metadata)")

    print(f"\n🚀 Next steps:")
    print(f"   1. Prepare other splits (val, test)")
    print(f"   2. Download pretrained model:")
    print(f"      python download_pretrained.py --model base")
    print(f"   3. Fine-tune:")
    print(f"      python finetune_how2sign.py --config configs/finetune_base.yaml")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    import os

    sys.exit(main())
