"""
Test Sign Language Feature Extraction
======================================

Questo script testa l'estrazione di landmarks da video ASL usando sign-language-translator.
Serve come proof-of-concept per la pipeline Video→Text.

Usage:
    python test_sign_language_extraction.py --video_path data/raw/ASLLRP/videos/83664512.mp4
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import sign_language_translator as slt

    SLT_AVAILABLE = True
except ImportError:
    print("⚠️  sign-language-translator not installed!")
    print("Install with: pip install 'sign-language-translator[all]'")
    SLT_AVAILABLE = False


def test_single_video(video_path: str, show_visualization: bool = True):
    """
    Testa l'estrazione di landmarks da un singolo video.

    Args:
        video_path: Percorso al video .mp4
        show_visualization: Se mostrare la visualizzazione 3D dei landmarks
    """
    if not SLT_AVAILABLE:
        print("❌ Cannot proceed without sign-language-translator")
        return None

    print(f"\n{'='*60}")
    print(f"Testing Feature Extraction on: {os.path.basename(video_path)}")
    print(f"{'='*60}\n")

    # 1. Load video
    print("📹 Loading video...")
    try:
        video = slt.Video(video_path)
        print(f"   ✓ Video loaded successfully")
        print(f"   - Duration: {video.duration:.2f} seconds")
        print(f"   - Total frames: {len(video)}")
        print(f"   - Shape: {video.shape}")
    except Exception as e:
        print(f"   ❌ Error loading video: {e}")
        return None

    # 2. Extract landmarks using MediaPipe
    print("\n🔍 Extracting MediaPipe landmarks...")
    try:
        model = slt.models.MediaPipeLandmarksModel()

        # Extract both world (3D) and image (2D) coordinates
        print("   - Extracting 3D world coordinates...")
        landmarks_3d = model.embed(video.iter_frames(), landmark_type="world")

        print("   - Extracting 2D image coordinates...")
        landmarks_2d = model.embed(video.iter_frames(), landmark_type="image")

        print(f"   ✓ Landmarks extracted successfully")
        print(f"   - 3D landmarks shape: {landmarks_3d.shape}")
        print(f"   - 2D landmarks shape: {landmarks_2d.shape}")
        print(f"   - Expected: (n_frames, 75 landmarks * 5 coords)")

    except Exception as e:
        print(f"   ❌ Error extracting landmarks: {e}")
        return None

    # 3. Reshape for visualization
    print("\n📊 Landmark Statistics:")
    landmarks_reshaped = landmarks_3d.reshape((-1, 75, 5))
    print(f"   - Reshaped to: {landmarks_reshaped.shape}")
    print(f"   - Landmarks per frame: {landmarks_reshaped.shape[1]}")
    print(f"   - Coordinates per landmark: {landmarks_reshaped.shape[2]}")
    print(f"   - Mean value: {landmarks_3d.mean():.4f}")
    print(f"   - Std value: {landmarks_3d.std():.4f}")
    print(f"   - Min/Max: {landmarks_3d.min():.4f} / {landmarks_3d.max():.4f}")

    # 4. Visualize (optional)
    if show_visualization:
        print("\n🎨 Showing landmark visualization...")
        try:
            landmarks_obj = slt.Landmarks(
                landmarks_reshaped, connections="mediapipe-world"
            )
            landmarks_obj.show()
            print("   ✓ Visualization displayed (close window to continue)")
        except Exception as e:
            print(f"   ⚠️  Could not show visualization: {e}")

    # 5. Save sample output
    output_dir = Path("results/sign_language_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{Path(video_path).stem}_landmarks.npz"
    print(f"\n💾 Saving landmarks to: {output_file}")
    np.savez_compressed(
        output_file,
        landmarks_3d=landmarks_3d,
        landmarks_2d=landmarks_2d,
        video_name=os.path.basename(video_path),
        n_frames=len(video),
        duration=video.duration,
    )
    print("   ✓ Saved successfully")

    return {
        "landmarks_3d": landmarks_3d,
        "landmarks_2d": landmarks_2d,
        "n_frames": len(video),
        "duration": video.duration,
    }


def test_dataset_sample(
    csv_path: str = "data/processed/golden_label_sentiment.csv",
    video_dirs: list = None,
    n_samples: int = 5,
):
    """
    Testa l'estrazione su un campione del dataset completo.

    Args:
        csv_path: Percorso al CSV con annotazioni
        video_dirs: Lista di directory dove cercare i video
        n_samples: Numero di video da processare
    """
    if not SLT_AVAILABLE:
        print("❌ Cannot proceed without sign-language-translator")
        return

    if video_dirs is None:
        video_dirs = [
            "data/raw/ASLLRP/batch_utterance_video_v3_1",
            "data/raw/WLASL/videos",
        ]

    print(f"\n{'='*60}")
    print(f"Testing on Dataset Sample")
    print(f"{'='*60}\n")

    # Load annotations
    if not os.path.exists(csv_path):
        print(f"❌ CSV file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"📊 Dataset info:")
    print(f"   - Total videos: {len(df)}")
    print(f"   - Columns: {df.columns.tolist()}")
    print(f"   - Emotions: {df['emotion'].value_counts().to_dict()}")

    # Sample videos
    sample_df = df.sample(min(n_samples, len(df)), random_state=42)
    print(f"\n🎯 Processing {len(sample_df)} sample videos...\n")

    results = []
    for idx, row in sample_df.iterrows():
        video_name = row["video_name"]
        caption = row["caption"]
        emotion = row["emotion"]

        # Find video file
        video_path = None
        for vdir in video_dirs:
            candidate = os.path.join(vdir, video_name)
            if os.path.exists(candidate):
                video_path = candidate
                break

        if video_path is None:
            print(f"⚠️  Video not found: {video_name}")
            continue

        print(f"\n{'─'*60}")
        print(f"Video: {video_name}")
        print(f"Caption: {caption[:80]}{'...' if len(caption) > 80 else ''}")
        print(f"Emotion: {emotion}")
        print(f"{'─'*60}")

        # Extract landmarks (without visualization for batch processing)
        result = test_single_video(video_path, show_visualization=False)

        if result:
            result["video_name"] = video_name
            result["caption"] = caption
            result["emotion"] = emotion
            results.append(result)

    # Summary
    print(f"\n\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Successfully processed: {len(results)}/{len(sample_df)} videos")

    if results:
        avg_frames = np.mean([r["n_frames"] for r in results])
        avg_duration = np.mean([r["duration"] for r in results])
        print(f"📊 Average statistics:")
        print(f"   - Frames per video: {avg_frames:.1f}")
        print(f"   - Duration: {avg_duration:.2f} seconds")

    print(f"\n💡 Next steps:")
    print(f"   1. Verify landmark quality in results/sign_language_test/")
    print(f"   2. Implement tokenizer for captions")
    print(f"   3. Build Seq2Seq dataset loader")
    print(f"   4. Design Sign-to-Text model architecture")


def main():
    parser = argparse.ArgumentParser(
        description="Test sign language feature extraction pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch"],
        default="single",
        help="Test mode: single video or batch processing",
    )
    parser.add_argument(
        "--video_path", type=str, help="Path to video file (for single mode)"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="data/processed/golden_label_sentiment.csv",
        help="Path to annotations CSV (for batch mode)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of samples to process in batch mode",
    )
    parser.add_argument(
        "--no_viz",
        action="store_true",
        help="Disable visualization (faster processing)",
    )

    args = parser.parse_args()

    # Check if library is installed
    if not SLT_AVAILABLE:
        print("\n" + "=" * 60)
        print("⚠️  INSTALLATION REQUIRED")
        print("=" * 60)
        print("\nPlease install sign-language-translator first:")
        print("\n  pip install 'sign-language-translator[all]'\n")
        print("This will install:")
        print("  - MediaPipe for landmark extraction")
        print("  - PyAV for video processing")
        print("  - All other dependencies\n")
        return

    print("\n" + "=" * 60)
    print("🎬 Sign Language Feature Extraction Test")
    print("=" * 60)
    print(f"\nLibrary version: {slt.__version__}")
    print(f"Mode: {args.mode}")

    if args.mode == "single":
        if args.video_path is None:
            print("\n❌ Error: --video_path required for single mode")
            print("Example:")
            print("  python test_sign_language_extraction.py --mode single \\")
            print("    --video_path data/raw/ASLLRP/videos/83664512.mp4")
            return

        test_single_video(args.video_path, show_visualization=not args.no_viz)

    else:  # batch mode
        test_dataset_sample(csv_path=args.csv_path, n_samples=args.n_samples)

    print("\n✅ Test completed!\n")


if __name__ == "__main__":
    main()
