"""
Extract SignHiera Features from How2Sign Videos
================================================

Extract visual features from How2Sign videos using pretrained SignHiera model.
These features will be used for fine-tuning the SONAR translation model.

Usage:
    # Single split
    python extract_features_signhiera.py \\
        --manifest data/processed/how2sign_ssvp/manifest/train.tsv \\
        --video_dir data/processed/how2sign_ssvp/clips/train \\
        --model_path models/pretrained_ssvp/dm_70h_ub_signhiera.pth \\
        --output_dir data/processed/how2sign_ssvp/features/train \\
        --batch_size 8
"""

import argparse
import csv
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


def check_requirements():
    """Check if required packages are installed."""
    required_packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "cv2": "OpenCV (cv2)",
        "PIL": "Pillow",
    }

    missing = []
    for package, name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing.append(name)

    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install torch torchvision opencv-python pillow")
        sys.exit(1)


def load_signhiera_model(model_path: Path, device: str):
    """
    Load SignHiera model from checkpoint.

    Args:
        model_path: Path to SignHiera checkpoint
        device: Device to load model on

    Returns:
        SignHiera model in eval mode
    """
    print(f"\n📦 Loading SignHiera model...")
    print(f"   Path: {model_path}")
    print(f"   Device: {device}")

    # Add SSVP-SLT repo to path
    ssvp_repo = Path(__file__).parent / "models" / "ssvp_slt_repo" / "src"
    if not ssvp_repo.exists():
        print(f"\n❌ SSVP-SLT repo not found at: {ssvp_repo}")
        print("   Clone it first:")
        print("   cd src/sign_to_text_ssvp/models")
        print("   git clone https://github.com/facebookresearch/ssvp_slt.git ssvp_slt_repo")
        sys.exit(1)

    sys.path.insert(0, str(ssvp_repo))

    try:
        from ssvp_slt.modeling.sign_hiera import SignHiera
    except ImportError as e:
        print(f"\n❌ Failed to import SignHiera: {e}")
        print("   Make sure SSVP-SLT repo is cloned correctly")
        sys.exit(1)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Extract model state dict (handle different checkpoint formats)
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    # Initialize model
    model = SignHiera(
        embed_dim=768,
        num_heads=12,
        depth=24,
        mlp_ratio=4.0,
    )

    # Load weights
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    print("   ✓ Model loaded successfully")

    return model


def load_video(video_path: Path, target_fps: int = 25) -> Optional[torch.Tensor]:
    """
    Load video and extract frames.

    Args:
        video_path: Path to video file
        target_fps: Target FPS for extraction

    Returns:
        Video tensor of shape (T, C, H, W) or None if failed
    """
    import cv2

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(fps / target_fps))

        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize to 224x224
                frame = cv2.resize(frame, (224, 224))
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0

                # ImageNet Normalization (Standard per SignHiera/ViT)
                mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
                std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
                frame = (frame - mean) / std

                frames.append(frame)

            frame_idx += 1

        cap.release()

        if len(frames) == 0:
            return None

        # Stack frames: (T, H, W, C) -> (T, C, H, W)
        video = np.stack(frames, axis=0)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)

        return video

    except Exception as e:
        print(f"\n⚠️  Failed to load video {video_path.name}: {e}")
        return None


def extract_features(
    model,
    video: torch.Tensor,
    device: str,
) -> Optional[np.ndarray]:
    """
    Extract features from video using SignHiera.

    Args:
        model: SignHiera model
        video: Video tensor (T, C, H, W)
        device: Device to run on

    Returns:
        Features array of shape (T, D) or None if failed
    """
    try:
        # Add batch dimension: (T, C, H, W) -> (1, T, C, H, W)
        video = video.unsqueeze(0).to(device)

        with torch.no_grad():
            features = model(video)

        # Remove batch dimension and convert to numpy
        features = features.squeeze(0).cpu().numpy()

        return features

    except Exception as e:
        print(f"\n⚠️  Feature extraction failed: {e}")
        return None


def process_manifest(
    manifest_path: Path,
    video_dir: Path,
    model_path: Path,
    output_dir: Path,
    batch_size: int = 8,
    device: str = "cuda",
    max_samples: Optional[int] = None,
):
    """
    Process all videos in manifest and extract features.

    Args:
        manifest_path: Path to TSV manifest file
        video_dir: Directory containing videos
        model_path: Path to SignHiera checkpoint
        output_dir: Directory to save features
        batch_size: Batch size (currently processes one at a time)
        device: Device to run on
        max_samples: Maximum number of samples to process
    """
    print("\n" + "=" * 80)
    print("🎬 SIGNHIERA FEATURE EXTRACTION")
    print("=" * 80)

    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        print("\n⚠️  CUDA not available, using CPU")
        device = "cpu"

    # Load model
    model = load_signhiera_model(model_path, device)

    # Read manifest
    print(f"\n📋 Reading manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        samples = list(reader)

    if max_samples:
        samples = samples[:max_samples]
        print(f"   ⚠️  Limited to {max_samples} samples for testing")

    print(f"   ✓ Found {len(samples)} samples")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📂 Output directory: {output_dir}")

    # Process videos
    print(f"\n🎥 Processing videos...")
    success_count = 0
    failed_count = 0
    failed_videos = []

    for sample in tqdm(samples, desc="   Extracting features"):
        video_id = sample["id"]
        video_name = f"{video_id}.mp4"
        video_path = video_dir / video_name

        # Check if video exists
        if not video_path.exists():
            # Try without .mp4 extension
            video_path = video_dir / video_id
            if not video_path.exists():
                failed_count += 1
                failed_videos.append(video_id)
                continue

        # Check if features already exist
        feature_path = output_dir / f"{video_id}.npy"
        if feature_path.exists():
            success_count += 1
            continue

        # Load video
        video = load_video(video_path)
        if video is None:
            failed_count += 1
            failed_videos.append(video_id)
            continue

        # Extract features
        features = extract_features(model, video, device)
        if features is None:
            failed_count += 1
            failed_videos.append(video_id)
            continue

        # Save features
        np.save(feature_path, features)
        success_count += 1

    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failed:  {failed_count}")

    if failed_videos:
        print(f"\n   Failed videos (first 10):")
        for video in failed_videos[:10]:
            print(f"      - {video}")
        if len(failed_videos) > 10:
            print(f"      ... and {len(failed_videos) - 10} more")

    print(f"\n✅ Features saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract SignHiera features from How2Sign videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to TSV manifest file (e.g., manifest/train.tsv)",
    )

    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing videos (e.g., clips/train)",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to SignHiera checkpoint (.pth)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for features",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on (default: cuda)",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )

    args = parser.parse_args()

    # Check requirements
    check_requirements()

    # Convert to Path objects
    manifest_path = Path(args.manifest)
    video_dir = Path(args.video_dir)
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)

    # Validate inputs
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        return 1

    if not video_dir.exists():
        print(f"❌ Video directory not found: {video_dir}")
        return 1

    if not model_path.exists():
        print(f"❌ Model checkpoint not found: {model_path}")
        return 1

    # Process
    process_manifest(
        manifest_path=manifest_path,
        video_dir=video_dir,
        model_path=model_path,
        output_dir=output_dir,
        batch_size=args.batch_size,
        device=args.device,
        max_samples=args.max_samples,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
