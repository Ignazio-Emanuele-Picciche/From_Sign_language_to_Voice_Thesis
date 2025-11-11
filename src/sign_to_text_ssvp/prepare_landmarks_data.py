"""
Prepare Landmarks Data for Sign-to-Text Translation
====================================================

Convert OpenPose JSON landmarks to training format for Transformer model.

Usage:
    python prepare_landmarks_data.py \\
        --openpose_dir ../../data/raw/train/openpose_output_train/json \\
        --manifest ../../data/processed/how2sign_ssvp/manifest/train.tsv \\
        --output_dir ../../data/processed/landmarks_how2sign/train \\
        --split train
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")


def load_openpose_landmarks(json_dir: Path, video_id: str) -> Optional[np.ndarray]:
    """
    Load OpenPose landmarks for a video.
    
    Args:
        json_dir: Directory containing OpenPose JSON files
        video_id: Video ID (e.g., '--7E2sU6zP4_10-5-rgb_front')
    
    Returns:
        Array of shape (num_frames, 137, 2) or None if not found
        - 137 keypoints: body(25) + hands(42) + face(70)
        - 2 coordinates: (x, y)
    """
    video_folder = json_dir / video_id
    
    if not video_folder.exists():
        return None
    
    # Get all JSON files sorted by frame number
    json_files = sorted(video_folder.glob("*.json"))
    
    if len(json_files) == 0:
        return None
    
    landmarks_sequence = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract person 0 (main signer)
            if not data.get('people') or len(data['people']) == 0:
                # No person detected, use zeros
                frame_landmarks = np.zeros((137, 2))
            else:
                person = data['people'][0]
                
                # Body keypoints (25 × 3, but we take only x, y)
                body = np.array(person.get('pose_keypoints_2d', [])).reshape(-1, 3)[:, :2]
                if body.shape[0] != 25:
                    body = np.zeros((25, 2))
                
                # Left hand keypoints (21 × 3)
                left_hand = np.array(person.get('hand_left_keypoints_2d', [])).reshape(-1, 3)[:, :2]
                if left_hand.shape[0] != 21:
                    left_hand = np.zeros((21, 2))
                
                # Right hand keypoints (21 × 3)
                right_hand = np.array(person.get('hand_right_keypoints_2d', [])).reshape(-1, 3)[:, :2]
                if right_hand.shape[0] != 21:
                    right_hand = np.zeros((21, 2))
                
                # Face keypoints (70 × 3)
                face = np.array(person.get('face_keypoints_2d', [])).reshape(-1, 3)[:, :2]
                if face.shape[0] != 70:
                    face = np.zeros((70, 2))
                
                # Concatenate all keypoints
                frame_landmarks = np.vstack([body, left_hand, right_hand, face])
            
            landmarks_sequence.append(frame_landmarks)
        
        except Exception as e:
            print(f"⚠️  Error loading {json_file.name}: {e}")
            continue
    
    if len(landmarks_sequence) == 0:
        return None
    
    # Stack frames: (num_frames, 137, 2)
    return np.stack(landmarks_sequence)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks to [0, 1] based on frame dimensions.
    
    Args:
        landmarks: Array of shape (num_frames, 137, 2)
    
    Returns:
        Normalized landmarks
    """
    # Assume video resolution (can be adjusted)
    width, height = 1920, 1080
    
    normalized = landmarks.copy()
    normalized[:, :, 0] /= width   # Normalize x
    normalized[:, :, 1] /= height  # Normalize y
    
    # Clip to [0, 1]
    normalized = np.clip(normalized, 0, 1)
    
    return normalized


def prepare_dataset(
    openpose_dir: Path,
    manifest_file: Path,
    output_dir: Path,
    split: str,
    normalize: bool = True,
) -> Tuple[int, int]:
    """
    Prepare landmarks dataset from OpenPose output and manifest.
    
    Args:
        openpose_dir: Directory containing OpenPose JSON files
        manifest_file: TSV manifest with video IDs and captions
        output_dir: Output directory for processed data
        split: Split name (train/val/test)
        normalize: Whether to normalize landmarks
    
    Returns:
        Tuple of (success_count, failed_count)
    """
    print(f"\n{'=' * 80}")
    print(f"📊 Preparing {split.upper()} dataset")
    print(f"{'=' * 80}")
    
    # Read manifest
    print(f"\n1️⃣  Reading manifest: {manifest_file}")
    df = pd.read_csv(manifest_file, sep='\t')
    print(f"   ✓ Found {len(df)} samples")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each video
    print(f"\n2️⃣  Processing landmarks...")
    landmarks_data = []
    success_count = 0
    failed_count = 0
    failed_videos = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"   Processing {split}"):
        video_id = row['id']
        caption = row['text']
        
        # Load landmarks
        landmarks = load_openpose_landmarks(openpose_dir, video_id)
        
        if landmarks is None:
            failed_count += 1
            failed_videos.append(video_id)
            continue
        
        # Normalize if requested
        if normalize:
            landmarks = normalize_landmarks(landmarks)
        
        # Save landmarks + caption
        sample = {
            'id': video_id,
            'landmarks': landmarks,  # (num_frames, 137, 2)
            'caption': caption,
            'num_frames': len(landmarks),
        }
        
        landmarks_data.append(sample)
        success_count += 1
    
    # Save processed data
    print(f"\n3️⃣  Saving processed data...")
    output_file = output_dir / f"{split}_landmarks.pkl"
    
    import pickle
    with open(output_file, 'wb') as f:
        pickle.dump(landmarks_data, f)
    
    print(f"   ✓ Saved {len(landmarks_data)} samples to {output_file}")
    
    # Save metadata
    metadata = {
        'split': split,
        'num_samples': len(landmarks_data),
        'num_failed': failed_count,
        'landmark_dim': 137,
        'coordinate_dim': 2,
        'normalized': normalize,
    }
    
    metadata_file = output_dir / f"{split}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"   ✓ Saved metadata to {metadata_file}")
    
    # Summary
    print(f"\n📊 {split.upper()} Summary:")
    print(f"   ✅ Success: {success_count}")
    print(f"   ❌ Failed:  {failed_count}")
    
    if failed_videos and failed_count <= 20:
        print(f"\n   Failed videos:")
        for video in failed_videos:
            print(f"      - {video}")
    elif failed_count > 20:
        print(f"\n   Failed videos (first 10):")
        for video in failed_videos[:10]:
            print(f"      - {video}")
        print(f"      ... and {failed_count - 10} more")
    
    return success_count, failed_count


def main():
    parser = argparse.ArgumentParser(
        description="Prepare landmarks data for sign-to-text translation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare train split
  python prepare_landmarks_data.py \\
      --openpose_dir ../../data/raw/train/openpose_output_train/json \\
      --manifest ../../data/processed/how2sign_ssvp/manifest/train.tsv \\
      --output_dir ../../data/processed/landmarks_how2sign/train \\
      --split train
  
  # Prepare val split
  python prepare_landmarks_data.py \\
      --openpose_dir ../../data/raw/val/openpose_output_val/json \\
      --manifest ../../data/processed/how2sign_ssvp/manifest/val.tsv \\
      --output_dir ../../data/processed/landmarks_how2sign/val \\
      --split val
        """,
    )
    
    parser.add_argument(
        '--openpose_dir',
        type=str,
        required=True,
        help='Directory containing OpenPose JSON files',
    )
    
    parser.add_argument(
        '--manifest',
        type=str,
        required=True,
        help='TSV manifest file with video IDs and captions',
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for processed landmarks data',
    )
    
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['train', 'val', 'test'],
        help='Dataset split name',
    )
    
    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Do not normalize landmarks (keep pixel coordinates)',
    )
    
    args = parser.parse_args()
    
    # Convert to Path objects
    openpose_dir = Path(args.openpose_dir)
    manifest_file = Path(args.manifest)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not openpose_dir.exists():
        print(f"❌ OpenPose directory not found: {openpose_dir}")
        return 1
    
    if not manifest_file.exists():
        print(f"❌ Manifest file not found: {manifest_file}")
        return 1
    
    print("=" * 80)
    print("🎬 LANDMARKS DATA PREPARATION")
    print("=" * 80)
    print(f"\n📂 Paths:")
    print(f"   OpenPose:  {openpose_dir}")
    print(f"   Manifest:  {manifest_file}")
    print(f"   Output:    {output_dir}")
    print(f"\n⚙️  Settings:")
    print(f"   Split:      {args.split}")
    print(f"   Normalize:  {not args.no_normalize}")
    
    # Prepare dataset
    success, failed = prepare_dataset(
        openpose_dir=openpose_dir,
        manifest_file=manifest_file,
        output_dir=output_dir,
        split=args.split,
        normalize=not args.no_normalize,
    )
    
    # Final summary
    print("\n" + "=" * 80)
    print("✅ PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"\n📊 Total: {success} samples prepared")
    
    if failed > 0:
        print(f"⚠️  {failed} samples failed (missing landmarks)")
    
    print(f"\n📍 Output:")
    print(f"   {output_dir}/{args.split}_landmarks.pkl")
    print(f"   {output_dir}/{args.split}_metadata.json")
    
    print(f"\n🚀 Next step:")
    print(f"   Train model with: python train_landmarks_to_text.py")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
