"""
How2Sign Dataset Loader
========================

PyTorch Dataset per How2Sign con landmarks OpenPose.

Differenze con SignLanguageDataset:
- Landmarks OpenPose (405 features) invece di MediaPipe (375)
- Carica JSON OpenPose direttamente da directory
- Supporta 31k+ samples

Usage:
    from src.sign_to_text.data import How2SignDataset

    train_dataset = How2SignDataset(
        split_csv='results/how2sign_splits/train_split.csv',
        openpose_dir='data/raw/train/openpose_output_train/json',
        tokenizer=tokenizer,
        max_frames=200
    )
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Dict, List

try:
    from .tokenizer import SignLanguageTokenizer
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).parent))
    from tokenizer import SignLanguageTokenizer


def load_openpose_landmarks(json_dir: Path) -> np.ndarray:
    """
    Carica landmarks OpenPose da directory JSON.

    OpenPose format per frame:
    - pose_keypoints_2d: 25 keypoints × 3 coords (x, y, conf) = 75 values
    - hand_left_keypoints_2d: 21 × 3 = 63 values
    - hand_right_keypoints_2d: 21 × 3 = 63 values
    - face_keypoints_2d: 70 × 3 = 210 values

    Total: 135 keypoints × 3 = 405 values per frame

    Args:
        json_dir: Directory con JSON (1 file per frame)

    Returns:
        landmarks: (n_frames, 405) numpy array
    """
    if not json_dir.exists():
        return None

    # Trova tutti i JSON ordinati
    json_files = sorted(json_dir.glob("*.json"))

    if len(json_files) == 0:
        return None

    frames_data = []

    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # OpenPose: {"people": [...]}
            if "people" not in data or len(data["people"]) == 0:
                # Nessuna persona -> frame vuoto
                frame_landmarks = np.zeros(405, dtype=np.float32)
            else:
                person = data["people"][0]  # Prima persona

                # Estrai keypoints (già flatten)
                pose = person.get("pose_keypoints_2d", [0.0] * 75)
                hand_left = person.get("hand_left_keypoints_2d", [0.0] * 63)
                hand_right = person.get("hand_right_keypoints_2d", [0.0] * 63)
                face = person.get("face_keypoints_2d", [0.0] * 210)

                # Concatena
                frame_landmarks = np.array(
                    pose + hand_left + hand_right + face, dtype=np.float32
                )

            frames_data.append(frame_landmarks)

        except Exception as e:
            # Errore -> frame vuoto
            frame_landmarks = np.zeros(405, dtype=np.float32)
            frames_data.append(frame_landmarks)

    # Stack frames
    landmarks = np.stack(frames_data, axis=0)  # (n_frames, 405)

    return landmarks


class How2SignDataset(Dataset):
    """
    Dataset per How2Sign con landmarks OpenPose.

    Compatibile con SignLanguageDataset ma usa OpenPose (405 features).
    """

    def __init__(
        self,
        split_csv: str,
        openpose_dir: str,
        tokenizer: SignLanguageTokenizer,
        max_frames: int = 200,
        max_caption_len: int = 50,  # How2Sign ha caption più lunghe
        normalize_landmarks: bool = True,
        landmark_features: int = 405,  # OpenPose default
    ):
        """
        Args:
            split_csv: CSV con colonne [video_name, caption, ...]
            openpose_dir: Directory base OpenPose JSON
            tokenizer: SignLanguageTokenizer instance
            max_frames: Max numero frames
            max_caption_len: Max lunghezza caption
            normalize_landmarks: Se normalizzare
            landmark_features: Numero feature landmarks (405 per OpenPose)
        """
        self.split_csv = split_csv
        self.openpose_dir = Path(openpose_dir)
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.max_caption_len = max_caption_len
        self.normalize_landmarks = normalize_landmarks
        self.landmark_features = landmark_features

        # Carica split
        print(f"\n📂 Loading How2Sign split: {split_csv}")
        self.df = pd.read_csv(split_csv)

        # Filtra NaN in caption E video_name
        self.df = self.df[~self.df["caption"].isna()].copy()
        self.df = self.df[~self.df["video_name"].isna()].copy()

        # Verifica landmarks disponibili
        print(f"   🔍 Verificando landmarks OpenPose...")
        self.valid_indices = []

        for idx, row in self.df.iterrows():
            video_name = str(row["video_name"])  # Converti a string per sicurezza
            landmark_dir = self.openpose_dir / video_name

            if landmark_dir.exists() and len(list(landmark_dir.glob("*.json"))) > 0:
                self.valid_indices.append(idx)

            # Progress ogni 1000
            if (idx + 1) % 1000 == 0:
                print(f"      Verificati {idx + 1}/{len(self.df)}...")

        self.df = self.df.loc[self.valid_indices].reset_index(drop=True)

        print(f"   ✓ Samples validi: {len(self.df)}")
        print(f"   ✓ Landmark features: {landmark_features} (OpenPose)")
        print(f"   ✓ Max frames: {max_frames}")
        print(f"   ✓ Max caption len: {max_caption_len}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Ritorna sample compatibile con SignLanguageDataset.

        Returns:
            {
                'landmarks': (max_frames, 405) tensor
                'landmarks_mask': (max_frames,) bool
                'caption_ids': (max_caption_len,) long
                'caption_mask': (max_caption_len,) bool
                'caption_text': str
                'video_name': str
                'n_frames_original': int
            }
        """
        row = self.df.iloc[idx]
        video_name = row["video_name"]
        caption_text = row["caption"]

        # 1. Carica landmarks OpenPose
        landmark_dir = self.openpose_dir / video_name
        landmarks = load_openpose_landmarks(landmark_dir)

        if landmarks is None:
            # Fallback: landmarks vuoti
            landmarks = np.zeros((1, self.landmark_features), dtype=np.float32)

        n_frames_original = landmarks.shape[0]

        # 2. Padding/Truncate frames
        if n_frames_original > self.max_frames:
            # Truncate
            landmarks = landmarks[: self.max_frames]
            landmarks_mask = torch.ones(self.max_frames, dtype=torch.bool)
        else:
            # Pad
            pad_frames = self.max_frames - n_frames_original
            landmarks = np.pad(
                landmarks, ((0, pad_frames), (0, 0)), mode="constant", constant_values=0
            )
            landmarks_mask = torch.cat(
                [
                    torch.ones(n_frames_original, dtype=torch.bool),
                    torch.zeros(pad_frames, dtype=torch.bool),
                ]
            )

        # 3. Normalizza
        if self.normalize_landmarks:
            valid_landmarks = landmarks[:n_frames_original]
            mean = valid_landmarks.mean()
            std = valid_landmarks.std()
            if std > 0:
                landmarks[:n_frames_original] = (valid_landmarks - mean) / std

        landmarks = torch.from_numpy(landmarks).float()

        # 4. Tokenizza caption
        caption_ids = self.tokenizer.encode(
            caption_text,
            add_special_tokens=True,
            max_length=self.max_caption_len,
            padding=True,
        )

        caption_mask = torch.tensor(
            [1 if id != self.tokenizer.pad_token_id else 0 for id in caption_ids],
            dtype=torch.bool,
        )

        caption_ids = torch.tensor(caption_ids, dtype=torch.long)

        return {
            "landmarks": landmarks,
            "landmarks_mask": landmarks_mask,
            "caption_ids": caption_ids,
            "caption_mask": caption_mask,
            "caption_text": caption_text,
            "video_name": video_name,
            "n_frames_original": n_frames_original,
        }


# Test
if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"🧪 TESTING HOW2SIGN DATASET")
    print(f"{'='*80}")

    # Carica tokenizer
    from tokenizer import SignLanguageTokenizer

    tokenizer = SignLanguageTokenizer.load("models/sign_to_text/tokenizer.json")

    # Test dataset
    dataset = How2SignDataset(
        split_csv="results/how2sign_splits/train_split.csv",
        openpose_dir="data/raw/train/openpose_output_train/json",
        tokenizer=tokenizer,
        max_frames=200,
        max_caption_len=50,
    )

    print(f"\n📦 Test sample...")
    sample = dataset[0]

    print(f"   Keys: {list(sample.keys())}")
    print(f"   Landmarks: {sample['landmarks'].shape}")
    print(f"   Video: {sample['video_name']}")
    print(f"   Caption: '{sample['caption_text'][:80]}...'")
    print(f"   Frames original: {sample['n_frames_original']}")

    print(f"\n✅ How2Sign dataset OK!")
