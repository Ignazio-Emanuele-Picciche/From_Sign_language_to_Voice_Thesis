"""
Sign Language Dataset
======================

PyTorch Dataset per Sign-to-Text translation.
Carica landmarks MediaPipe + caption tokenizzate.

Usage:
    from src.sign_to_text.data import SignLanguageDataset, SignLanguageTokenizer

    tokenizer = SignLanguageTokenizer.load('models/sign_to_text/tokenizer.json')

    train_dataset = SignLanguageDataset(
        split_csv='results/utterances_analysis/train_split.csv',
        landmarks_dir='data/processed/sign_language_landmarks',
        tokenizer=tokenizer,
        max_frames=200,
        max_caption_len=30
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                               collate_fn=sign_language_collate_fn)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Dict, List

try:
    from .tokenizer import SignLanguageTokenizer
except ImportError:
    # Fallback per esecuzione diretta
    import sys

    sys.path.append(str(Path(__file__).parent))
    from tokenizer import SignLanguageTokenizer


class SignLanguageDataset(Dataset):
    """
    Dataset per Sign Language Translation.

    Carica:
    - Landmarks MediaPipe (n_frames, 375) da .npz
    - Caption tokenizzate

    Gestisce:
    - Padding variabile per frames
    - Padding variabile per caption
    - Maschere per attention
    """

    def __init__(
        self,
        split_csv: str,
        landmarks_dir: str,
        tokenizer: SignLanguageTokenizer,
        max_frames: int = 200,
        max_caption_len: int = 30,
        normalize_landmarks: bool = True,
    ):
        """
        Args:
            split_csv: Path al CSV split (train/val/test)
            landmarks_dir: Directory con landmarks .npz
            tokenizer: SignLanguageTokenizer instance
            max_frames: Max numero frames (padding/truncate)
            max_caption_len: Max lunghezza caption (padding/truncate)
            normalize_landmarks: Se normalizzare landmarks (mean=0, std=1)
        """
        self.split_csv = split_csv
        self.landmarks_dir = Path(landmarks_dir)
        self.tokenizer = tokenizer
        self.max_frames = max_frames
        self.max_caption_len = max_caption_len
        self.normalize_landmarks = normalize_landmarks

        # Carica split
        print(f"\n📂 Loading dataset split: {split_csv}")
        self.df = pd.read_csv(split_csv)
        self.df = self.df[~self.df["caption"].isna()]  # Rimuovi NaN

        # Verifica landmarks disponibili
        self.valid_indices = []
        for idx, row in self.df.iterrows():
            video_name = row["video_name"]
            landmark_path = (
                self.landmarks_dir / f"{Path(video_name).stem}_landmarks.npz"
            )
            if landmark_path.exists():
                self.valid_indices.append(idx)

        self.df = self.df.loc[self.valid_indices].reset_index(drop=True)

        print(f"   ✓ Samples: {len(self.df)}")
        print(f"   ✓ Max frames: {max_frames}")
        print(f"   ✓ Max caption len: {max_caption_len}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Ritorna un sample.

        Returns:
            {
                'landmarks': (max_frames, 375) float tensor
                'landmarks_mask': (max_frames,) bool tensor (True = valido)
                'caption_ids': (max_caption_len,) long tensor
                'caption_mask': (max_caption_len,) bool tensor
                'caption_text': str (original)
                'video_name': str
            }
        """
        row = self.df.iloc[idx]
        video_name = row["video_name"]
        caption_text = row["caption"]

        # 1. Carica landmarks
        landmark_path = self.landmarks_dir / f"{Path(video_name).stem}_landmarks.npz"
        data = np.load(landmark_path, allow_pickle=True)
        landmarks = data["landmarks_3d"]  # (n_frames, 375)

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

        # 3. Normalizza landmarks (opzionale)
        if self.normalize_landmarks:
            # Normalizza solo i frames validi
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

        # Crea maschera caption
        caption_mask = torch.tensor(
            [1 if id != self.tokenizer.pad_token_id else 0 for id in caption_ids],
            dtype=torch.bool,
        )

        caption_ids = torch.tensor(caption_ids, dtype=torch.long)

        return {
            "landmarks": landmarks,  # (max_frames, 375)
            "landmarks_mask": landmarks_mask,  # (max_frames,)
            "caption_ids": caption_ids,  # (max_caption_len,)
            "caption_mask": caption_mask,  # (max_caption_len,)
            "caption_text": caption_text,  # str
            "video_name": video_name,  # str
            "n_frames_original": n_frames_original,  # int
        }


def sign_language_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function per DataLoader.

    Combina batch di sample in tensori.

    Args:
        batch: Lista di dict da __getitem__

    Returns:
        Batch dict con tensori (batch_size, ...)
    """
    landmarks = torch.stack([item["landmarks"] for item in batch])
    landmarks_mask = torch.stack([item["landmarks_mask"] for item in batch])
    caption_ids = torch.stack([item["caption_ids"] for item in batch])
    caption_mask = torch.stack([item["caption_mask"] for item in batch])

    # Metadata (liste)
    caption_texts = [item["caption_text"] for item in batch]
    video_names = [item["video_name"] for item in batch]
    n_frames_original = torch.tensor([item["n_frames_original"] for item in batch])

    return {
        "landmarks": landmarks,  # (B, max_frames, 375)
        "landmarks_mask": landmarks_mask,  # (B, max_frames)
        "caption_ids": caption_ids,  # (B, max_caption_len)
        "caption_mask": caption_mask,  # (B, max_caption_len)
        "caption_texts": caption_texts,  # List[str]
        "video_names": video_names,  # List[str]
        "n_frames_original": n_frames_original,  # (B,)
    }


def get_dataloaders(
    train_csv: str = "results/utterances_analysis/train_split.csv",
    val_csv: str = "results/utterances_analysis/val_split.csv",
    test_csv: str = "results/utterances_analysis/test_split.csv",
    landmarks_dir: str = "data/processed/sign_language_landmarks",
    tokenizer_path: str = "models/sign_to_text/tokenizer.json",
    batch_size: int = 16,
    max_frames: int = 200,
    max_caption_len: int = 30,
    num_workers: int = 4,
) -> Dict[str, DataLoader]:
    """
    Utility per creare train/val/test dataloaders.

    Args:
        train_csv, val_csv, test_csv: Path ai CSV splits
        landmarks_dir: Directory landmarks
        tokenizer_path: Path tokenizer
        batch_size: Batch size
        max_frames: Max frames
        max_caption_len: Max caption length
        num_workers: Num workers per DataLoader

    Returns:
        {
            'train': DataLoader,
            'val': DataLoader,
            'test': DataLoader
        }
    """
    print(f"\n{'='*80}")
    print(f"📊 CREATING DATALOADERS")
    print(f"{'='*80}")

    # Carica tokenizer
    print(f"\n🔧 Loading tokenizer from {tokenizer_path}")
    tokenizer = SignLanguageTokenizer.load(tokenizer_path)

    # Crea datasets
    train_dataset = SignLanguageDataset(
        split_csv=train_csv,
        landmarks_dir=landmarks_dir,
        tokenizer=tokenizer,
        max_frames=max_frames,
        max_caption_len=max_caption_len,
    )

    val_dataset = SignLanguageDataset(
        split_csv=val_csv,
        landmarks_dir=landmarks_dir,
        tokenizer=tokenizer,
        max_frames=max_frames,
        max_caption_len=max_caption_len,
    )

    test_dataset = SignLanguageDataset(
        split_csv=test_csv,
        landmarks_dir=landmarks_dir,
        tokenizer=tokenizer,
        max_frames=max_frames,
        max_caption_len=max_caption_len,
    )

    # Crea dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sign_language_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sign_language_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sign_language_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"\n✅ Dataloaders created:")
    print(f"   Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"   Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"   Test:  {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"   Batch size: {batch_size}")

    return {"train": train_loader, "val": val_loader, "test": test_loader}


# Test script
if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"🧪 TESTING SIGN LANGUAGE DATASET")
    print(f"{'='*80}")

    # Carica tokenizer
    tokenizer = SignLanguageTokenizer.load("models/sign_to_text/tokenizer.json")

    # Crea dataset
    train_dataset = SignLanguageDataset(
        split_csv="results/utterances_analysis/train_split.csv",
        landmarks_dir="data/processed/sign_language_landmarks",
        tokenizer=tokenizer,
        max_frames=200,
        max_caption_len=30,
    )

    print(f"\n📦 Test sample loading...")
    sample = train_dataset[0]

    print(f"\n   Sample keys: {list(sample.keys())}")
    print(f"   Landmarks shape: {sample['landmarks'].shape}")
    print(f"   Landmarks mask shape: {sample['landmarks_mask'].shape}")
    print(f"   Caption IDs shape: {sample['caption_ids'].shape}")
    print(f"   Caption mask shape: {sample['caption_mask'].shape}")
    print(f"   Video: {sample['video_name']}")
    print(f"   Caption: '{sample['caption_text']}'")
    print(f"   Caption IDs: {sample['caption_ids'][:15].tolist()}...")
    print(f"   Decoded: '{tokenizer.decode(sample['caption_ids'].tolist())}'")

    # Test DataLoader
    print(f"\n🔄 Test DataLoader (batch=4)...")
    loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=sign_language_collate_fn
    )

    batch = next(iter(loader))
    print(f"\n   Batch keys: {list(batch.keys())}")
    print(f"   Landmarks: {batch['landmarks'].shape}")
    print(f"   Caption IDs: {batch['caption_ids'].shape}")
    print(f"   Caption texts: {len(batch['caption_texts'])} captions")
    print(f"\n   Batch captions:")
    for i, text in enumerate(batch["caption_texts"]):
        print(f"      [{i}] {text}")

    print(f"\n{'='*80}")
    print(f"✅ DATASET TEST PASSED!")
    print(f"{'='*80}")
    print(f"\n🚀 Next: Implementa Seq2Seq Transformer model")
    print(f"\n")
