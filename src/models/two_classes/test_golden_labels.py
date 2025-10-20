# =================================================================================================
# DATASET CLASS FOR GOLDEN LABELS
# =================================================================================================
#
# This module defines the GoldenLabelDataset class for loading and processing golden label data.
# It's based on the LandmarkDataset but specifically designed for golden labels testing.
#
# =================================================================================================

import torch
from torch.utils.data import Dataset
import json
import os
import numpy as np
import pandas as pd
import logging

# Configura il logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Setup paths
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)


class GoldenLabelDataset(Dataset):
    """
    Dataset class for golden labels.
    Similar to LandmarkDataset but specifically designed for golden label data processing.
    """

    def __init__(
        self,
        landmarks_dir,
        processed_file,
        max_seq_len=100,
        num_features=50,
    ):
        """
        Args:
            landmarks_dir (str): Directory containing the landmark JSON files
            processed_file (str): Path to the CSV file containing metadata
            max_seq_len (int): Maximum sequence length for standardization
            num_features (int): Fixed number of features per frame
        """
        self.landmarks_dir = landmarks_dir
        self.processed = pd.read_csv(processed_file)
        self.max_seq_len = max_seq_len
        self.num_features = num_features

        # Create label mapping
        self.labels = sorted(self.processed["emotion"].unique())
        self.label2id = {label: i for i, label in enumerate(self.labels)}

        print(f"GoldenLabelDataset initialized:")
        print(f"  Landmarks dir: {landmarks_dir}")
        print(f"  Processed file: {processed_file}")
        print(f"  Number of samples: {len(self.processed)}")
        print(f"  Labels: {self.labels}")
        print(f"  Max sequence length: {max_seq_len}")
        print(f"  Number of features: {num_features}")

    def __len__(self):
        return len(self.processed)

    def __getitem__(self, idx):
        if idx >= len(self.processed):
            raise IndexError(
                f"Index {idx} out of range for dataset with {len(self.processed)} samples"
            )

        # Get video info
        video_info = self.processed.iloc[idx]
        video_name = str(video_info["video_name"]).replace(".mp4", "")
        label_str = video_info["emotion"]
        label = self.label2id[label_str]

        # Find video directory
        video_dir = os.path.join(self.landmarks_dir, video_name)
        if not os.path.isdir(video_dir):
            raise FileNotFoundError(
                f"Directory not found for video {video_name}: {video_dir}"
            )

        # Get sorted JSON files
        try:
            json_files = sorted(
                [f for f in os.listdir(video_dir) if f.endswith(".json")]
            )
        except FileNotFoundError:
            raise FileNotFoundError(f"Error reading directory: {video_dir}")

        if not json_files:
            raise ValueError(f"No JSON files found for video {video_name}")

        # Extract features from JSON files
        sequence = []
        for json_file in json_files:
            frame_path = os.path.join(video_dir, json_file)
            try:
                with open(frame_path, "r") as f:
                    frame_data = json.load(f)

                # Extract features using the method that can be overridden
                features = self.extract_features_from_json(frame_data)
                if features is not None:
                    sequence.append(features)

            except (json.JSONDecodeError, Exception) as e:
                logging.warning(f"Error processing {frame_path}: {e}")
                continue

        if not sequence:
            raise ValueError(f"No valid features found for video {video_name}")

        # Convert to numpy array
        sequence = np.array(sequence, dtype=np.float32)

        # Standardize sequence length
        if len(sequence) > self.max_seq_len:
            sequence = sequence[: self.max_seq_len]
        else:
            padding = np.zeros(
                (self.max_seq_len - len(sequence), self.num_features), dtype=np.float32
            )
            sequence = np.vstack((sequence, padding))

        return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.long)

    def extract_features_from_json(self, frame_json):
        """
        Extract features from a single frame JSON.
        This method can be overridden in subclasses for different feature extraction logic.

        Args:
            frame_json (dict): JSON data for a single frame

        Returns:
            np.ndarray: Feature vector for the frame
        """
        people = frame_json.get("people", [])
        if not people:
            return np.zeros(self.num_features, dtype=np.float32)

        # Get first person
        person_data = people[0]
        keypoints = person_data.get("pose_keypoints_2d", [])

        if not keypoints:
            return np.zeros(self.num_features, dtype=np.float32)

        # Extract x, y coordinates (remove confidence/z values)
        flat_landmarks = [coord for i, coord in enumerate(keypoints) if i % 3 != 2]

        # Convert to numpy array
        landmarks_array = np.array(flat_landmarks, dtype=np.float32)

        # Standardize to required number of features
        if len(landmarks_array) > self.num_features:
            landmarks_array = landmarks_array[: self.num_features]
        elif len(landmarks_array) < self.num_features:
            padding = np.zeros(self.num_features - len(landmarks_array))
            landmarks_array = np.hstack([landmarks_array, padding])

        return landmarks_array


def test_golden_dataset():
    """
    Test function for the GoldenLabelDataset
    """
    print("=" * 60)
    print("TESTING GOLDEN LABEL DATASET")
    print("=" * 60)

    landmarks_dir = os.path.join(
        BASE_DIR, "data", "raw", "ASLLRP", "mediapipe_output_golden_label", "json"
    )
    processed_file = os.path.join(
        BASE_DIR, "data", "processed", "golden_label_sentiment.csv"
    )

    print(f"Landmarks dir: {landmarks_dir}")
    print(f"Processed file: {processed_file}")

    if not os.path.exists(landmarks_dir):
        print(f"ERROR: Landmarks directory not found: {landmarks_dir}")
        return

    if not os.path.exists(processed_file):
        print(f"ERROR: Processed file not found: {processed_file}")
        return

    try:
        dataset = GoldenLabelDataset(landmarks_dir, processed_file)
        print(f"Dataset loaded successfully with {len(dataset)} samples")

        if len(dataset) > 0:
            # Test first sample
            features, label = dataset[0]
            print(f"First sample:")
            print(f"  Features shape: {features.shape}")
            print(f"  Label: {label} ({dataset.labels[label]})")
            print(f"  Feature range: [{features.min():.6f}, {features.max():.6f}]")
            print(f"  Feature mean: {features.mean():.6f}")
            print(f"  Feature std: {features.std():.6f}")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    test_golden_dataset()
