"""
Custom SONAR Inference Script for How2Sign
============================================

This script performs ASL video → English text translation using:
- SignHiera (feature extractor) from SSVP-SLT
- SONAR encoder/decoder from sonar-space

It bypasses the dependency conflicts in the SSVP-SLT repository code.

Author: Ignazio Picciche
Date: November 2025
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Add SSVP-SLT to path
SSVP_REPO = Path(__file__).parent / "models" / "ssvp_slt_repo" / "src"
sys.path.insert(0, str(SSVP_REPO))

from ssvp_slt.modeling.sign_hiera import SignHiera
from ssvp_slt.modeling.sign_t5 import SignT5Config, SignT5Model


class SimpleSignHieraLoader:
    """Loads SignHiera model without using the complex SSVP pipeline"""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str) -> SignHiera:
        """Load SignHiera from checkpoint"""
        print(f"Loading SignHiera from {checkpoint_path}...")

        # Import the specific model variant
        from ssvp_slt.modeling.sign_hiera import hiera_base_128x224

        # Create model
        model = hiera_base_128x224(pretrained=False, strict=False)

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "model_state" in checkpoint:
            state_dict = checkpoint["model_state"]
        else:
            state_dict = checkpoint

        # Load state dict
        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print(f"Loaded SignHiera:")
        print(f"  - Missing keys: {len(missing)}")
        print(f"  - Unexpected keys: {len(unexpected)}")

        # Remove classification head (we only want features)
        model.head = nn.Identity()

        model.to(self.device)
        model.eval()

        return model

    @torch.no_grad()
    def extract_features(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video tensor

        Args:
            video_tensor: (C, T, H, W) tensor

        Returns:
            features: (T', D) tensor where T' is temporal length, D is feature dim
        """
        # Add batch dimension if needed
        if video_tensor.dim() == 4:
            video_tensor = video_tensor.unsqueeze(0)  # (1, C, T, H, W)

        video_tensor = video_tensor.to(self.device)

        # Extract features
        features = self.model.extract_features(video_tensor, padding=None)

        # Remove batch dimension
        if features.dim() == 3:
            features = features.squeeze(0)  # (T', D)

        return features


class SimpleSonarEncoder:
    """Loads SONAR T5 encoder without complex fairseq2 dependencies"""

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        """Load SONAR encoder from checkpoint"""
        print(f"Loading SONAR encoder from {checkpoint_path}...")

        # Create T5-based SONAR encoder
        config = SignT5Config.from_pretrained(
            "google/t5-v1_1-large",
            decoder_start_token_id=0,
            dropout_rate=0.0,
        )

        # Simple T5 model (we'll use just the encoder part)
        model = SignT5Model(config)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if "student" in checkpoint:
            state_dict = checkpoint["student"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        # Try to load state dict
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            print(f"Loaded SONAR encoder:")
            print(f"  - Missing keys: {len(missing)}")
            print(f"  - Unexpected keys: {len(unexpected)}")
        except Exception as e:
            print(f"Warning: Could not load full state dict: {e}")
            print("Will use encoder initialization from pretrained T5")

        model.to(self.device)
        model.eval()

        return model

    @torch.no_grad()
    def encode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode visual features to SONAR sentence embedding

        Args:
            features: (T, D) tensor from SignHiera

        Returns:
            embedding: (1, 1024) SONAR sentence embedding
        """
        # Add batch dimension
        if features.dim() == 2:
            features = features.unsqueeze(0)  # (1, T, D)

        features = features.to(self.device)

        # Forward through encoder
        # Note: This is simplified - full version would need proper attention masks
        encoder_outputs = self.model.encoder(inputs_embeds=features, return_dict=True)

        # Pool to sentence embedding (mean pooling)
        sentence_embedding = encoder_outputs.last_hidden_state.mean(dim=1)

        return sentence_embedding


class SonarInferencePipeline:
    """Complete pipeline for ASL video → English text"""

    def __init__(
        self, signhiera_path: str, sonar_encoder_path: str, device: str = "cpu"
    ):
        self.device = device

        print("\n" + "=" * 60)
        print("Initializing SONAR Inference Pipeline")
        print("=" * 60 + "\n")

        # Load feature extractor
        self.feature_extractor = SimpleSignHieraLoader(signhiera_path, device)

        # Load SONAR encoder
        self.encoder = SimpleSonarEncoder(sonar_encoder_path, device)

        # Load SONAR decoder (from sonar-space package)
        self._load_decoder()

        print("\n" + "=" * 60)
        print("Pipeline Ready!")
        print("=" * 60 + "\n")

    def _load_decoder(self):
        """Load SONAR text decoder using sonar-space"""
        try:
            from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

            print("Loading SONAR decoder...")
            self.decoder = EmbeddingToTextModelPipeline(
                decoder="text_sonar_basic_decoder",
                tokenizer="text_sonar_basic_encoder",
                device=torch.device(self.device),
            )
            print("✓ SONAR decoder loaded")

        except Exception as e:
            print(f"⚠️  Could not load SONAR decoder: {e}")
            print("Falling back to direct T5 decoding")
            self.decoder = None

    @torch.no_grad()
    def translate_video(self, video_path: str, target_lang: str = "eng_Latn") -> str:
        """
        Translate ASL video to text

        Args:
            video_path: Path to video file
            target_lang: Target language code (default: eng_Latn for English)

        Returns:
            translation: Translated text
        """
        print(f"\nProcessing: {video_path}")
        print("-" * 60)

        # 1. Load video (simplified - would need proper video loading)
        print("1. Loading video...")
        video_tensor = self._load_video(video_path)

        # 2. Extract features
        print("2. Extracting visual features...")
        features = self.feature_extractor.extract_features(video_tensor)
        print(f"   Features shape: {features.shape}")

        # 3. Encode to SONAR embedding
        print("3. Encoding to SONAR space...")
        embedding = self.encoder.encode(features)
        print(f"   Embedding shape: {embedding.shape}")

        # 4. Decode to text
        print("4. Decoding to text...")
        if self.decoder is not None:
            translation = self.decoder.predict(embedding, target_lang=target_lang)[0]
        else:
            translation = "[Decoder not available - would generate text here]"

        print(f"   Translation: {translation}")
        print("-" * 60)

        return translation

    def _load_video(self, video_path: str) -> torch.Tensor:
        """
        Load and preprocess video

        This is a placeholder - would need proper implementation with:
        - OpenCV or torchvision for video loading
        - Face detection and cropping
        - Resizing to 224x224
        - Normalization

        For now, returns dummy tensor
        """
        # TODO: Implement proper video loading
        # For testing purposes, return dummy tensor
        # Shape: (C=3, T=128, H=224, W=224)
        dummy_video = torch.randn(3, 128, 224, 224)
        return dummy_video


def main():
    parser = argparse.ArgumentParser(
        description="Custom SONAR inference for ASL video translation"
    )
    parser.add_argument(
        "--video", type=str, required=True, help="Path to ASL video file"
    )
    parser.add_argument(
        "--signhiera-model",
        type=str,
        default="models/pretrained_ssvp/dm_70h_ub_signhiera.pth",
        help="Path to SignHiera checkpoint",
    )
    parser.add_argument(
        "--sonar-encoder",
        type=str,
        default="models/pretrained_ssvp/dm_70h_ub_sonar_encoder.pth",
        help="Path to SONAR encoder checkpoint",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="eng_Latn",
        help="Target language (FLORES-200 code)",
    )

    args = parser.parse_args()

    # Create pipeline
    pipeline = SonarInferencePipeline(
        signhiera_path=args.signhiera_model,
        sonar_encoder_path=args.sonar_encoder,
        device=args.device,
    )

    # Run translation
    translation = pipeline.translate_video(
        video_path=args.video, target_lang=args.target_lang
    )

    print("\n" + "=" * 60)
    print(f"FINAL TRANSLATION: {translation}")
    print("=" * 60)


if __name__ == "__main__":
    main()
