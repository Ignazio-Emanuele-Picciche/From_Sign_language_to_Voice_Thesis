"""
Landmarks-to-Text Transformer Model
====================================

Transformer model for sign language translation using pose landmarks.

Architecture:
    Input: Landmarks sequence (num_frames, 137, 2)
    Encoder: Temporal Transformer (6 layers)
    Decoder: Cross-Attention Transformer (6 layers)
    Output: English text tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class LandmarksEncoder(nn.Module):
    """Encode landmarks sequence to hidden representations."""

    def __init__(
        self,
        landmark_dim: int = 137,
        coordinate_dim: int = 2,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 1000,
    ):
        super().__init__()

        # Embed landmarks to d_model dimension
        self.landmark_embedding = nn.Linear(landmark_dim * coordinate_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.d_model = d_model

    def forward(
        self,
        landmarks: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            landmarks: (batch, seq_len, 137, 2)
            src_key_padding_mask: (batch, seq_len) - True for padding positions

        Returns:
            Encoded features: (batch, seq_len, d_model)
        """
        batch_size, seq_len = landmarks.shape[:2]

        # Flatten landmarks: (batch, seq_len, 137*2)
        landmarks_flat = landmarks.reshape(batch_size, seq_len, -1)

        # Embed to d_model: (batch, seq_len, d_model)
        x = self.landmark_embedding(landmarks_flat)
        x = x * math.sqrt(self.d_model)  # Scale

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        return x


class TextDecoder(nn.Module):
    """Decode hidden representations to text tokens."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_len: int = 200,
    ):
        super().__init__()

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        self.d_model = d_model

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target tokens (batch, tgt_len)
            memory: Encoder output (batch, src_len, d_model)
            tgt_mask: Causal mask for decoder (tgt_len, tgt_len)
            memory_key_padding_mask: (batch, src_len)
            tgt_key_padding_mask: (batch, tgt_len)

        Returns:
            Logits: (batch, tgt_len, vocab_size)
        """
        # Embed tokens: (batch, tgt_len, d_model)
        x = self.token_embedding(tgt)
        x = x * math.sqrt(self.d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer decoding
        x = self.transformer_decoder(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        # Project to vocabulary
        logits = self.output_projection(x)

        return logits

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask


class LandmarksToTextTransformer(nn.Module):
    """Complete Landmarks-to-Text translation model."""

    def __init__(
        self,
        vocab_size: int,
        landmark_dim: int = 137,
        coordinate_dim: int = 2,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_src_len: int = 1000,
        max_tgt_len: int = 200,
    ):
        super().__init__()

        self.encoder = LandmarksEncoder(
            landmark_dim=landmark_dim,
            coordinate_dim=coordinate_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_src_len,
        )

        self.decoder = TextDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_tgt_len,
        )

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src_landmarks: torch.Tensor,
        tgt_tokens: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training.

        Args:
            src_landmarks: (batch, src_len, 137, 2)
            tgt_tokens: (batch, tgt_len)
            src_key_padding_mask: (batch, src_len)
            tgt_key_padding_mask: (batch, tgt_len)

        Returns:
            Logits: (batch, tgt_len, vocab_size)
        """
        # Encode landmarks
        memory = self.encoder(src_landmarks, src_key_padding_mask)

        # Generate causal mask for decoder
        tgt_len = tgt_tokens.size(1)
        tgt_mask = self.decoder.generate_square_subsequent_mask(
            tgt_len, tgt_tokens.device
        )

        # Decode to text
        logits = self.decoder(
            tgt=tgt_tokens,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )

        return logits

    @torch.no_grad()
    def generate(
        self,
        src_landmarks: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        max_len: int = 100,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text from landmarks (inference).

        Args:
            src_landmarks: (batch, src_len, 137, 2)
            src_key_padding_mask: (batch, src_len)
            max_len: Maximum generation length
            bos_token_id: Beginning-of-sequence token ID
            eos_token_id: End-of-sequence token ID
            temperature: Sampling temperature
            top_k: Top-k sampling (None = greedy)

        Returns:
            Generated tokens: (batch, gen_len)
        """
        batch_size = src_landmarks.size(0)
        device = src_landmarks.device

        # Encode landmarks once
        memory = self.encoder(src_landmarks, src_key_padding_mask)

        # Initialize with BOS token
        generated = torch.full(
            (batch_size, 1), bos_token_id, dtype=torch.long, device=device
        )

        # Track finished sequences
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            # Generate causal mask
            tgt_mask = self.decoder.generate_square_subsequent_mask(
                generated.size(1), device
            )

            # Decode
            logits = self.decoder(
                tgt=generated,
                memory=memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=src_key_padding_mask,
            )

            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature

            # Sample next token
            if top_k is not None:
                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(
                    next_token_logits, top_k, dim=-1
                )
                probs = F.softmax(top_k_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token = torch.gather(top_k_indices, 1, next_token)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS
            finished |= next_token.squeeze(-1) == eos_token_id
            if finished.all():
                break

        return generated

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(vocab_size: int, **kwargs) -> LandmarksToTextTransformer:
    """
    Create Landmarks-to-Text Transformer model.

    Args:
        vocab_size: Size of vocabulary
        **kwargs: Additional model arguments

    Returns:
        LandmarksToTextTransformer model
    """
    model = LandmarksToTextTransformer(vocab_size=vocab_size, **kwargs)

    num_params = model.count_parameters()
    print(f"Created model with {num_params:,} parameters")

    return model


if __name__ == "__main__":
    # Test model
    print("Testing Landmarks-to-Text Transformer...")

    vocab_size = 5000
    batch_size = 4
    src_len = 100
    tgt_len = 20

    # Create model
    model = create_model(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
    )

    # Test forward pass
    src_landmarks = torch.randn(batch_size, src_len, 137, 2)
    tgt_tokens = torch.randint(0, vocab_size, (batch_size, tgt_len))

    logits = model(src_landmarks, tgt_tokens)
    print(f"✓ Forward pass: {logits.shape}")

    # Test generation
    generated = model.generate(src_landmarks, max_len=30)
    print(f"✓ Generation: {generated.shape}")

    print("\n✅ Model test passed!")
