"""
Sign-to-Text Transformer Model
===============================

Seq2Seq Transformer per traduzione Sign Language → Testo.

Architettura:
- Encoder: Processa landmarks MediaPipe (frames × 375 features)
- Decoder: Genera caption token-by-token con attention

Usage:
    model = SignToTextTransformer(
        vocab_size=3000,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4
    )

    logits = model(
        src=landmarks,           # (B, src_len, 375)
        tgt=caption_ids,         # (B, tgt_len)
        src_key_padding_mask=landmarks_mask,
        tgt_key_padding_mask=caption_mask
    )
    # Output: (B, tgt_len, vocab_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class PositionalEncoding(nn.Module):
    """
    Positional Encoding per Transformer.
    Aggiunge informazione di posizione ai token embeddings.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Crea positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (seq_len, batch_size, d_model) or (batch_size, seq_len, d_model)

        Returns:
            x con positional encoding aggiunto
        """
        if x.dim() == 3 and x.size(1) != 1:
            # Batch-first: (B, seq_len, d_model)
            x = x + self.pe[: x.size(1), 0, :].unsqueeze(0)
        else:
            # Seq-first: (seq_len, B, d_model)
            x = x + self.pe[: x.size(0)]

        return self.dropout(x)


class SignToTextTransformer(nn.Module):
    """
    Transformer Encoder-Decoder per Sign-to-Text translation.

    Architecture:
        Input (landmarks) → Encoder → Context vectors
        Context + Target tokens → Decoder → Output logits
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_src_len: int = 200,
        max_tgt_len: int = 50,
        landmark_dim: int = 375,
        pad_idx: int = 0,
    ):
        """
        Args:
            vocab_size: Dimensione vocabulary
            d_model: Dimensione hidden state
            nhead: Numero attention heads
            num_encoder_layers: Numero encoder layers
            num_decoder_layers: Numero decoder layers
            dim_feedforward: Dimensione feedforward network
            dropout: Dropout rate
            max_src_len: Max lunghezza source (frames)
            max_tgt_len: Max lunghezza target (caption)
            landmark_dim: Dimensione input landmarks (375)
            pad_idx: Index del PAD token
        """
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx

        # ===== ENCODER =====

        # Input projection: landmarks → d_model
        self.src_projection = nn.Linear(landmark_dim, d_model)

        # Positional encoding per source
        self.src_pos_encoder = PositionalEncoding(
            d_model, max_len=max_src_len, dropout=dropout
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers, norm=nn.LayerNorm(d_model)
        )

        # ===== DECODER =====

        # Token embedding
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        # Positional encoding per target
        self.tgt_pos_encoder = PositionalEncoding(
            d_model, max_len=max_tgt_len, dropout=dropout
        )

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers, norm=nn.LayerNorm(d_model)
        )

        # ===== OUTPUT =====

        # Output projection: d_model → vocab_size
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier/Kaiming initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self, src: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode landmarks → context vectors.

        Args:
            src: (B, src_len, landmark_dim) landmarks
            src_key_padding_mask: (B, src_len) bool mask (True = padding)

        Returns:
            memory: (B, src_len, d_model) context vectors
        """
        # Project landmarks → d_model
        src = self.src_projection(src)  # (B, src_len, d_model)

        # Add positional encoding
        src = self.src_pos_encoder(src)

        # Encode
        # Nota: PyTorch TransformerEncoder accetta key_padding_mask invertito
        # (True = ignore, False = attend)
        memory = self.encoder(
            src,
            src_key_padding_mask=(
                ~src_key_padding_mask if src_key_padding_mask is not None else None
            ),
        )

        return memory

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode context + target tokens → output logits.

        Args:
            tgt: (B, tgt_len) target token IDs
            memory: (B, src_len, d_model) encoder output
            tgt_mask: (tgt_len, tgt_len) causal mask
            tgt_key_padding_mask: (B, tgt_len) bool mask
            memory_key_padding_mask: (B, src_len) bool mask

        Returns:
            logits: (B, tgt_len, vocab_size)
        """
        # Embed target tokens
        tgt = self.tgt_embedding(tgt) * math.sqrt(self.d_model)  # (B, tgt_len, d_model)

        # Add positional encoding
        tgt = self.tgt_pos_encoder(tgt)

        # Decode
        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=(
                ~tgt_key_padding_mask if tgt_key_padding_mask is not None else None
            ),
            memory_key_padding_mask=(
                ~memory_key_padding_mask
                if memory_key_padding_mask is not None
                else None
            ),
        )

        # Project to vocabulary
        logits = self.output_projection(output)  # (B, tgt_len, vocab_size)

        return logits

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass completo.

        Args:
            src: (B, src_len, landmark_dim) source landmarks
            tgt: (B, tgt_len) target token IDs
            src_key_padding_mask: (B, src_len) bool mask (True = valido, False = padding)
            tgt_key_padding_mask: (B, tgt_len) bool mask (True = valido, False = padding)

        Returns:
            logits: (B, tgt_len, vocab_size)
        """
        # Encode
        memory = self.encode(src, src_key_padding_mask)

        # Genera causal mask per decoder (impedisce di vedere token futuri)
        tgt_len = tgt.size(1)
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)

        # Decode
        logits = self.decode(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return logits

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """
        Genera causal mask per decoder.

        Args:
            sz: Dimensione sequenza

        Returns:
            mask: (sz, sz) upper-triangular matrix con -inf
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def generate(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        max_len: int = 30,
        sos_idx: int = 2,
        eos_idx: int = 3,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generazione autoregressive (beam search=1).

        Args:
            src: (B, src_len, landmark_dim) source
            src_key_padding_mask: (B, src_len) mask
            max_len: Max lunghezza generazione
            sos_idx: Index [SOS] token
            eos_idx: Index [EOS] token
            temperature: Sampling temperature (1.0 = normal)
            top_k: Top-k sampling (None = greedy)

        Returns:
            generated: (B, max_len) token IDs generati
        """
        self.eval()
        batch_size = src.size(0)
        device = src.device

        # Encode una volta sola
        with torch.no_grad():
            memory = self.encode(src, src_key_padding_mask)

        # Inizializza con [SOS]
        generated = torch.full(
            (batch_size, 1), fill_value=sos_idx, dtype=torch.long, device=device
        )

        # Genera token-by-token
        for _ in range(max_len - 1):
            # Decode
            with torch.no_grad():
                logits = self.decode(
                    generated,
                    memory,
                    tgt_mask=self.generate_square_subsequent_mask(generated.size(1)).to(
                        device
                    ),
                    memory_key_padding_mask=src_key_padding_mask,
                )

            # Prendi logits ultimo token
            next_token_logits = logits[:, -1, :] / temperature  # (B, vocab_size)

            # Sampling
            if top_k is not None:
                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(
                    next_token_logits, top_k, dim=-1
                )
                probs = F.softmax(top_k_logits, dim=-1)
                next_token_idx = torch.multinomial(probs, num_samples=1)
                next_token = top_k_indices.gather(-1, next_token_idx)
            else:
                # Greedy
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Append
            generated = torch.cat([generated, next_token], dim=1)

            # Check se tutti hanno generato [EOS]
            if (next_token == eos_idx).all():
                break

        return generated

    def count_parameters(self) -> int:
        """Conta parametri trainable."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test script
if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"🧪 TESTING SIGN-TO-TEXT TRANSFORMER")
    print(f"{'='*80}")

    # Hyperparameters
    vocab_size = 3000
    d_model = 256
    nhead = 8
    num_encoder_layers = 4
    num_decoder_layers = 4

    # Crea modello
    print(f"\n🔧 Creating model...")
    model = SignToTextTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=1024,
        dropout=0.1,
        max_src_len=200,
        max_tgt_len=30,
        landmark_dim=375,
        pad_idx=0,
    )

    print(f"   ✓ Model created")
    print(f"   Parameters: {model.count_parameters():,}")

    # Test forward pass
    print(f"\n🔄 Testing forward pass...")

    batch_size = 4
    src_len = 150
    tgt_len = 20

    # Dummy input
    src = torch.randn(batch_size, src_len, 375)
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
    src_mask = torch.ones(batch_size, src_len, dtype=torch.bool)
    tgt_mask = torch.ones(batch_size, tgt_len, dtype=torch.bool)

    # Forward
    logits = model(src, tgt, src_mask, tgt_mask)

    print(f"   Input landmarks: {src.shape}")
    print(f"   Input captions: {tgt.shape}")
    print(f"   Output logits: {logits.shape}")
    print(f"   ✓ Forward pass successful!")

    # Test generation
    print(f"\n🎯 Testing generation...")

    generated = model.generate(
        src[:1],  # Single sample
        src_key_padding_mask=src_mask[:1],
        max_len=30,
        sos_idx=2,
        eos_idx=3,
    )

    print(f"   Generated tokens: {generated.shape}")
    print(f"   Generated IDs: {generated[0, :15].tolist()}...")
    print(f"   ✓ Generation successful!")

    # Memory usage
    print(f"\n💾 Model size:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = total_params * 4 / (1024**2)  # Assuming float32

    print(f"   Total params: {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")
    print(f"   Model size: {size_mb:.2f} MB")

    print(f"\n{'='*80}")
    print(f"✅ ALL TESTS PASSED!")
    print(f"{'='*80}")
    print(f"\n🚀 Next: Implement training script (src/sign_to_text/train.py)")
    print(f"\n")
