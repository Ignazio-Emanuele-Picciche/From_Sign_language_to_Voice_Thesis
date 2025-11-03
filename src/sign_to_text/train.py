"""
Sign-to-Text Training Script
=============================

Training loop per Seq2Seq Transformer con:
- MLflow logging
- Early stopping
- Learning rate scheduling
- BLEU metric
- Checkpoint saving

Usage:
    python src/sign_to_text/train.py --epochs 50 --batch_size 16 --lr 1e-4
    
python src/sign_to_text/train.py \
    --epochs 50 \
    --batch_size 16 \
    --d_model 256 \
    --nhead 8 \
    --num_encoder_layers 4 \
    --num_decoder_layers 4 \
    --lr 1e-4 \
    --patience 15
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import mlflow
import mlflow.pytorch
from tqdm import tqdm
import time
from typing import Dict, List
import sys
import os

# Enable MPS fallback for unsupported operations on Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.sign_to_text.data.tokenizer import SignLanguageTokenizer
from src.sign_to_text.data.dataset import get_dataloaders
from src.sign_to_text.models.seq2seq_transformer import SignToTextTransformer

# Import BLEU
try:
    from torchmetrics.text import BLEUScore

    BLEU_AVAILABLE = True
except ImportError:
    print("⚠️  torchmetrics not available, BLEU will be skipped")
    BLEU_AVAILABLE = False


class Trainer:
    """Training manager per Sign-to-Text model."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: SignLanguageTokenizer,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        patience: int = 10,
        checkpoint_dir: str = "models/sign_to_text/checkpoints",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        # Scheduler: ReduceLROnPlateau
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id, label_smoothing=0.1
        )

        # Early stopping
        self.patience = patience
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # BLEU metric
        if BLEU_AVAILABLE:
            self.bleu_metric = BLEUScore(n_gram=4)
        else:
            self.bleu_metric = None

        # Tracking
        self.epoch = 0
        self.global_step = 0

    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1} [Train]")

        for batch in progress_bar:
            # Move to device
            src = batch["landmarks"].to(self.device)
            tgt = batch["caption_ids"].to(self.device)
            src_mask = batch["landmarks_mask"].to(self.device)
            tgt_mask = batch["caption_mask"].to(self.device)

            # Forward
            # Teacher forcing: usa ground truth come input al decoder
            # Shifta target: input = [SOS, w1, w2], output = [w1, w2, EOS]
            tgt_input = tgt[:, :-1]  # Remove last token
            tgt_output = tgt[:, 1:]  # Remove first token ([SOS])
            tgt_mask_input = tgt_mask[:, :-1]

            logits = self.model(
                src=src,
                tgt=tgt_input,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_mask_input,
            )

            # Loss
            # Reshape: (B, T, V) → (B*T, V), (B, T) → (B*T,)
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = tgt_output.reshape(-1)

            loss = self.criterion(logits_flat, targets_flat)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # Track
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss/num_batches:.4f}",
                }
            )

            # Log to MLflow ogni 50 step
            if self.global_step % 50 == 0:
                mlflow.log_metric("train_loss_step", loss.item(), step=self.global_step)

        avg_loss = total_loss / num_batches
        return {"loss": avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        # Per BLEU
        all_predictions = []
        all_references = []

        progress_bar = tqdm(self.val_loader, desc=f"Epoch {self.epoch+1} [Val]")

        for batch in progress_bar:
            src = batch["landmarks"].to(self.device)
            tgt = batch["caption_ids"].to(self.device)
            src_mask = batch["landmarks_mask"].to(self.device)
            tgt_mask = batch["caption_mask"].to(self.device)

            # Forward (teacher forcing)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_mask_input = tgt_mask[:, :-1]

            logits = self.model(
                src=src,
                tgt=tgt_input,
                src_key_padding_mask=src_mask,
                tgt_key_padding_mask=tgt_mask_input,
            )

            # Loss
            logits_flat = logits.reshape(-1, logits.size(-1))
            targets_flat = tgt_output.reshape(-1)
            loss = self.criterion(logits_flat, targets_flat)

            total_loss += loss.item()
            num_batches += 1

            # Generate predictions per BLEU (sample 4 batch)
            if num_batches <= 4:
                generated = self.model.generate(
                    src,
                    src_key_padding_mask=src_mask,
                    max_len=30,
                    sos_idx=self.tokenizer.sos_token_id,
                    eos_idx=self.tokenizer.eos_token_id,
                )

                # Decode
                for pred_ids, ref_text in zip(generated, batch["caption_texts"]):
                    pred_text = self.tokenizer.decode(pred_ids.tolist())
                    all_predictions.append(pred_text)
                    all_references.append([ref_text])  # BLEU expects list of references

            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{total_loss/num_batches:.4f}",
                }
            )

        avg_loss = total_loss / num_batches

        # Compute BLEU
        bleu_score = 0.0
        if self.bleu_metric and len(all_predictions) > 0:
            try:
                bleu_score = self.bleu_metric(all_predictions, all_references).item()
            except Exception as e:
                print(f"   ⚠️  BLEU computation failed: {e}")

        return {
            "loss": avg_loss,
            "bleu": bleu_score,
            "sample_predictions": all_predictions[:5],
            "sample_references": [ref[0] for ref in all_references[:5]],
        }

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        # Save latest
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"   💾 Best model saved: {best_path}")

    def train(self, num_epochs: int):
        """Full training loop."""
        print(f"\n{'='*80}")
        print(f"🚀 TRAINING STARTED")
        print(f"{'='*80}")
        print(f"   Device: {self.device}")
        print(f"   Train batches: {len(self.train_loader)}")
        print(f"   Val batches: {len(self.val_loader)}")
        print(f"   Epochs: {num_epochs}")
        print(f"   Learning rate: {self.optimizer.param_groups[0]['lr']}")
        print(f"\n")

        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            epoch_time = time.time() - epoch_start

            # Logging
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{num_epochs} - {epoch_time:.1f}s")
            print(f"{'='*80}")
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss:   {val_metrics['loss']:.4f}")
            print(f"Val BLEU:   {val_metrics['bleu']:.4f}")

            # Sample predictions
            if val_metrics["sample_predictions"]:
                print(f"\nSample Predictions:")
                for i, (pred, ref) in enumerate(
                    zip(
                        val_metrics["sample_predictions"][:3],
                        val_metrics["sample_references"][:3],
                    )
                ):
                    print(f"  [{i+1}] Pred: {pred}")
                    print(f"      Ref:  {ref}")

            # MLflow
            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "val_loss": val_metrics["loss"],
                    "val_bleu": val_metrics["bleu"],
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    "epoch_time": epoch_time,
                },
                step=epoch,
            )

            # Scheduler step
            self.scheduler.step(val_metrics["loss"])

            # Early stopping
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.patience_counter = 0
                self.save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(is_best=False)

            if self.patience_counter >= self.patience:
                print(f"\n⚠️  Early stopping triggered (patience={self.patience})")
                break

        print(f"\n{'='*80}")
        print(f"✅ TRAINING COMPLETED!")
        print(f"{'='*80}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"\n")


def main():
    parser = argparse.ArgumentParser(description="Train Sign-to-Text model")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=256, help="Model dimension")
    parser.add_argument(
        "--nhead", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_encoder_layers", type=int, default=4, help="Encoder layers"
    )
    parser.add_argument(
        "--num_decoder_layers", type=int, default=4, help="Decoder layers"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--experiment_name", type=str, default="sign_to_text", help="MLflow experiment"
    )

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Device
    # Use CPU for now due to MPS compatibility issues with TransformerEncoder
    # (nested_tensor_from_mask_left_aligned not implemented on MPS)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"  # Skip MPS due to compatibility issues
    )
    print(f"\n🖥️  Device: {device}")

    # Load tokenizer
    print(f"\n📂 Loading tokenizer...")
    tokenizer = SignLanguageTokenizer.load("models/sign_to_text/tokenizer.json")

    # Create dataloaders
    print(f"\n📊 Creating dataloaders...")
    loaders = get_dataloaders(
        batch_size=args.batch_size, max_frames=200, max_caption_len=30, num_workers=4
    )

    # Create model
    print(f"\n🔧 Creating model...")
    model = SignToTextTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
        max_src_len=200,
        max_tgt_len=30,
        landmark_dim=375,
        pad_idx=tokenizer.pad_token_id,
    )

    print(f"   ✓ Model parameters: {model.count_parameters():,}")

    # MLflow setup
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():
        # Log params
        mlflow.log_params(
            {
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "d_model": args.d_model,
                "nhead": args.nhead,
                "num_encoder_layers": args.num_encoder_layers,
                "num_decoder_layers": args.num_decoder_layers,
                "dropout": args.dropout,
                "patience": args.patience,
                "vocab_size": tokenizer.get_vocab_size(),
                "train_samples": len(loaders["train"].dataset),
                "val_samples": len(loaders["val"].dataset),
                "model_parameters": model.count_parameters(),
            }
        )

        # Train
        trainer = Trainer(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            tokenizer=tokenizer,
            device=device,
            lr=args.lr,
            patience=args.patience,
        )

        trainer.train(num_epochs=args.epochs)

        # Log final model
        mlflow.pytorch.log_model(model, "model")

        print(f"✅ Model logged to MLflow")


if __name__ == "__main__":
    main()
