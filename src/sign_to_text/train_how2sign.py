"""
Training Script per How2Sign Dataset
=====================================

Train Sign-to-Text Transformer su How2Sign (31k samples) con OpenPose landmarks.

Differenze rispetto a train.py:
- Usa How2SignDataset invece di SignLanguageDataset
- Landmarks OpenPose: 411 features (vs 375 MediaPipe)
- Caption più lunghe: max_len=50 (vs 30)
- Dataset più grande: batch accumulation per gestire memoria

Usage:
    # Training completo
    python src/sign_to_text/train_how2sign.py --epochs 30
    
    # Training veloce (test)
    python src/sign_to_text/train_how2sign.py --epochs 2 --batch_size 8
    
    # Con hyperparameters ottimizzati
    python src/sign_to_text/train_how2sign.py \
        --lr 3.24e-05 \
        --batch_size 16 \
        --d_model 512 \
        --num_encoder_layers 2
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
from tqdm import tqdm

# Import modelli e dataset
import sys

sys.path.append(str(Path(__file__).parent))

from models.seq2seq_transformer import SignToTextTransformer
from data.tokenizer import SignLanguageTokenizer
from data.how2sign_dataset import How2SignDataset, load_openpose_landmarks


def collate_fn(batch):
    """Collate function compatibile con How2SignDataset."""
    landmarks = torch.stack([item["landmarks"] for item in batch])
    landmarks_mask = torch.stack([item["landmarks_mask"] for item in batch])
    caption_ids = torch.stack([item["caption_ids"] for item in batch])
    caption_mask = torch.stack([item["caption_mask"] for item in batch])

    return {
        "landmarks": landmarks,
        "landmarks_mask": landmarks_mask,
        "caption_ids": caption_ids,
        "caption_mask": caption_mask,
        "caption_texts": [item["caption_text"] for item in batch],
        "video_names": [item["video_name"] for item in batch],
        "n_frames_original": torch.tensor(
            [item["n_frames_original"] for item in batch]
        ),
    }


def train_epoch(model, dataloader, optimizer, criterion, device, grad_clip=1.0):
    """Training loop per un epoch."""
    model.train()
    total_loss = 0
    total_tokens = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        src = batch["landmarks"].to(device)  # (B, max_frames, 411)
        tgt_ids = batch["caption_ids"].to(device)  # (B, max_caption_len)
        src_mask = batch["landmarks_mask"].to(device)  # (B, max_frames)
        tgt_mask = batch["caption_mask"].to(device)  # (B, max_caption_len)

        # Forward pass
        # Input: tgt_ids[:, :-1] (senza ultimo token)
        # Target: tgt_ids[:, 1:] (senza primo token BOS)
        tgt_input = tgt_ids[:, :-1]
        tgt_output = tgt_ids[:, 1:]
        tgt_mask_input = tgt_mask[:, :-1]

        logits = model(
            src=src,
            tgt=tgt_input,
            src_key_padding_mask=~src_mask,  # PyTorch usa True=masked
            tgt_key_padding_mask=~tgt_mask_input,
        )

        # Calcola loss
        # logits: (B, tgt_len-1, vocab_size)
        # tgt_output: (B, tgt_len-1)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Stats
        batch_tokens = tgt_mask[:, 1:].sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

        # Update progress
        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / total_tokens:.4f}",
            }
        )

    return total_loss / total_tokens


def validate(model, dataloader, criterion, device):
    """Validation loop."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            src = batch["landmarks"].to(device)
            tgt_ids = batch["caption_ids"].to(device)
            src_mask = batch["landmarks_mask"].to(device)
            tgt_mask = batch["caption_mask"].to(device)

            tgt_input = tgt_ids[:, :-1]
            tgt_output = tgt_ids[:, 1:]
            tgt_mask_input = tgt_mask[:, :-1]

            logits = model(
                src=src,
                tgt=tgt_input,
                src_key_padding_mask=~src_mask,
                tgt_key_padding_mask=~tgt_mask_input,
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
            )

            batch_tokens = tgt_mask[:, 1:].sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    return total_loss / total_tokens


def main():
    parser = argparse.ArgumentParser(description="Train Sign-to-Text on How2Sign")

    # Paths
    parser.add_argument(
        "--train_csv", type=str, default="results/how2sign_splits/train_split.csv"
    )
    parser.add_argument(
        "--val_csv", type=str, default="results/how2sign_splits/val_split.csv"
    )
    parser.add_argument(
        "--train_openpose_dir",
        type=str,
        default="data/raw/train/openpose_output_train/json",
    )
    parser.add_argument(
        "--val_openpose_dir", type=str, default="data/raw/val/openpose_output_val/json"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, default="models/sign_to_text/tokenizer.json"
    )
    parser.add_argument(
        "--output_dir", type=str, default="models/sign_to_text/how2sign"
    )

    # Model hyperparameters
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder_layers", type=int, default=4)
    parser.add_argument("--num_decoder_layers", type=int, default=4)
    parser.add_argument("--dim_feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Stop training if val_loss doesn't improve for N epochs",
    )

    # Dataset params
    parser.add_argument("--max_frames", type=int, default=200)
    parser.add_argument(
        "--max_caption_len", type=int, default=50
    )  # How2Sign ha caption lunghe
    parser.add_argument("--landmark_features", type=int, default=411)  # OpenPose

    # Other
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("=" * 80)
    print("🚀 TRAINING SIGN-TO-TEXT ON HOW2SIGN")
    print("=" * 80)
    print(f"\n📊 Dataset: How2Sign (31k train, 1.7k val)")
    print(f"🔧 Landmark features: {args.landmark_features} (OpenPose)")
    print(f"💻 Device: {args.device}")
    print(f"📁 Output: {args.output_dir}")

    # Load tokenizer
    print(f"\n1️⃣  Loading tokenizer...")
    tokenizer = SignLanguageTokenizer.load(args.tokenizer_path)
    print(f"   ✓ Vocab size: {tokenizer.vocab_size}")

    # Create datasets
    print(f"\n2️⃣  Creating datasets...")
    train_dataset = How2SignDataset(
        split_csv=args.train_csv,
        openpose_dir=args.train_openpose_dir,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        max_caption_len=args.max_caption_len,
        landmark_features=args.landmark_features,
    )

    val_dataset = How2SignDataset(
        split_csv=args.val_csv,
        openpose_dir=args.val_openpose_dir,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        max_caption_len=args.max_caption_len,
        landmark_features=args.landmark_features,
    )

    print(f"   ✓ Train: {len(train_dataset)} samples")
    print(f"   ✓ Val:   {len(val_dataset)} samples")

    # Create dataloaders
    print(f"\n3️⃣  Creating dataloaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True if args.device == "cuda" else False,
    )

    print(f"   ✓ Train batches: {len(train_loader)}")
    print(f"   ✓ Val batches:   {len(val_loader)}")

    # Create model
    print(f"\n4️⃣  Creating model...")
    model = SignToTextTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_src_len=args.max_frames,
        max_tgt_len=args.max_caption_len,
        landmark_dim=args.landmark_features,  # 411 per OpenPose
        pad_idx=tokenizer.pad_token_id,
    )

    model = model.to(args.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✓ Model parameters: {num_params:,}")
    print(f"   ✓ Input dim: {args.landmark_features} features")
    print(f"   ✓ Model dim: {args.d_model}")

    # Loss & Optimizer
    print(f"\n5️⃣  Setup training...")
    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id, label_smoothing=args.label_smoothing
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Learning rate scheduler - CosineAnnealingLR per decay dolce
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    print(f"   ✓ Criterion: CrossEntropyLoss (label_smoothing={args.label_smoothing})")
    print(f"   ✓ Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})")
    print(f"   ✓ Scheduler: CosineAnnealingLR (eta_min=1e-6)")

    # Training loop
    print(f"\n{'='*80}")
    print(f"🎯 TRAINING START")
    print(f"{'='*80}")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\n📅 Epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")

        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, args.device, args.grad_clip
        )

        # Validate
        val_loss = validate(model, val_loader, criterion, args.device)

        # Scheduler step (CosineAnnealingLR non richiede metrica)
        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        # Print stats
        print(f"\n📊 Epoch {epoch} Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")
        print(f"   LR:         {current_lr:.2e}")
        print(f"   Time:       {epoch_time/60:.1f} min")

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)

        # Save checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": vars(args),
        }

        # Save last checkpoint
        torch.save(checkpoint, output_dir / "last_checkpoint.pt")

        # Save best checkpoint and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best_checkpoint.pt")
            print(f"   ✅ New best model saved! (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"   ⚠️  No improvement for {epochs_without_improvement} epoch(s)")

            # Early stopping check
            if epochs_without_improvement >= args.early_stopping_patience:
                print(f"\n{'='*80}")
                print(f"⛔ EARLY STOPPING triggered!")
                print(f"{'='*80}")
                print(f"   No improvement for {args.early_stopping_patience} epochs")
                print(
                    f"   Best val_loss: {best_val_loss:.4f} (epoch {epoch - args.early_stopping_patience})"
                )
                print(f"   Stopping training to prevent overfitting")
                break

        # Save history
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📊 Final Results:")
    print(f"   Best Val Loss: {best_val_loss:.4f}")
    print(f"   Total Epochs:  {epoch}")
    if epochs_without_improvement >= args.early_stopping_patience:
        print(
            f"   Early Stopped: Yes (after {args.early_stopping_patience} epochs without improvement)"
        )
    print(f"   Checkpoints:   {output_dir}")
    print(f"\n🎉 How2Sign model ready!")
    print(f"\n")


if __name__ == "__main__":
    main()
