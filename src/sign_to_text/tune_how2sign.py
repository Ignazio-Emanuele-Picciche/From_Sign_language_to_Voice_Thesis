"""
Hyperparameter Tuning per How2Sign
===================================

Optuna-based hyperparameter optimization per Sign-to-Text su How2Sign (31k samples).

Differenze rispetto a tune.py (ASLLRP):
- Dataset How2Sign (29.6k train vs 1.8k ASLLRP)
- Landmarks OpenPose 411 features (vs 375 MediaPipe)
- Caption più lunghe (max 50 vs 30)
- Più trials possibili (dataset più grande → convergenza più stabile)

Usage:
    # Tuning rapido (10 trials, 3 epochs)
    python src/sign_to_text/tune_how2sign.py --n_trials 10 --epochs 3
    
    # Tuning completo (30 trials, 5 epochs)
    python src/sign_to_text/tune_how2sign.py --n_trials 50 --epochs 10
    
    # Background con log
    nohup python src/sign_to_text/tune_how2sign.py --n_trials 30 --epochs 5 \
        > logs/tuning_how2sign.log 2>&1 &
"""

import argparse
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
from tqdm import tqdm
import mlflow
import os

# Enable MPS fallback for unsupported operations on Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Import modelli e dataset
import sys

sys.path.append(str(Path(__file__).parent))

from models.seq2seq_transformer import SignToTextTransformer
from data.tokenizer import SignLanguageTokenizer
from data.how2sign_dataset import How2SignDataset


def collate_fn(batch):
    """Collate function per How2SignDataset."""
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


class FastTrainer:
    """Trainer veloce per tuning."""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        grad_clip=1.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.grad_clip = grad_clip

    def train_epoch(self):
        """Training epoch veloce."""
        self.model.train()
        total_loss = 0
        total_tokens = 0

        for batch in self.train_loader:
            src = batch["landmarks"].to(self.device)
            tgt_ids = batch["caption_ids"].to(self.device)
            src_mask = batch["landmarks_mask"].to(self.device)
            tgt_mask = batch["caption_mask"].to(self.device)

            tgt_input = tgt_ids[:, :-1]
            tgt_output = tgt_ids[:, 1:]
            tgt_mask_input = tgt_mask[:, :-1]

            logits = self.model(
                src=src,
                tgt=tgt_input,
                src_key_padding_mask=~src_mask,
                tgt_key_padding_mask=~tgt_mask_input,
            )

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()

            batch_tokens = tgt_mask[:, 1:].sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

        return total_loss / total_tokens

    def validate(self):
        """Validation veloce."""
        self.model.eval()
        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.val_loader:
                src = batch["landmarks"].to(self.device)
                tgt_ids = batch["caption_ids"].to(self.device)
                src_mask = batch["landmarks_mask"].to(self.device)
                tgt_mask = batch["caption_mask"].to(self.device)

                tgt_input = tgt_ids[:, :-1]
                tgt_output = tgt_ids[:, 1:]
                tgt_mask_input = tgt_mask[:, :-1]

                logits = self.model(
                    src=src,
                    tgt=tgt_input,
                    src_key_padding_mask=~src_mask,
                    tgt_key_padding_mask=~tgt_mask_input,
                )

                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
                )

                batch_tokens = tgt_mask[:, 1:].sum().item()
                total_loss += loss.item() * batch_tokens
                total_tokens += batch_tokens

        return total_loss / total_tokens


def objective(trial, args, tokenizer, device):
    """Objective function per Optuna."""

    # ========== HYPERPARAMETERS TO TUNE ==========

    # Learning rate (log scale)
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)

    # Batch size (potenze di 2)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    # Model architecture
    d_model = trial.suggest_categorical("d_model", [256, 512])
    num_encoder_layers = trial.suggest_int("num_encoder_layers", 2, 6)
    num_decoder_layers = trial.suggest_int("num_decoder_layers", 2, 6)
    nhead = trial.suggest_categorical("nhead", [4, 8])
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [1024, 2048])

    # Regularization
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)

    # ========== CREATE DATASETS ==========

    # Usa subset per tuning veloce (25% del dataset)
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

    # Subset per velocità (solo per tuning)
    if args.use_subset:
        subset_size_train = min(5000, len(train_dataset))  # Max 5k samples
        subset_size_val = min(500, len(val_dataset))

        # Convert to list of ints (torch.randperm returns tensor)
        train_indices = torch.randperm(len(train_dataset))[:subset_size_train].tolist()
        val_indices = torch.randperm(len(val_dataset))[:subset_size_val].tolist()

        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 per evitare problemi con Optuna
        collate_fn=collate_fn,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    # ========== CREATE MODEL ==========

    model = SignToTextTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_src_len=args.max_frames,
        max_tgt_len=args.max_caption_len,
        landmark_dim=args.landmark_features,
        pad_idx=tokenizer.pad_token_id,
    )

    model = model.to(device)

    # ========== SETUP TRAINING ==========

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id, label_smoothing=label_smoothing
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    trainer = FastTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        grad_clip=1.0,
    )

    # ========== TRAINING LOOP ==========

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_loss = trainer.train_epoch()
        val_loss = trainer.validate()

        # Update best
        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Report intermediate value (per pruning)
        trial.report(val_loss, epoch)

        # Early stopping se pruned
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Log a MLflow
        if args.use_mlflow:
            mlflow.log_metrics(
                {
                    f"trial_{trial.number}_train_loss": train_loss,
                    f"trial_{trial.number}_val_loss": val_loss,
                },
                step=epoch,
            )

    return best_val_loss


def main():
    parser = argparse.ArgumentParser(description="Tune Sign-to-Text on How2Sign")

    # Tuning params
    parser.add_argument(
        "--n_trials", type=int, default=20, help="Number of Optuna trials"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Epochs per trial")
    parser.add_argument(
        "--use_subset", action="store_true", help="Use dataset subset for faster tuning"
    )
    parser.add_argument(
        "--study_name", type=str, default="how2sign_tuning", help="Optuna study name"
    )

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
    parser.add_argument("--output_dir", type=str, default="results/how2sign_tuning")

    # Dataset params
    parser.add_argument(
        "--max_frames", type=int, default=150, help="Max frames (ridotto per tuning)"
    )
    parser.add_argument("--max_caption_len", type=int, default=50)
    parser.add_argument("--landmark_features", type=int, default=411)

    # MLflow
    parser.add_argument("--use_mlflow", action="store_true", help="Log to MLflow")
    parser.add_argument("--mlflow_experiment", type=str, default="how2sign_tuning")

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
    )

    args = parser.parse_args()

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("🔍 HYPERPARAMETER TUNING - HOW2SIGN")
    print("=" * 80)
    print(f"\n📊 Configuration:")
    print(f"   Trials: {args.n_trials}")
    print(f"   Epochs per trial: {args.epochs}")
    print(f"   Use subset: {args.use_subset}")
    print(f"   Device: {args.device}")
    print(f"   Output: {args.output_dir}")

    # Setup MLflow
    if args.use_mlflow:
        mlflow.set_experiment(args.mlflow_experiment)
        print(f"   MLflow experiment: {args.mlflow_experiment}")

    # Load tokenizer
    print(f"\n1️⃣  Loading tokenizer...")
    tokenizer = SignLanguageTokenizer.load(args.tokenizer_path)
    print(f"   ✓ Vocab size: {tokenizer.vocab_size}")

    # Create Optuna study
    print(f"\n2️⃣  Creating Optuna study...")

    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=2, interval_steps=1
        ),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    print(f"   ✓ Study: {args.study_name}")
    print(f"   ✓ Pruner: MedianPruner")
    print(f"   ✓ Sampler: TPESampler")

    # Run optimization
    print(f"\n{'='*80}")
    print(f"🎯 TUNING START")
    print(f"{'='*80}\n")

    start_time = time.time()

    if args.use_mlflow:
        with mlflow.start_run(run_name=f"{args.study_name}_optimization"):
            mlflow.log_params(
                {
                    "n_trials": args.n_trials,
                    "epochs_per_trial": args.epochs,
                    "use_subset": args.use_subset,
                    "max_frames": args.max_frames,
                    "landmark_features": args.landmark_features,
                }
            )

            study.optimize(
                lambda trial: objective(trial, args, tokenizer, args.device),
                n_trials=args.n_trials,
                show_progress_bar=True,
            )
    else:
        study.optimize(
            lambda trial: objective(trial, args, tokenizer, args.device),
            n_trials=args.n_trials,
            show_progress_bar=True,
        )

    elapsed = time.time() - start_time

    # ========== RESULTS ==========

    print(f"\n{'='*80}")
    print(f"✅ TUNING COMPLETE!")
    print(f"{'='*80}")

    print(f"\n📊 Results:")
    print(f"   Total trials: {len(study.trials)}")
    print(f"   Best trial: {study.best_trial.number}")
    print(f"   Best val loss: {study.best_value:.4f}")
    print(f"   Time: {elapsed/60:.1f} min")

    print(f"\n🏆 Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")

    # Save results
    print(f"\n💾 Saving results...")

    # Best params JSON
    with open(output_dir / "best_hyperparameters.json", "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"   ✓ Best params: {output_dir / 'best_hyperparameters.json'}")

    # Study statistics
    study_stats = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
        "elapsed_time_min": elapsed / 60,
        "best_params": study.best_params,
    }

    with open(output_dir / "study_statistics.json", "w") as f:
        json.dump(study_stats, f, indent=2)
    print(f"   ✓ Statistics: {output_dir / 'study_statistics.json'}")

    # Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)

        print(f"\n📈 Parameter Importance:")
        for param, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
            print(f"   {param}: {imp:.4f}")

        with open(output_dir / "param_importances.json", "w") as f:
            json.dump(importance, f, indent=2)
        print(f"   ✓ Importances: {output_dir / 'param_importances.json'}")

    except Exception as e:
        print(f"   ⚠️  Could not compute importances: {e}")

    # Visualization
    try:
        import optuna.visualization as vis

        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(str(output_dir / "optimization_history.html"))

        # Param importances
        fig = vis.plot_param_importances(study)
        fig.write_html(str(output_dir / "param_importances.html"))

        print(f"   ✓ Visualizations: {output_dir}/*.html")

    except Exception as e:
        print(f"   ⚠️  Could not create visualizations: {e}")

    print(f"\n🚀 Next step: Train with best hyperparameters!")
    print(f"   python src/sign_to_text/train_how2sign_from_tuning.py")
    print(f"\n")


if __name__ == "__main__":
    main()
