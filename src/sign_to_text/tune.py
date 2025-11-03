"""
Sign-to-Text Hyperparameter Tuning with Optuna
===============================================

Ottimizzazione automatica degli iperparametri usando Optuna:
- Integrazione con MLflow per logging
- Trials rapidi con budget limitato di epoche
- Ottimizzazione basata su Val BLEU o Val Loss
- Ricerca su learning rate, batch size, architettura, regularization

Usage:
    # Tune con 30 trials, 5 epochs per trial, ottimizza BLEU
    python src/sign_to_text/tune.py --n_trials 30 --epochs 5 --optimize bleu

    # Tune con 20 trials, 10 epochs per trial, ottimizza loss
    python src/sign_to_text/tune.py --n_trials 20 --epochs 10 --optimize loss
    
.venv/bin/python src/sign_to_text/tune.py \
    --n_trials 50 \
    --epochs 10 \
    --optimize bleu
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback
import sys
import os
import json
from typing import Dict, Any

# Enable MPS fallback for unsupported operations on Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.sign_to_text.data.tokenizer import SignLanguageTokenizer
from src.sign_to_text.data.dataset import get_dataloaders, SignLanguageDataset
from src.sign_to_text.models.seq2seq_transformer import SignToTextTransformer

# Import BLEU
try:
    from torchmetrics.text import BLEUScore

    BLEU_AVAILABLE = True
except ImportError:
    print("⚠️  torchmetrics not available, BLEU will be skipped")
    BLEU_AVAILABLE = False


class FastTrainer:
    """Trainer veloce per tuning (senza salvataggio checkpoints, logging minimo)."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer: SignLanguageTokenizer,
        device: torch.device,
        lr: float,
        weight_decay: float,
        label_smoothing: float,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9,
        )

        # Loss
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id, label_smoothing=label_smoothing
        )

        # BLEU
        if BLEU_AVAILABLE:
            self.bleu_metric = BLEUScore(n_gram=4)
        else:
            self.bleu_metric = None

    def train_epoch(self) -> float:
        """Train per 1 epoch, ritorna loss media."""
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            landmarks = batch["landmarks"].to(self.device)
            caption_ids = batch["caption_ids"].to(self.device)
            caption_mask = batch["caption_mask"].to(self.device)
            landmarks_mask = batch["landmarks_mask"].to(self.device)

            # Forward
            tgt_input = caption_ids[:, :-1]
            tgt_output = caption_ids[:, 1:]
            tgt_mask = caption_mask[:, 1:]

            logits = self.model(
                src=landmarks,
                tgt=tgt_input,
                src_key_padding_mask=~landmarks_mask,
                tgt_key_padding_mask=~caption_mask[:, :-1],
            )

            # Loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validazione, ritorna loss e BLEU."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_refs = []

        for batch in self.val_loader:
            landmarks = batch["landmarks"].to(self.device)
            caption_ids = batch["caption_ids"].to(self.device)
            caption_mask = batch["caption_mask"].to(self.device)
            landmarks_mask = batch["landmarks_mask"].to(self.device)

            # Forward
            tgt_input = caption_ids[:, :-1]
            tgt_output = caption_ids[:, 1:]

            logits = self.model(
                src=landmarks,
                tgt=tgt_input,
                src_key_padding_mask=~landmarks_mask,
                tgt_key_padding_mask=~caption_mask[:, :-1],
            )

            # Loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1)
            )
            total_loss += loss.item()

            # BLEU (sample 3 esempi per batch)
            if self.bleu_metric is not None and len(all_preds) < 30:
                for i in range(min(3, landmarks.size(0))):
                    pred_ids = self.model.generate(
                        landmarks[i : i + 1],
                        max_len=30,
                        sos_idx=self.tokenizer.sos_token_id,
                        eos_idx=self.tokenizer.eos_token_id,
                    )
                    pred_text = self.tokenizer.decode(pred_ids[0].cpu().tolist())
                    ref_text = self.tokenizer.decode(caption_ids[i].cpu().tolist())

                    all_preds.append(pred_text)
                    all_refs.append([ref_text])

        avg_loss = total_loss / len(self.val_loader)

        # Compute BLEU
        bleu_score = 0.0
        if self.bleu_metric is not None and len(all_preds) > 0:
            try:
                bleu_score = self.bleu_metric(all_preds, all_refs).item()
            except:
                bleu_score = 0.0

        return {"val_loss": avg_loss, "val_bleu": bleu_score}

    def train(self, num_epochs: int) -> Dict[str, float]:
        """Train per num_epochs, ritorna best val metrics."""
        best_val_loss = float("inf")
        best_val_bleu = 0.0

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_metrics = self.validate()

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
            if val_metrics["val_bleu"] > best_val_bleu:
                best_val_bleu = val_metrics["val_bleu"]

        return {"best_val_loss": best_val_loss, "best_val_bleu": best_val_bleu}


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """
    Optuna objective function.

    Suggerisce iperparametri, lancia training, ritorna metrica da ottimizzare.
    """
    # Suggest hyperparameters
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    num_encoder_layers = trial.suggest_int("num_encoder_layers", 2, 6)
    num_decoder_layers = trial.suggest_int("num_decoder_layers", 2, 6)
    nhead = trial.suggest_categorical("nhead", [4, 8])
    dim_feedforward = trial.suggest_categorical("dim_feedforward", [512, 1024, 2048])
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)

    # Device
    device = torch.device("cpu")  # Force CPU per evitare MPS bugs

    print(f"\n{'='*80}")
    print(f"Trial {trial.number}")
    print(f"{'='*80}")
    print(f"  lr={lr:.2e}, batch_size={batch_size}, d_model={d_model}")
    print(f"  encoder_layers={num_encoder_layers}, decoder_layers={num_decoder_layers}")
    print(f"  nhead={nhead}, dim_ff={dim_feedforward}, dropout={dropout:.3f}")
    print(f"  weight_decay={weight_decay:.2e}, label_smoothing={label_smoothing:.3f}")

    # Load tokenizer
    tokenizer = SignLanguageTokenizer.load("models/sign_to_text/tokenizer.json")

    # Load dataloaders con batch_size suggerito
    dataloaders = get_dataloaders(
        train_csv="results/utterances_analysis/train_split.csv",
        val_csv="results/utterances_analysis/val_split.csv",
        test_csv="results/utterances_analysis/test_split.csv",
        landmarks_dir="data/processed/sign_language_landmarks",
        tokenizer_path="models/sign_to_text/tokenizer.json",
        batch_size=batch_size,
        max_frames=200,
        max_caption_len=30,
        num_workers=0,  # 0 per evitare fork issues
    )

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # Opzionale: usa subset del training per velocizzare
    if args.subset_fraction < 1.0:
        train_size = int(len(train_loader.dataset) * args.subset_fraction)
        train_indices = list(range(train_size))
        train_subset = Subset(train_loader.dataset, train_indices)
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=train_loader.collate_fn,
        )

    # Create model con parametri suggeriti
    model = SignToTextTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    )

    # Create trainer
    trainer = FastTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        label_smoothing=label_smoothing,
    )

    # Train
    results = trainer.train(num_epochs=args.epochs)

    # Log metrics to trial
    trial.set_user_attr("best_val_loss", results["best_val_loss"])
    trial.set_user_attr("best_val_bleu", results["best_val_bleu"])

    print(f"\n  ✅ Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"  ✅ Best Val BLEU: {results['best_val_bleu']:.4f}\n")

    # Return metric to optimize
    if args.optimize == "bleu":
        return results["best_val_bleu"]  # Maximize
    else:
        return results["best_val_loss"]  # Minimize


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning con Optuna")

    # Tuning parameters
    parser.add_argument(
        "--n_trials", type=int, default=30, help="Numero di trials Optuna"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Epoche per trial (budget limitato)"
    )
    parser.add_argument(
        "--optimize",
        type=str,
        default="bleu",
        choices=["bleu", "loss"],
        help="Metrica da ottimizzare: bleu (maximize) o loss (minimize)",
    )
    parser.add_argument(
        "--subset_fraction",
        type=float,
        default=1.0,
        help="Frazione del train set da usare (0.0-1.0, default 1.0 = full)",
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="sign_to_text_tuning",
        help="Nome dello studio Optuna",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (es: sqlite:///optuna_study.db)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("🔧 HYPERPARAMETER TUNING WITH OPTUNA")
    print("=" * 80)
    print(f"  Trials: {args.n_trials}")
    print(f"  Epochs per trial: {args.epochs}")
    print(f"  Optimize: {args.optimize}")
    print(f"  Subset fraction: {args.subset_fraction}")
    print(f"  Study name: {args.study_name}")
    print("=" * 80 + "\n")

    # MLflow experiment
    mlflow.set_experiment(args.study_name)

    # Create Optuna study
    direction = "maximize" if args.optimize == "bleu" else "minimize"
    study = optuna.create_study(
        study_name=args.study_name,
        direction=direction,
        storage=args.storage,
        load_if_exists=True,
    )

    # MLflow callback
    mlflc = MLflowCallback(
        tracking_uri=mlflow.get_tracking_uri(),
        metric_name=f"val_{args.optimize}",
        mlflow_kwargs={"nested": True},
    )

    # Optimize
    study.optimize(
        lambda trial: objective(trial, args),
        n_trials=args.n_trials,
        callbacks=[mlflc],
        show_progress_bar=True,
    )

    # Best trial
    print("\n" + "=" * 80)
    print("🏆 BEST TRIAL")
    print("=" * 80)
    print(f"  Trial number: {study.best_trial.number}")
    print(f"  Best value ({args.optimize}): {study.best_trial.value:.4f}")
    print(f"\n  Best hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")

    # User attributes
    if hasattr(study.best_trial, "user_attrs"):
        print(f"\n  Additional metrics:")
        for key, value in study.best_trial.user_attrs.items():
            print(f"    {key}: {value:.4f}")

    # Save best params to JSON
    best_params_path = Path("results") / "best_hyperparameters.json"
    best_params_path.parent.mkdir(parents=True, exist_ok=True)

    best_params = {
        "trial_number": study.best_trial.number,
        "best_value": study.best_trial.value,
        "optimize_metric": args.optimize,
        "params": study.best_trial.params,
        "user_attrs": (
            dict(study.best_trial.user_attrs)
            if hasattr(study.best_trial, "user_attrs")
            else {}
        ),
    }

    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\n  💾 Best parameters saved to: {best_params_path}")
    print("=" * 80 + "\n")

    # Optuna visualization (importanza parametri)
    try:
        from optuna.visualization import plot_param_importances
        import plotly

        fig = plot_param_importances(study)
        fig_path = Path("results") / "param_importances.html"
        plotly.offline.plot(fig, filename=str(fig_path), auto_open=False)
        print(f"  📊 Parameter importances plot saved to: {fig_path}\n")
    except:
        pass


if __name__ == "__main__":
    main()
