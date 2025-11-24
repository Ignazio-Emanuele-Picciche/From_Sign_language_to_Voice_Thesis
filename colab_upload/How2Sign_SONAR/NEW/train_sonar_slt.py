import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import torch.optim as optim
import evaluate
import random
import mlflow


# python train_sonar_slt.py \
#   --train_features_dir "data/features/train" \
#   --train_manifest "data/manifests/train.tsv" \
#   --val_features_dir "data/features/val" \
#   --val_manifest "data/manifests/val.tsv" \
#   --output_dir "models/run_english_final" \
#   --lang "eng_Latn" \
#   --batch_size 8 \
#   --epochs 2 \
#   --save_every 2 \
#   --val_every 2 \
#   --patience 5 \
#   --random_seed 42


# ==========================================
# 0. FUNZIONE PER LA RIPRODUCIBILITÀ
# ==========================================
def set_seed(seed):
    """Fissa il seed per riproducibilità totale."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🌱 Random Seed fissato a: {seed}")


# ==========================================
# 1. DATASET
# ==========================================
class SignTranslationDataset(Dataset):
    def __init__(
        self,
        features_dir,
        manifest_path,
        tokenizer,
        max_length=128,
        tgt_lang="eng_Latn",
        split_name="data",
    ):
        self.features_dir = Path(features_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tgt_lang = tgt_lang
        self.split_name = split_name

        try:
            df = pd.read_csv(manifest_path, sep="\t")
            if "text" not in df.columns or "id" not in df.columns:
                print(
                    f"⚠️ [{split_name}] Header non trovato, ricarico con nomi manuali..."
                )
                df = pd.read_csv(
                    manifest_path,
                    sep="\t",
                    header=None,
                    names=["id", "duration", "text"],
                )
        except Exception as e:
            raise ValueError(f"Errore lettura Manifest {split_name}: {e}")

        df["id"] = df["id"].astype(str)
        df["text"] = df["text"].astype(str)

        self.ids = []
        self.texts = []

        print(f"🔍 Verifica file per split: {split_name}...")
        missing_count = 0
        iterator = (
            tqdm(df.iterrows(), total=len(df), desc=f"Check {split_name}")
            if len(df) > 100
            else df.iterrows()
        )

        for _, row in iterator:
            vid_id = row["id"].strip()
            npy_path = self.features_dir / f"{vid_id}.npy"

            if npy_path.exists():
                self.ids.append(vid_id)
                self.texts.append(row["text"])
            else:
                missing_count += 1

        print(f"✅ Dataset {split_name} pronto: {len(self.ids)} video validi.")
        if missing_count > 0:
            print(
                f"⚠️ {missing_count} video ignorati in {split_name} (file .npy mancante)."
            )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        video_id = self.ids[idx]
        text = self.texts[idx]

        feature_path = self.features_dir / f"{video_id}.npy"
        video_features = np.load(feature_path)
        video_tensor = torch.from_numpy(video_features).float()

        self.tokenizer.src_lang = "eng_Latn"
        self.tokenizer.tgt_lang = self.tgt_lang

        labels = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"input_features": video_tensor, "labels": labels, "text": text}


# ==========================================
# 2. MODELLO
# ==========================================
class SonarSignModel(nn.Module):
    def __init__(
        self, pretrained_model="facebook/nllb-200-distilled-600M", feature_dim=768
    ):
        super().__init__()
        print(f"🏗️  Inizializzazione SONAR/NLLB ({pretrained_model})...")
        self.nllb = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
        hidden_dim = self.nllb.config.d_model

        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, input_features, labels=None):
        inputs_embeds = self.adapter(input_features)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], device=inputs_embeds.device
        )
        outputs = self.nllb(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def generate(self, input_features, tokenizer, max_new_tokens=60):
        inputs_embeds = self.adapter(input_features)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], device=inputs_embeds.device
        )

        forced_bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.tgt_lang)

        generated_ids = self.nllb.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
            num_beams=4,
        )
        return generated_ids


# ==========================================
# 3. UTILITY
# ==========================================
def save_checkpoint(
    model, optimizer, scheduler, scaler, epoch, args, filename="checkpoint_latest.pth"
):
    save_path = Path(args.output_dir) / filename
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict(),
        "args": args,
    }
    torch.save(state, save_path)
    print(f"💾 Checkpoint salvato: {save_path}")


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    if not os.path.exists(path):
        print(f"⚠️ Nessun checkpoint trovato in {path}, inizio da zero.")
        return 0

    print(f"♻️ Caricamento checkpoint da {path}...")
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    print(f"✅ Ripristinato! Si riparte dall'epoca {start_epoch}")
    return start_epoch


def calculate_metrics(predictions, references):
    print("📊 Calcolo metriche...")
    results = {}
    from sacrebleu.metrics import BLEU
    import evaluate

    sacrebleu_refs = [references]

    b1 = BLEU(max_ngram_order=1)
    results["BLEU-1"] = b1.corpus_score(predictions, sacrebleu_refs).score

    b2 = BLEU(max_ngram_order=2)
    results["BLEU-2"] = b2.corpus_score(predictions, sacrebleu_refs).score

    b3 = BLEU(max_ngram_order=3)
    results["BLEU-3"] = b3.corpus_score(predictions, sacrebleu_refs).score

    b4 = BLEU(max_ngram_order=4)
    results["BLEU-4"] = b4.corpus_score(predictions, sacrebleu_refs).score

    try:
        rouge_metric = evaluate.load("rouge")
        rouge = rouge_metric.compute(predictions=predictions, references=references)
        results["ROUGE-L"] = rouge["rougeL"] * 100
    except Exception as e:
        print(f"⚠️ Errore ROUGE: {e}")
        results["ROUGE-L"] = 0.0

    try:
        bleurt_metric = evaluate.load("bleurt", config_name="bleurt-tiny-128")
        bleurt_res = bleurt_metric.compute(
            predictions=predictions, references=references
        )
        results["BLEURT"] = np.mean(bleurt_res["scores"])
    except Exception:
        pass

    return results


class EarlyStopping:
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return True
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"⏳ EarlyStopping counter: {self.counter} su {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_loss = val_loss
            self.counter = 0
            return True


# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training su device: {device}")

    # --- MLFLOW SETUP ---
    mlflow.set_experiment("SONAR_SLT_Experiment")

    # Inizia il tracciamento MLflow
    with mlflow.start_run():
        # Logghiamo tutti gli iperparametri
        mlflow.log_params(vars(args))
        print("📊 MLflow run avviata. Parametri loggati.")

        model_name = "facebook/nllb-200-distilled-600M"
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        print("\n📂 Caricamento TRAIN SET...")
        train_dataset = SignTranslationDataset(
            features_dir=args.train_features_dir,
            manifest_path=args.train_manifest,
            tokenizer=tokenizer,
            tgt_lang=args.lang,
            split_name="TRAIN",
        )

        print("\n📂 Caricamento VAL SET...")
        val_dataset = SignTranslationDataset(
            features_dir=args.val_features_dir,
            manifest_path=args.val_manifest,
            tokenizer=tokenizer,
            tgt_lang=args.lang,
            split_name="VAL",
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
        )

        model = SonarSignModel(pretrained_model=model_name).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        scaler = torch.amp.GradScaler("cuda")

        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=100, num_training_steps=total_steps
        )

        start_epoch = 0
        if args.resume_from:
            start_epoch = load_checkpoint(
                args.resume_from, model, optimizer, scheduler, scaler
            )

        early_stopper = EarlyStopping(patience=args.patience, min_delta=0.001)

        print(
            f"🔥 Inizio Training: {len(train_dataset)} train samples, {len(val_dataset)} val samples."
        )

        for epoch in range(start_epoch, args.epochs):
            # 1. TRAIN
            model.train()
            total_loss = 0
            progress_bar = tqdm(
                train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"
            )

            for batch in progress_bar:
                inputs = batch["input_features"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs, labels=labels)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = total_loss / len(train_loader)
            print(f"\n📉 Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")

            # MLFLOW LOG: Train Loss
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch + 1)

            # 2. VALIDATION
            if (epoch + 1) % args.val_every == 0:
                model.eval()
                val_loss = 0
                all_preds = []
                all_targets = []

                print(f"🔍 Validazione su {len(val_dataset)} campioni...")
                with torch.no_grad():
                    for i, batch in enumerate(tqdm(val_loader, desc="Validation")):
                        inputs = batch["input_features"].to(device)
                        labels = batch["labels"].to(device)
                        texts = batch["text"]

                        with torch.amp.autocast("cuda"):
                            outputs = model(inputs, labels=labels)
                            val_loss += outputs.loss.item()

                        tokenizer.tgt_lang = args.lang
                        gen_ids = model.generate(inputs, tokenizer)
                        decoded_preds = tokenizer.batch_decode(
                            gen_ids, skip_special_tokens=True
                        )
                        all_preds.extend(decoded_preds)
                        all_targets.extend(texts)

                avg_val_loss = val_loss / len(val_loader)

                metrics = calculate_metrics(all_preds, all_targets)

                print(f"\n📊 REPORT VALIDAZIONE (Epoch {epoch+1}):")
                print(f"   Loss Val: {avg_val_loss:.4f}")

                # MLFLOW LOG: Val Loss & Metrics
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch + 1)

                for k, v in metrics.items():
                    print(f"   {k}: {v:.2f}")
                    mlflow.log_metric(k, v, step=epoch + 1)

                if len(all_preds) > 0:
                    print("\n👀 Esempio:")
                    print(f"   ✅: {all_targets[0]}")
                    print(f"   🤖: {all_preds[0]}")
                    print("-" * 40)

                # Early Stopping & Best Model Logic
                is_best = early_stopper(avg_val_loss)

                if is_best:
                    print(
                        f"🏆 Nuova Best Val Loss: {avg_val_loss:.4f}. Salvataggio checkpoint_best.pth..."
                    )
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        args,
                        filename="checkpoint_best.pth",
                    )
                    # Opzionale: Loggare che abbiamo trovato un best model su MLflow
                    mlflow.log_metric(
                        "best_val_loss_so_far", avg_val_loss, step=epoch + 1
                    )

                if (epoch + 1) % args.save_every == 0:
                    save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        args,
                        filename=f"checkpoint_epoch_{epoch+1}.pth",
                    )
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    args,
                    filename="checkpoint_last.pth",
                )

                if early_stopper.early_stop:
                    print(
                        f"🛑 Early Stopping attivato! Nessun miglioramento per {args.patience} controlli consecutivi."
                    )
                    break

    print(f"🎉 Training completato! Output in {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_features_dir", type=str, required=True)
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_features_dir", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)

    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lang", type=str, default="eng_Latn")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Quante validazioni aspettare prima di stoppare",
    )

    args = parser.parse_args()

    set_seed(args.random_seed)

    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
