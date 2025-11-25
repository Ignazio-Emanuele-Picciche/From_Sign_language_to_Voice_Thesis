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
    get_cosine_schedule_with_warmup,  # <--- CAMBIATO IN COSINE
)
import torch.optim as optim
import evaluate
import random
import mlflow
import shutil


# Evita frammentazione memoria
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# python train_sonar_slt_paper_params.py \
#   --train_features_dir "data/features/train" \
#   --train_manifest "data/manifests/train.tsv" \
#   --val_features_dir "data/features/val" \
#   --val_manifest "data/manifests/val.tsv" \
#   --output_dir "models/run_paper_params" \
#   --lang "eng_Latn" \
#   --batch_size 32 \
#   --epochs 200 \
#   --lr 1e-3 \
#   --patience 15 \
#   --random_seed 42


# ==========================================
# 0. SETUP
# ==========================================
def set_seed(seed):
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

        try:
            df = pd.read_csv(manifest_path, sep="\t")
            if "text" not in df.columns or "id" not in df.columns:
                df = pd.read_csv(
                    manifest_path,
                    sep="\t",
                    header=None,
                    names=["id", "duration", "text"],
                )
        except Exception as e:
            raise ValueError(f"Errore Manifest {split_name}: {e}")

        df["id"] = df["id"].astype(str)
        df["text"] = df["text"].astype(str)
        self.ids = []
        self.texts = []

        print(f"🔍 Check {split_name}...")
        iterator = (
            tqdm(df.iterrows(), total=len(df)) if len(df) > 100 else df.iterrows()
        )
        for _, row in iterator:
            vid_id = row["id"].strip()
            if (self.features_dir / f"{vid_id}.npy").exists():
                self.ids.append(vid_id)
                self.texts.append(row["text"])

        print(f"✅ {split_name}: {len(self.ids)} samples.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        video_id = self.ids[idx]
        text = self.texts[idx]

        feat = np.load(self.features_dir / f"{video_id}.npy")
        video_tensor = torch.from_numpy(feat).float()

        self.tokenizer.src_lang = "eng_Latn"
        self.tokenizer.tgt_lang = self.tgt_lang
        labels = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        # Label smoothing gestisce il padding internamente o lo ignoriamo nella loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"input_features": video_tensor, "labels": labels, "text": text}


# ==========================================
# 2. MODELLO (Adapter 0.3 Dropout)
# ==========================================
class SonarSignModel(nn.Module):
    def __init__(
        self,
        pretrained_model="facebook/nllb-200-distilled-600M",
        feature_dim=768,
        freeze_decoder=False,
    ):
        super().__init__()
        print(f"🏗️  Init NLLB ({pretrained_model})...")
        self.nllb = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)

        if freeze_decoder:
            print("❄️  FREEZING ATTIVO.")
            for param in self.nllb.parameters():
                param.requires_grad = False

        hidden_dim = self.nllb.config.d_model

        # PAPER: Dropout probability 0.3
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, input_features, labels=None):
        inputs_embeds = self.adapter(input_features)
        attention_mask = torch.ones(
            inputs_embeds.shape[:2], device=inputs_embeds.device
        )

        # CORREZIONE: Passiamo 'labels' a NLLB.
        # Questo permette al modello di creare i 'decoder_input_ids' necessari per il training.
        # NLLB calcolerà anche una sua 'loss', ma noi useremo solo i 'logits'.
        outputs = self.nllb(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
        )
        return outputs

    def generate(self, input_features, tokenizer, max_new_tokens=60, num_beams=5):
        inputs_embeds = self.adapter(input_features)
        att_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device)
        forced_bos = tokenizer.convert_tokens_to_ids(tokenizer.tgt_lang)

        # PAPER: Number of beams 5
        gen_ids = self.nllb.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=att_mask,
            forced_bos_token_id=forced_bos,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        return gen_ids


# ==========================================
# 3. UTILITY
# ==========================================
def save_checkpoint(model, optimizer, scheduler, epoch, args, filename):
    path = Path(args.output_dir) / filename
    tmp = Path("/tmp") / filename
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "sched": scheduler.state_dict() if scheduler else None,
        "args": args,
    }
    try:
        torch.save(state, tmp)
        shutil.move(str(tmp), str(path))
        print(f"💾 Saved: {path}")
    except Exception as e:
        print(f"⚠️ Save failed: {e}")


def calculate_metrics(preds, refs):
    from sacrebleu.metrics import BLEU
    import evaluate

    res = {}
    refs_list = [refs]

    res["BLEU-4"] = BLEU(max_ngram_order=4).corpus_score(preds, refs_list).score

    try:
        rouge = evaluate.load("rouge")
        res["ROUGE-L"] = (
            rouge.compute(predictions=preds, references=refs)["rougeL"] * 100
        )
    except:
        res["ROUGE-L"] = 0.0
    return res


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
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
            if self.counter >= self.patience:
                self.early_stop = True
            return False
        else:
            self.best_loss = val_loss
            self.counter = 0
            return True


# ==========================================
# 4. TRAIN LOOP
# ==========================================
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    mlflow.set_experiment("SONAR_SLT_PaperParams")
    with mlflow.start_run():
        mlflow.log_params(vars(args))

        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        train_ds = SignTranslationDataset(
            args.train_features_dir, args.train_manifest, tokenizer, split_name="TRAIN"
        )
        val_ds = SignTranslationDataset(
            args.val_features_dir, args.val_manifest, tokenizer, split_name="VAL"
        )

        # Aumentiamo workers per A100
        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        model = SonarSignModel(freeze_decoder=False).to(device)

        # PAPER: Optimizer momentum beta1=0.9, beta2=0.95, Weight decay 1e-1
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(
            trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
        )

        scaler = torch.amp.GradScaler("cuda")

        # PAPER: Cosine Decay Schedule with Warmup
        # Warmup epochs: 10 (lo convertiamo in steps)
        num_steps = len(train_dl) * args.epochs
        num_warmup_steps = len(train_dl) * 10  # 10 epoche di warmup

        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_steps
        )

        # PAPER: Label Smoothing 0.2
        loss_fct = nn.CrossEntropyLoss(label_smoothing=0.2, ignore_index=-100)

        start_epoch = 0
        # Resume logic (semplificata)
        if args.resume_from and os.path.exists(args.resume_from):
            ckpt = torch.load(args.resume_from, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            # Non carichiamo optim/sched se cambiamo iperparametri
            start_epoch = ckpt["epoch"] + 1
            print("♻️ Resumed model weights only (Hyperparams changed).")

        early_stopper = EarlyStopping(patience=args.patience)

        print(f"🔥 Start Training: {len(train_ds)} samples. Params aligned with Paper.")

        for epoch in range(start_epoch, args.epochs):
            model.train()
            total_loss = 0
            pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")

            for batch in pbar:
                inputs = batch["input_features"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    # CORREZIONE: Passiamo le labels vere al modello!
                    # Il modello serve le labels per fare Teacher Forcing nel decoder.
                    outputs = model(inputs, labels=labels)

                    logits = outputs.logits

                    # Calcolo Loss Manuale (Label Smoothing)
                    # Usiamo i logits usciti dal modello e le nostre labels
                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

                scaler.scale(loss).backward()

                # PAPER: Gradient Clipping 1.0
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                total_loss += loss.item()
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.6f}",
                    }
                )

            avg_train_loss = total_loss / len(train_dl)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch + 1)

            # VALIDATION
            if (epoch + 1) % args.val_every == 0:
                model.eval()
                val_loss = 0
                all_preds, all_targets = [], []

                print("🔍 Validating...")
                with torch.no_grad():
                    for batch in tqdm(val_dl):
                        inputs = batch["input_features"].to(device)
                        labels = batch["labels"].to(device)

                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            outputs = model(inputs, labels=None)
                            logits = outputs.logits
                            loss = loss_fct(
                                logits.view(-1, logits.size(-1)), labels.view(-1)
                            )
                            val_loss += loss.item()

                        # PAPER: Beam size 5
                        gen_ids = model.generate(inputs, tokenizer, num_beams=5)
                        decoded = tokenizer.batch_decode(
                            gen_ids, skip_special_tokens=True
                        )
                        all_preds.extend(decoded)
                        all_targets.extend(batch["text"])

                avg_val_loss = val_loss / len(val_dl)
                metrics = calculate_metrics(all_preds, all_targets)

                print(
                    f"📊 Val Loss: {avg_val_loss:.4f} | BLEU-4: {metrics['BLEU-4']:.2f}"
                )
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch + 1)
                for k, v in metrics.items():
                    mlflow.log_metric(k, v, step=epoch + 1)

                if len(all_preds) > 0:
                    print(f"Ex: {all_preds[0]}")

                if early_stopper(avg_val_loss):
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, args, "checkpoint_best.pth"
                    )
                    print("🏆 Best Model Saved.")

                save_checkpoint(
                    model, optimizer, scheduler, epoch, args, "checkpoint_last.pth"
                )

                if early_stopper.early_stop:
                    print("🛑 Early Stopping.")
                    break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features_dir", type=str, required=True)
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_features_dir", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    # PAPER: Batch size 32 (o quello che regge la A100)
    parser.add_argument("--batch_size", type=int, default=32)
    # PAPER: 200 Epochs (noi usiamo early stopping)
    parser.add_argument("--epochs", type=int, default=200)
    # LR 1e-3 per adapter
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lang", type=str, default="eng_Latn")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--val_every", type=int, default=5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
