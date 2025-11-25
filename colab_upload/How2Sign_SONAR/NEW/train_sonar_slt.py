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
)
import torch.optim as optim
import evaluate
import random
import mlflow
import shutil


# !python train_sonar_slt.py \
#   --train_features_dir "features/train" \
#   --train_manifest "manifests/train.tsv" \
#   --val_features_dir "features/val" \
#   --val_manifest "manifests/val.tsv" \
#   --output_dir "models/run_paper_params" \
#   --lang "eng_Latn" \
#   --batch_size 32 \
#   --epochs 200 \
#   --lr 1e-3 \
#   --patience 8 \
#   --random_seed 42 \
# --max_train_samples 5000


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
        max_samples=None,
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

        if max_samples is not None and max_samples > 0:
            print(f"✂️  Limitando {split_name} a {max_samples} campioni.")
            df = df.head(max_samples)

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
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {"input_features": video_tensor, "labels": labels, "text": text}


# ==========================================
# 2. MODELLO (PAPER CONFIG)
# ==========================================
class SonarSignModel(nn.Module):
    def __init__(
        self,
        pretrained_model="facebook/nllb-200-distilled-600M",
        feature_dim=768,
        freeze_decoder=True,
    ):
        super().__init__()
        print(f"🏗️  Init NLLB ({pretrained_model})...")
        self.nllb = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)

        if freeze_decoder:
            print("❄️  FREEZING ATTIVO (Training sicuro per High LR).")
            for param in self.nllb.parameters():
                param.requires_grad = False
            self.nllb.gradient_checkpointing_enable()
            print("🛡️  Gradient Checkpointing ATTIVO")

        hidden_dim = self.nllb.config.d_model

        # PAPER: Dropout 0.3
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, input_features, labels=None):
        inputs_embeds = self.adapter(input_features)
        att_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device)
        outputs = self.nllb(
            inputs_embeds=inputs_embeds,
            attention_mask=att_mask,
            labels=labels,
            use_cache=False,
        )
        return outputs

    def generate(self, input_features, tokenizer, max_new_tokens=60, num_beams=5):
        inputs_embeds = self.adapter(input_features)
        att_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device)

        target_lang = getattr(tokenizer, "tgt_lang", "eng_Latn")
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

        gen_ids = self.nllb.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=att_mask,
            forced_bos_token_id=forced_bos_token_id,
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
    print("📊 Calcolo metriche complete...")
    results = {}
    from sacrebleu.metrics import BLEU
    import evaluate

    sacrebleu_refs = [refs]

    # BLEU 1-4
    b1 = BLEU(max_ngram_order=1)
    results["BLEU-1"] = b1.corpus_score(preds, sacrebleu_refs).score
    b2 = BLEU(max_ngram_order=2)
    results["BLEU-2"] = b2.corpus_score(preds, sacrebleu_refs).score
    b3 = BLEU(max_ngram_order=3)
    results["BLEU-3"] = b3.corpus_score(preds, sacrebleu_refs).score
    b4 = BLEU(max_ngram_order=4)
    results["BLEU-4"] = b4.corpus_score(preds, sacrebleu_refs).score

    try:
        rouge = evaluate.load("rouge")
        results["ROUGE-L"] = (
            rouge.compute(predictions=preds, references=refs)["rougeL"] * 100
        )
    except:
        results["ROUGE-L"] = 0.0

    # BLEURT Opzionale
    try:
        bleurt = evaluate.load("bleurt", config_name="bleurt-tiny-128")
        scores = bleurt.compute(predictions=preds, references=refs)["scores"]
        results["BLEURT"] = np.mean(scores)
    except:
        pass

    return results


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
            args.train_features_dir,
            args.train_manifest,
            tokenizer,
            split_name="TRAIN",
            max_samples=args.max_train_samples,
        )
        val_ds = SignTranslationDataset(
            args.val_features_dir,
            args.val_manifest,
            tokenizer,
            split_name="VAL",
            max_samples=args.max_val_samples,
        )

        # --- DATALOADER OTTIMIZZATI (RAM 167GB) ---
        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=96,
            prefetch_factor=600,  # High Buffer
            pin_memory=True,
            persistent_workers=True,
        )

        val_bs = max(1, args.batch_size // 4)
        val_dl = DataLoader(
            val_ds,
            batch_size=val_bs,
            shuffle=False,
            num_workers=16,
            prefetch_factor=200,
            pin_memory=True,
            persistent_workers=True,
        )

        # PAPER: Freeze=True è essenziale con LR=1e-2
        model = SonarSignModel(freeze_decoder=False).to(device)

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())

        # PAPER: Weight Decay 1e-1, Betas (0.9, 0.95)
        optimizer = optim.AdamW(
            trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
        )

        scaler = torch.amp.GradScaler("cuda")

        # --- SCHEDULER (Paper: Cosine Decay con Min LR) ---
        num_steps = len(train_dl) * args.epochs
        num_warmup_steps = int(num_steps * 0.1)  # 10% Warmup

        # Scheduler Principale (Cosine che scende fino a 1e-4)
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps - num_warmup_steps, eta_min=1e-4  # Minimum LR
        )

        # Warmup Scheduler (Lineare da quasi-0 a Peak)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=num_warmup_steps
        )

        # Sequential LR
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[num_warmup_steps],
        )

        # PAPER: Label Smoothing 0.2
        loss_fct = nn.CrossEntropyLoss(label_smoothing=0.2, ignore_index=-100)

        start_epoch = 0
        if args.resume_from and os.path.exists(args.resume_from):
            print("♻️ Loading Checkpoint...")
            try:
                ckpt = torch.load(args.resume_from, map_location="cpu")
                model.load_state_dict(ckpt["model"], strict=False)
            except:
                print("⚠️ Warning: Could not load full state.")

        early_stopper = EarlyStopping(patience=args.patience)

        print(f"🔥 Start Training: {len(train_ds)} samples. Steps: {num_steps}")

        for epoch in range(start_epoch, args.epochs):
            model.train()
            total_loss = 0
            pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")

            for batch in pbar:
                inputs = batch["input_features"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(inputs, labels=labels)
                    logits = outputs.logits
                    loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # PAPER: Gradient Clipping 1.0
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
                            outputs = model(inputs, labels=labels)
                            logits = outputs.logits
                            loss = loss_fct(
                                logits.view(-1, logits.size(-1)), labels.view(-1)
                            )
                            val_loss += loss.item()

                        tokenizer.src_lang = "eng_Latn"
                        tokenizer.tgt_lang = args.lang

                        # PAPER: Beam Search 5
                        gen_ids = model.generate(inputs, tokenizer, num_beams=5)
                        decoded = tokenizer.batch_decode(
                            gen_ids, skip_special_tokens=True
                        )
                        all_preds.extend(decoded)
                        all_targets.extend(batch["text"])

                avg_val_loss = val_loss / len(val_dl)
                metrics = calculate_metrics(all_preds, all_targets)

                print(f"📊 Report (Ep {epoch+1}): Val Loss: {avg_val_loss:.4f}")
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch + 1)
                for k, v in metrics.items():
                    print(f"   {k}: {v:.2f}")
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)

    # PAPER: Peak LR 1e-2 (Usato come max)
    parser.add_argument("--lr", type=float, default=1e-2)

    parser.add_argument("--lang", type=str, default="eng_Latn")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)

    args = parser.parse_args()

    set_seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
