"""
################################################################################
# TRAIN SONAR SLT - Pipeline di Addestramento
################################################################################
#
# OBIETTIVO:
# Fine-tuning del modello NLLB-200 per tradurre video (Linguaggio dei Segni) in testo.
#
# INPUT:
# - Feature visive .npy (estratte con SignHiera/ResNet)
# - Manifest .tsv (ID video e traduzione testuale)
#
# MODALITÀ DI FUNZIONAMENTO:
# 1. Standard (Seq2Seq):
#    Il decoder riceve l'intera sequenza di frame (300+ token).
#    Migliore per traduzioni lunghe e dettagliate.
#
# 2. SONAR Mode (--sonar_pooling):
#    Simula l'approccio del paper SONAR. L'output dell'encoder viene
#    schiacciato (Mean Pooling) in un unico vettore prima di passare al Decoder.
#    Più leggero in memoria, approccio "Vector-to-Text".
#
# PARAMETRI CHIAVE:
# --freeze_encoder: Congela NLLB Encoder (utile in modalità pooling)
# --freeze_decoder: Congela NLLB Decoder (utile per trainare solo l'adapter)
#
################################################################################
"""

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
# 1. DATASET (INPUT SEMPRE .NPY)
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

        # Carichiamo la sequenza video (es. 128x768 o 300x768)
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
# 2. MODELLO (SONAR-LIKE POOLING)
# ==========================================
class SonarSignModel(nn.Module):
    def __init__(
        self,
        pretrained_model="facebook/nllb-200-distilled-600M",
        feature_dim=768,
        freeze_encoder=False,
        freeze_decoder=False,
        sonar_pooling=False,  # <--- SE TRUE, ACCENDE LA LOGICA "VETTORE UNICO"
    ):
        super().__init__()
        print(f"🏗️  Init NLLB ({pretrained_model})")
        print(
            f"🔮  SONAR Pooling Mode: {'ON (Vector-to-Text)' if sonar_pooling else 'OFF (Seq2Seq)'}"
        )

        self.nllb = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
        self.sonar_pooling = sonar_pooling

        # 1. Gestione ENCODER
        if freeze_encoder:
            print("❄️  FREEZING ENCODER (NLLB).")
            for param in self.nllb.get_encoder().parameters():
                param.requires_grad = False
        else:
            print("🔥  ENCODER TRAINABLE.")

        # 2. Gestione DECODER
        if freeze_decoder:
            print("❄️  FREEZING DECODER (NLLB).")
            for param in self.nllb.get_decoder().parameters():
                param.requires_grad = False
        else:
            print("🔥  DECODER TRAINABLE.")

        # Gradient Checkpointing (Se almeno una parte è trainabile)
        if not (freeze_encoder and freeze_decoder):
            self.nllb.gradient_checkpointing_enable()
            print("🛡️  Gradient Checkpointing ATTIVO")

        hidden_dim = self.nllb.config.d_model

        # Adapter: Porta le feature video (768) alla dimensione di NLLB (1024)
        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, input_features, labels=None):
        # 1. Adattamento Video -> Dimensione Modello
        inputs_embeds = self.adapter(input_features)

        # 2. Passaggio attraverso Encoder NLLB (che elabora la sequenza)
        # Usiamo .encoder() direttamente per avere accesso agli hidden states
        encoder_outputs = self.nllb.model.encoder(
            inputs_embeds=inputs_embeds, return_dict=True
        )
        last_hidden_state = (
            encoder_outputs.last_hidden_state
        )  # Shape: (Batch, Time, Dim)

        # 3. LOGICA SONAR: Comprimere la sequenza in un vettore
        if self.sonar_pooling:
            # Mean Pooling sulla dimensione temporale (dim=1)
            # Trasforma (Batch, 300, 1024) -> (Batch, 1, 1024)
            pooled_state = torch.mean(last_hidden_state, dim=1, keepdim=True)

            # Normalizzazione (spesso aiuta la convergenza negli spazi di embedding)
            encoder_hidden_states = torch.nn.functional.normalize(pooled_state, dim=-1)
        else:
            # Seq2Seq Classico: passiamo tutta la sequenza
            encoder_hidden_states = last_hidden_state

        # 4. Passaggio al Decoder
        # Dobbiamo passare encoder_hidden_states manualmente
        outputs = self.nllb(
            encoder_outputs=None,  # Disabilitiamo il passaggio automatico
            encoder_hidden_states=encoder_hidden_states,  # Usiamo il nostro stato (pooled o full)
            labels=labels,
            use_cache=False,
        )
        return outputs

    def generate(self, input_features, tokenizer, max_new_tokens=60, num_beams=5):
        inputs_embeds = self.adapter(input_features)

        # Encoding
        encoder_outputs = self.nllb.model.encoder(
            inputs_embeds=inputs_embeds, return_dict=True
        )
        last_hidden_state = encoder_outputs.last_hidden_state

        # Pooling per generazione
        if self.sonar_pooling:
            pooled_state = torch.mean(last_hidden_state, dim=1, keepdim=True)
            encoder_hidden_states = torch.nn.functional.normalize(pooled_state, dim=-1)
        else:
            encoder_hidden_states = last_hidden_state

        target_lang = getattr(tokenizer, "tgt_lang", "eng_Latn")
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

        # Generating
        gen_ids = self.nllb.generate(
            encoder_hidden_states=encoder_hidden_states,  # <--- Cruciale
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

    try:
        bleurt = evaluate.load("bleurt", config_name="bleurt-tiny-128")
        scores = bleurt.compute(predictions=preds, references=refs)["scores"]
        results["BLEURT"] = np.mean(scores)
    except:
        pass

    return results


def log_predictions(preds, targets, video_ids, epoch, args):
    log_path = Path(args.output_dir) / "validation_log.txt"
    total_samples = len(preds)

    fixed_indices = list(range(min(5, total_samples)))
    remaining_indices = list(range(5, total_samples))
    random_indices = []
    if remaining_indices:
        random_indices = random.sample(
            remaining_indices, min(5, len(remaining_indices))
        )
    indices_to_log = fixed_indices + random_indices

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"\n{'='*20} EPOCH {epoch+1} {'='*20}\n")
        for idx in indices_to_log:
            is_fixed = "FIXED" if idx in fixed_indices else "RANDOM"
            vid_id = video_ids[idx] if idx < len(video_ids) else "N/A"
            f.write(f"[{is_fixed}] ID: {vid_id}\n")
            f.write(f"GT  : {targets[idx]}\n")
            f.write(f"PRED: {preds[idx]}\n")
            f.write("-" * 40 + "\n")
    print(f"📝 Esempi salvati in: {log_path}")


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

    mlflow.set_experiment("SONAR_SLT_Pooling")
    with mlflow.start_run():
        mlflow.log_params(vars(args))

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

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

        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=8,  # 96 è eccessivo su molte macchine, riporto a 8/16 per sicurezza
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
        )

        val_bs = max(1, args.batch_size // 4)
        val_dl = DataLoader(
            val_ds,
            batch_size=val_bs,
            shuffle=False,
            num_workers=4,
            prefetch_factor=4,
            pin_memory=True,
            persistent_workers=True,
        )

        # --- MODELLO INIZIALIZZATO CON POOLING ---
        model = SonarSignModel(
            pretrained_model=args.model_name,
            freeze_encoder=args.freeze_encoder,
            freeze_decoder=args.freeze_decoder,
            sonar_pooling=args.sonar_pooling,  # <--- Passaggio del flag
        ).to(device)

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"📉 Parametri trainabili: {num_params}")

        optimizer = optim.AdamW(
            trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
        )

        scaler = torch.amp.GradScaler("cuda")

        num_steps = len(train_dl) * args.epochs
        num_warmup_steps = int(num_steps * 0.1)

        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps - num_warmup_steps, eta_min=1e-4
        )
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=num_warmup_steps
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[num_warmup_steps],
        )

        loss_fct = nn.CrossEntropyLoss(label_smoothing=0.2, ignore_index=-100)
        early_stopper = EarlyStopping(patience=args.patience)

        print(f"🔥 Start Training: {len(train_ds)} samples. Steps: {num_steps}")

        for epoch in range(args.epochs):
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
                all_video_ids = []

                print("🔍 Validating...")
                with torch.no_grad():
                    for batch_idx, batch in enumerate(tqdm(val_dl)):
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

                        gen_ids = model.generate(inputs, tokenizer, num_beams=5)
                        decoded = tokenizer.batch_decode(
                            gen_ids, skip_special_tokens=True
                        )
                        all_preds.extend(decoded)
                        all_targets.extend(batch["text"])

                        start_idx = batch_idx * val_bs
                        end_idx = start_idx + len(decoded)
                        batch_ids = val_ds.ids[start_idx:end_idx]
                        all_video_ids.extend(batch_ids)

                avg_val_loss = val_loss / len(val_dl)
                metrics = calculate_metrics(all_preds, all_targets)

                print(f"📊 Report (Ep {epoch+1}): Val Loss: {avg_val_loss:.4f}")
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch + 1)
                for k, v in metrics.items():
                    print(f"   {k}: {v:.2f}")
                    mlflow.log_metric(k, v, step=epoch + 1)

                log_predictions(all_preds, all_targets, all_video_ids, epoch, args)

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
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--lang", type=str, default="eng_Latn")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)

    # --- FLAGS ---
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--freeze_decoder", action="store_true")

    # NEW: Attiva il pooling per simulare SONAR
    parser.add_argument(
        "--sonar_pooling", action="store_true", help="Usa mean pooling (Vector-to-Text)"
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/nllb-200-distilled-600M",
        help="Nome del modello HuggingFace",
    )

    args = parser.parse_args()

    set_seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
