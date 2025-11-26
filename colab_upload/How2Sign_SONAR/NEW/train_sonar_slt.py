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
    get_cosine_schedule_with_warmup,
)
import torch.optim as optim
import evaluate
import random
import mlflow
import shutil


# mlflow ui --backend-store-uri ./mlruns --port 5001

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
        max_samples=None,  # <--- NUOVO
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

        # --- TAGLIO DATASET ---
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
# 2. MODELLO (FIXED & OPTIMIZED)
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
            print("❄️  FREEZING ATTIVO.")
            for param in self.nllb.parameters():
                param.requires_grad = False
            self.nllb.gradient_checkpointing_enable()
            print("🛡️  Gradient Checkpointing ATTIVO")

        hidden_dim = self.nllb.config.d_model

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

        # FIX: Se tokenizer.tgt_lang è None, usiamo quello di default o args
        target_lang = getattr(tokenizer, "tgt_lang", "eng_Latn")

        # FIX 2: convert_tokens_to_ids a volte vuole una lista, a volte una stringa
        # NLLB usa un token speciale per la lingua (es. 'eng_Latn')
        # Ci assicuriamo di ottenere l'ID intero (int)
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

    # Import librerie
    from sacrebleu.metrics import BLEU
    import evaluate

    # SacreBLEU vuole una lista di liste per le referenze
    sacrebleu_refs = [refs]

    # 1. BLEU Scores (1, 2, 3, 4)
    # max_ngram_order=N calcola lo score cumulativo fino a N
    b1 = BLEU(max_ngram_order=1)
    results["BLEU-1"] = b1.corpus_score(preds, sacrebleu_refs).score

    b2 = BLEU(max_ngram_order=2)
    results["BLEU-2"] = b2.corpus_score(preds, sacrebleu_refs).score

    b3 = BLEU(max_ngram_order=3)
    results["BLEU-3"] = b3.corpus_score(preds, sacrebleu_refs).score

    b4 = BLEU(max_ngram_order=4)
    results["BLEU-4"] = b4.corpus_score(preds, sacrebleu_refs).score

    # 2. ROUGE-L
    try:
        rouge = evaluate.load("rouge")
        # Compute ritorna un dizionario, prendiamo rougeL e moltiplichiamo per 100
        results["ROUGE-L"] = (
            rouge.compute(predictions=preds, references=refs)["rougeL"] * 100
        )
    except Exception as e:
        print(f"⚠️ ROUGE saltato: {e}")
        results["ROUGE-L"] = 0.0

    # 3. BLEURT (Opzionale - scarica un modello)
    try:
        # Usa 'bleurt-tiny-128' che è veloce e leggero
        bleurt = evaluate.load("bleurt", config_name="bleurt-tiny-128")
        scores = bleurt.compute(predictions=preds, references=refs)["scores"]
        results["BLEURT"] = np.mean(scores)
    except Exception:
        # Se fallisce (es. niente internet o modello non trovato), metti 0
        pass

    return results


def log_predictions(preds, targets, video_ids, epoch, args):
    """
    Salva 5 campioni fissi (i primi 5) e 5 casuali in un file di testo.
    Modalità 'append' per avere uno storico unico.
    """
    log_path = Path(args.output_dir) / "validation_log.txt"

    total_samples = len(preds)
    # 1. I primi 5 sono "Fissi" (perché shuffle=False nel val_dl)
    fixed_indices = list(range(min(5, total_samples)))

    # 2. Altri 5 casuali (escludendo i primi 5 per non avere duplicati)
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

        # ... (Configurazione DataLoader invariata) ...
        train_dl = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=96,  # Consiglio 8 per stabilità
            prefetch_factor=600,
            pin_memory=True,
            persistent_workers=True,
        )

        val_bs = max(1, args.batch_size // 4)
        val_dl = DataLoader(
            val_ds,
            batch_size=val_bs,
            shuffle=False,  # IMPORTANTE: deve essere False per avere i fissi
            num_workers=16,
            prefetch_factor=200,
            pin_memory=True,
            persistent_workers=True,
        )

        model = SonarSignModel(freeze_decoder=False).to(device)

        # ... (Optimizer, Scheduler, ecc. invariati) ...
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(
            trainable_params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
        )
        scaler = torch.amp.GradScaler("cuda")
        num_steps = len(train_dl) * args.epochs
        num_warmup_steps = int(num_steps * 0.1)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_steps
        )
        loss_fct = nn.CrossEntropyLoss(label_smoothing=0.2, ignore_index=-100)
        early_stopper = EarlyStopping(patience=args.patience)

        print(f"🔥 Start Training: {len(train_ds)} samples.")

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")

            for batch in pbar:
                # ... (Training loop invariato) ...
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
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_train_loss = total_loss / len(train_dl)
            mlflow.log_metric("train_loss", avg_train_loss, step=epoch + 1)

            # --- VALIDATION ---
            if (epoch + 1) % args.val_every == 0:
                model.eval()
                val_loss = 0
                all_preds, all_targets = [], []

                # Lista per tenere traccia degli ID per il log
                all_video_ids = []  # <--- NUOVO

                print("🔍 Validating...")
                with torch.no_grad():
                    for batch_idx, batch in enumerate(
                        tqdm(val_dl)
                    ):  # enumerate utile se serve
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

                        # Recuperiamo gli ID dal dataset originale usando gli indici del batch
                        # Poiché shuffle=False, possiamo ricostruire o semplicemente non loggare l'ID
                        # MA: Il metodo più pulito è modificare il Dataset __getitem__ per ritornare anche l'ID
                        # OPPURE (metodo rapido):
                        start_idx = batch_idx * val_bs
                        end_idx = start_idx + len(decoded)
                        batch_ids = val_ds.ids[start_idx:end_idx]
                        all_video_ids.extend(batch_ids)  # <--- NUOVO

                avg_val_loss = val_loss / len(val_dl)
                metrics = calculate_metrics(all_preds, all_targets)

                print(
                    f"📊 Val Loss: {avg_val_loss:.4f} | BLEU-4: {metrics['BLEU-4']:.2f}"
                )

                # --- CHIAMATA ALLA FUNZIONE LOG ---
                log_predictions(
                    all_preds, all_targets, all_video_ids, epoch, args
                )  # <--- NUOVO
                # ----------------------------------

                # ... (Log mlflow e salvataggio checkpoint invariati) ...
                if early_stopper(avg_val_loss):
                    save_checkpoint(
                        model, optimizer, scheduler, epoch, args, "checkpoint_best.pth"
                    )

                save_checkpoint(
                    model, optimizer, scheduler, epoch, args, "checkpoint_last.pth"
                )

                if early_stopper.early_stop:
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lang", type=str, default="eng_Latn")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--random_seed", type=int, default=42)

    # NUOVO FLAG
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Numero massimo di campioni training (debug)",
    )
    parser.add_argument("--max_val_samples", type=int, default=None)

    args = parser.parse_args()

    set_seed(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
