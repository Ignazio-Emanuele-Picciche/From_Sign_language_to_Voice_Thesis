# !pip install evaluate sacrebleu rouge_score bert_score git+https://github.com/google-research/bleurt.git

# python train_sonar_slt.py \
#   --train_features_dir "data/features/train" \
#   --train_manifest "data/manifests/train.tsv" \
#   --val_features_dir "data/features/val" \
#   --val_manifest "data/manifests/val.tsv" \
#   --output_dir "models/run_english" \
#   --lang "eng_Latn" \
#   --batch_size 8 \
#   --epochs 5 \
#   --save_every 5 \
#   --val_every 5


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

# ==========================================
# 1. DATASET (Invariato ma usato due volte)
# ==========================================


class SignTranslationDataset(Dataset):
    def __init__(
        self,
        features_dir,
        manifest_path,
        tokenizer,
        max_length=128,
        tgt_lang="eng_Latn",
        split_name="data",  # Solo per print di debug
    ):
        self.features_dir = Path(features_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tgt_lang = tgt_lang
        self.split_name = split_name

        # --- CARICAMENTO ROBUSTO DEL MANIFEST ---
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
        # Usiamo tqdm solo se ci sono tanti file
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

        self.tokenizer.src_lang = "eng_Latn"  # Dummy source
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
# 2. MODELLO (Invariato)
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

        # --- FIX QUI SOTTO ---
        # Invece di tokenizer.lang_code_to_id[...] usiamo convert_tokens_to_ids
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
# 3. UTILITY (Invariato)
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

    # Importiamo direttamente la libreria per evitare i limiti del wrapper 'evaluate'
    import sacrebleu
    import evaluate  # Serve ancora per ROUGE/BLEURT

    # PREPARAZIONE DATI PER SACREBLEU
    # SacreBLEU si aspetta che le referenze siano una lista di liste (es. [ [ref1_tutta_la_lista], [ref2_opzionale] ])
    # Noi abbiamo una sola referenza per video, quindi mettiamo tutta la lista dentro un'altra lista.
    sacrebleu_refs = [references]

    # 1. BLEU-4 (Standard)
    # Questo è lo score ufficiale che si usa nei paper
    bleu4 = sacrebleu.corpus_bleu(predictions, sacrebleu_refs)
    results["BLEU-4"] = bleu4.score

    # 2. BLEU-1, 2, 3 (Custom)
    # Usiamo i pesi (weights) per isolare gli n-grammi

    # BLEU-1 (Solo parole singole)
    b1 = sacrebleu.corpus_bleu(predictions, sacrebleu_refs, weights=[1.0, 0, 0, 0])
    results["BLEU-1"] = b1.score

    # BLEU-2 (Media tra 1-gram e 2-gram)
    b2 = sacrebleu.corpus_bleu(predictions, sacrebleu_refs, weights=[0.5, 0.5, 0, 0])
    results["BLEU-2"] = b2.score

    # BLEU-3 (Media tra 1, 2 e 3-gram)
    b3 = sacrebleu.corpus_bleu(
        predictions, sacrebleu_refs, weights=[1 / 3, 1 / 3, 1 / 3, 0]
    )
    results["BLEU-3"] = b3.score

    # 3. ROUGE-L
    try:
        rouge_metric = evaluate.load("rouge")
        rouge = rouge_metric.compute(predictions=predictions, references=references)
        results["ROUGE-L"] = rouge["rougeL"] * 100
    except Exception as e:
        print(f"⚠️ Errore ROUGE: {e}")
        results["ROUGE-L"] = 0.0

    # 4. BLEURT (Opzionale - se fallisce lo ignoriamo per non bloccare tutto)
    try:
        bleurt_metric = evaluate.load("bleurt", config_name="bleurt-tiny-128")
        bleurt_res = bleurt_metric.compute(
            predictions=predictions, references=references
        )
        results["BLEURT"] = np.mean(bleurt_res["scores"])
    except Exception:
        # BLEURT spesso fallisce se manca internet o la libreria, non è critico
        pass

    return results


# ==========================================
# 4. MAIN
# ==========================================


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training su device: {device}")

    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- DATASETS SEPARATI ---
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

    # DataLoader non split, ma diretti
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    model = SonarSignModel(pretrained_model=model_name).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler()

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=total_steps
    )

    start_epoch = 0
    if args.resume_from:
        start_epoch = load_checkpoint(
            args.resume_from, model, optimizer, scheduler, scaler
        )

    print(
        f"🔥 Inizio Training: {len(train_dataset)} train samples, {len(val_dataset)} val samples."
    )

    for epoch in range(start_epoch, args.epochs):
        # --- TRAIN ---
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")

        for batch in progress_bar:
            inputs = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
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

        # --- CHECKPOINT ---
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

        # --- VALIDATION ---
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

                    with torch.cuda.amp.autocast():
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
            for k, v in metrics.items():
                print(f"   {k}: {v:.2f}")

            print("\n👀 Esempio:")
            print(f"   ✅: {all_targets[0]}")
            print(f"   🤖: {all_preds[0]}")
            print("-" * 40)

    print(f"🎉 Training completato! Output in {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Nuovi argomenti separati per Train e Val
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

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
