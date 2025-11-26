import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
import os

# Librerie per le metriche
try:
    from sacrebleu.metrics import BLEU
    import evaluate
except ImportError:
    print(
        "⚠️  Librerie mancanti. Installa: pip install sacrebleu evaluate rouge_score bleurt"
    )
    exit()


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
    ):
        self.features_dir = Path(features_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tgt_lang = tgt_lang

        try:
            df = pd.read_csv(manifest_path, sep="\t")
            if "text" not in df.columns:
                df = pd.read_csv(
                    manifest_path,
                    sep="\t",
                    header=None,
                    names=["id", "duration", "text"],
                )
        except:
            raise ValueError("Errore lettura manifest")

        df["id"] = df["id"].astype(str)
        df["text"] = df["text"].astype(str)
        self.ids = []
        self.texts = []

        print(f"🔍 Verifica file esistenti per il TEST...")
        found = 0
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Scanning"):
            vid_id = row["id"].strip()
            if (self.features_dir / f"{vid_id}.npy").exists():
                self.ids.append(vid_id)
                self.texts.append(row["text"])
                found += 1
        print(f"✅ Trovati {found} campioni validi.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        video_id = self.ids[idx]
        text = self.texts[idx]
        feat = np.load(self.features_dir / f"{video_id}.npy")
        video_tensor = torch.from_numpy(feat).float()
        return {"input_features": video_tensor, "text": text, "id": video_id}


# ==========================================
# 2. MODELLO (Con fix BaseModelOutput)
# ==========================================
class SonarSignModel(nn.Module):
    def __init__(self, pretrained_model, feature_dim=768, sonar_pooling=False):
        super().__init__()
        self.nllb = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
        self.sonar_pooling = sonar_pooling
        hidden_dim = self.nllb.config.d_model

        self.adapter = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def generate(self, input_features, tokenizer, max_new_tokens=100, num_beams=5):
        inputs_embeds = self.adapter(input_features)

        encoder_outputs = self.nllb.model.encoder(
            inputs_embeds=inputs_embeds, return_dict=True
        )
        last_hidden_state = encoder_outputs.last_hidden_state

        if self.sonar_pooling:
            pooled_state = torch.mean(last_hidden_state, dim=1, keepdim=True)
            custom_hidden_states = torch.nn.functional.normalize(pooled_state, dim=-1)
        else:
            custom_hidden_states = last_hidden_state

        target_lang = getattr(tokenizer, "tgt_lang", "eng_Latn")
        forced_bos_token_id = tokenizer.convert_tokens_to_ids(target_lang)

        # Wrapper fondamentale per NLLB
        wrapped_encoder_outputs = BaseModelOutput(
            last_hidden_state=custom_hidden_states
        )

        gen_ids = self.nllb.generate(
            encoder_outputs=wrapped_encoder_outputs,
            forced_bos_token_id=forced_bos_token_id,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
        )
        return gen_ids


# ==========================================
# 3. CALCOLO METRICHE AVANZATO
# ==========================================
def compute_all_metrics(preds, refs):
    print("\n📊 Calcolo metriche in corso (questo potrebbe richiedere un attimo)...")
    results = {}

    # Riferimenti per SacreBLEU (lista di liste)
    sacre_refs = [refs]

    # --- BLEU 1, 2, 3, 4 ---
    for n in range(1, 5):
        bleu = BLEU(max_ngram_order=n)
        score = bleu.corpus_score(preds, sacre_refs).score
        results[f"BLEU-{n}"] = score

    # --- ROUGE-L ---
    try:
        rouge = evaluate.load("rouge")
        r_score = rouge.compute(predictions=preds, references=refs)
        results["ROUGE-L"] = r_score["rougeL"] * 100  # Scala 0-100
    except Exception as e:
        print(f"⚠️ Errore ROUGE: {e}")
        results["ROUGE-L"] = -1

    # --- BLEURT ---
    try:
        print("   ↳ Caricamento BLEURT (può essere lento la prima volta)...")
        # Usa il modello tiny per velocità, oppure togli config_name per il full
        bleurt = evaluate.load("bleurt", config_name="bleurt-tiny-128")
        b_scores = bleurt.compute(predictions=preds, references=refs)["scores"]
        results["BLEURT"] = np.mean(b_scores) * 100  # Scala per coerenza leggibilità
    except Exception as e:
        print(f"⚠️ Errore BLEURT (probabile internet o spazio mancante): {e}")
        results["BLEURT"] = -1

    return results


# ==========================================
# 4. LOOP DI TEST
# ==========================================
def run_test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load Model Structure
    model = SonarSignModel(
        pretrained_model=args.model_name, sonar_pooling=args.sonar_pooling
    )

    # Load Checkpoint logic
    checkpoint_name = "BASELINE (Random)"
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"♻️  Caricamento pesi da {args.checkpoint}...")
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
            checkpoint_name = os.path.basename(args.checkpoint)
            print("✅ Checkpoint caricato.")
        except Exception as e:
            print(f"❌ Errore caricamento: {e}")
            return
    else:
        print("⚠️  NESSUN CHECKPOINT. Test in corso su modello non addestrato.")

    model.to(device)
    model.eval()

    # Load Data
    test_ds = SignTranslationDataset(
        args.test_features_dir, args.test_manifest, tokenizer, tgt_lang=args.lang
    )
    test_dl = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    all_preds = []
    all_refs = []
    all_ids = []

    print("🔮 Generazione traduzioni...")
    with torch.no_grad():
        for batch in tqdm(test_dl):
            inputs = batch["input_features"].to(device)
            texts = batch["text"]
            ids = batch["id"]

            tokenizer.tgt_lang = args.lang

            gen_ids = model.generate(inputs, tokenizer, num_beams=5, max_new_tokens=128)
            decoded_preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

            all_preds.extend(decoded_preds)
            all_refs.extend(texts)
            all_ids.extend(ids)

    # Calcolo Metriche Completo
    metrics = compute_all_metrics(all_preds, all_refs)

    # --- SALVATAGGIO REPORT ---
    out_file = Path(args.output_dir) / f"full_report_{args.lang}.txt"

    print(f"\n💾 Scrittura report in: {out_file}")
    with open(out_file, "w", encoding="utf-8") as f:
        # 1. HEADER CON METRICHE
        f.write("=" * 60 + "\n")
        f.write(f"TEST REPORT - Model: {checkpoint_name}\n")
        f.write(f"Data: {len(all_ids)} samples\n")
        f.write("=" * 60 + "\n")
        f.write("METRICHE GLOBALI:\n")
        f.write(f"🔹 BLEU-1  : {metrics['BLEU-1']:.2f}\n")
        f.write(f"🔹 BLEU-2  : {metrics['BLEU-2']:.2f}\n")
        f.write(f"🔹 BLEU-3  : {metrics['BLEU-3']:.2f}\n")
        f.write(f"🔹 BLEU-4  : {metrics['BLEU-4']:.2f}\n")
        f.write(f"🔸 ROUGE-L : {metrics['ROUGE-L']:.2f}\n")
        f.write(f"🔸 BLEURT  : {metrics['BLEURT']:.2f}\n")
        f.write("=" * 60 + "\n\n")

        # 2. DETTAGLIO FRASI
        f.write("DETTAGLIO PREDIZIONI:\n")
        for i in range(len(all_preds)):
            f.write(f"🆔 ID   : {all_ids[i]}\n")
            f.write(f"📝 GT   : {all_refs[i]}\n")
            f.write(f"🤖 PRED : {all_preds[i]}\n")
            f.write("-" * 40 + "\n")

    print("✅ Finito!")
    # Stampa a video le metriche principali
    print(f"\n🏆 BLEU-4: {metrics['BLEU-4']:.2f} | ROUGE-L: {metrics['ROUGE-L']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_features_dir", type=str, required=True)
    parser.add_argument("--test_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="test_results")

    # Checkpoint (opzionale per baseline)
    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument(
        "--model_name", type=str, default="facebook/nllb-200-distilled-600M"
    )
    parser.add_argument("--lang", type=str, default="eng_Latn")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sonar_pooling", action="store_true")

    args = parser.parse_args()
    Path(args.output_dir).mkdir(exist_ok=True, parents=True)
    run_test(args)
