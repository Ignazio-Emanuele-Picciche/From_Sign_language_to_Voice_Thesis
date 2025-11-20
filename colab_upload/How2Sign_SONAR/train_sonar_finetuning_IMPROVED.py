import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from typing import List, Dict, Tuple, Optional
import sacrebleu

# Tenta di importare sonar-space
try:
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    from sonar.models.sonar_text import load_sonar_text_encoder_model, load_sonar_tokenizer
    from sonar.models.sonar_text import SonarTextEncoder
    
    SONAR_AVAILABLE = True
except ImportError:
    print("⚠️ WARNING: sonar-space not installed. Decoding will not work.")
    SONAR_AVAILABLE = False

# ==========================================
# 1. DATASET (How2Sign Features)
# ==========================================
class How2SignDataset(Dataset):
    def __init__(self, features_dir: str, manifest_path: str, max_samples: int = None):
        """
        Args:
            features_dir: Directory contenente i file .npy (video_id.npy)
            manifest_path: Path al file .tsv (video_id, text)
            max_samples: Se specificato, carica solo N campioni (per debug)
        """
        self.features_dir = Path(features_dir)
        
        # Carica manifest
        self.data = pd.read_csv(manifest_path, sep="\t")
        
        # Rinomina colonne se necessario per standardizzare
        if "SENTENCE" in self.data.columns:
            self.data = self.data.rename(columns={"SENTENCE": "text", "SENTENCE_NAME": "video_id"})
        elif "text" not in self.data.columns:
            # Assumi colonne posizionali: video_id, text, ...
            self.data.columns = ["video_id", "text"] + list(self.data.columns[2:])
            
        # Filtra solo file esistenti
        self.valid_samples = []
        print(f"🔍 Scanning {len(self.data)} samples in {features_dir}...")
        
        for _, row in tqdm(self.data.iterrows(), total=len(self.data)):
            vid = row["video_id"]
            feat_path = self.features_dir / f"{vid}.npy"
            if feat_path.exists():
                self.valid_samples.append({
                    "video_id": vid,
                    "text": row["text"],
                    "path": str(feat_path)
                })
                
        if max_samples:
            self.valid_samples = self.valid_samples[:max_samples]
            
        print(f"✅ Loaded {len(self.valid_samples)} valid samples.")

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        
        # Carica features (T, 256)
        features = np.load(sample["path"])
        features = torch.from_numpy(features).float()
        
        return {
            "video_id": sample["video_id"],
            "features": features,
            "text": sample["text"]
        }

def collate_fn(batch):
    """Gestisce padding delle sequenze video"""
    video_ids = [b["video_id"] for b in batch]
    texts = [b["text"] for b in batch]
    features = [b["features"] for b in batch]
    
    # Padding features
    lengths = torch.tensor([f.size(0) for f in features])
    features_padded = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    
    return {
        "video_ids": video_ids,
        "texts": texts,
        "features": features_padded, # (B, max_len, 256)
        "lengths": lengths
    }

# ==========================================
# 2. MODEL (Encoder + SONAR Decoder)
# ==========================================
class SONARFineTuner(nn.Module):
    def __init__(self, encoder_checkpoint: str, device: str = "cuda", freeze_decoder: bool = True):
        super().__init__()
        self.device = device
        
        # 1. Encoder ASL (Trainable)
        # Proietta features (256) -> SONAR Space (1024)
        # Usiamo un'architettura semplice ma efficace
        self.encoder = self._build_encoder_from_state(encoder_checkpoint)
        
        # 2. SONAR Text Decoder (Frozen)
        # Usiamo la pipeline ufficiale per decodificare embeddings -> testo
        if SONAR_AVAILABLE:
            print("Loading SONAR Text Decoder...")
            self.decoder_pipeline = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder",
                tokenizer="text_sonar_basic_encoder",
                device=torch.device(device)
            )
            # Hack: Sostituiamo l'encoder con il nostro, o usiamo solo la parte di decodifica?
            # In realtà per il training ci serve:
            # - Encoder ASL (nostro) -> produce embedding
            # - Encoder Testo (SONAR) -> produce target embedding (per la loss)
            # - Decoder Testo (SONAR) -> per validazione (embedding -> testo)
            
            # Carichiamo anche un encoder testo separato per calcolare i target
            self.text_embedder = self.decoder_pipeline # Riutilizziamo la pipeline
            
            # Per la decodifica (embedding -> text), SONAR non ha un decoder pubblico facile
            # Ma possiamo usare la ricerca di vicini o un decoder se disponibile.
            # NOTA: SONAR è principalmente un encoder. Per la traduzione, si usa solitamente
            # un decoder seq2seq. Tuttavia, qui stiamo facendo "embedding alignment".
            # Se vogliamo generare testo, ci serve un decoder che prenda SONAR embeddings.
            # Il modello `f_sonar_text_decoder` è quello che cerchiamo.
            
            # Per ora, assumiamo che l'obiettivo sia allineare gli spazi.
            # La generazione del testo si farà con un decoder esterno o retrieval.
            # MA: Il codice originale usava `model.decode`. 
            # Se SONAR non ha un decoder text-to-text diretto esposto, usiamo un dummy o 
            # integriamo con un modello seq2seq pre-addestrato.
            
            # CORREZIONE: SONAR è un encoder sentence-level. 
            # Per generare testo da SONAR embeddings, serve un decoder addestrato.
            # NLLB o simili possono funzionare, ma qui usiamo l'approccio "embedding space".
            pass

    def _build_encoder_from_state(self, checkpoint_path: str) -> nn.Module:
        """
        Costruisce l'encoder e carica i pesi.
        Se il checkpoint è solo state_dict, costruisce l'architettura.
        """
        # Architettura definita nel notebook precedente
        encoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1024)
        )
        
        if os.path.exists(checkpoint_path):
            print(f"Loading encoder weights from {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location="cpu")
            encoder.load_state_dict(state)
        else:
            print("⚠️ Checkpoint not found, initializing random weights.")
            
        return encoder.to(self.device)

    def forward(self, features, lengths):
        """
        Args:
            features: (B, T, 256)
            lengths: (B,)
        Returns:
            embeddings: (B, 1024) Normalized SONAR embeddings
        """
        # Mean pooling mascherato
        mask = torch.arange(features.size(1), device=self.device)[None, :] < lengths[:, None]
        mask = mask.float().unsqueeze(2) # (B, T, 1)
        
        features_sum = (features * mask).sum(dim=1)
        features_avg = features_sum / lengths.unsqueeze(1).float()
        
        # Proiezione
        embeddings = self.encoder(features_avg)
        
        # === FIX 1: L2 NORMALIZATION ===
        # SONAR embeddings are unit norm. We MUST normalize.
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings

    def decode(self, embeddings: torch.Tensor, max_length: int = 128) -> List[str]:
        """
        Decodifica embeddings in testo.
        NOTA: SONAR non ha un decoder "ufficiale" pubblico semplice.
        Qui usiamo un placeholder o un retrieval se non abbiamo un decoder.
        
        Tuttavia, per calcolare il BLEU, ci serve testo.
        Se non abbiamo un decoder, questo script serve solo per allineare gli spazi.
        
        Per il fine-tuning, l'importante è la loss sugli embeddings.
        La validazione BLEU potrebbe fallire se non abbiamo un decoder.
        """
        # Placeholder: restituisce stringhe vuote se non c'è decoder
        # Se hai un decoder SONAR -> Text, usalo qui.
        return [""] * embeddings.size(0)

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Genera target embeddings usando SONAR text encoder.
        Args:
            texts: List[str] testi da codificare
        Returns:
            embeddings: (B, 1024) SONAR embeddings (detached, safe for backward)
        """
        if not SONAR_AVAILABLE:
            raise ImportError("SONAR (sonar-space) required for text encoding")
            
        with torch.no_grad():
            # SONAR API: predict(sentences, source_lang)
            embeddings = self.text_embedder.predict(
                texts, 
                source_lang="eng_Latn" # Inglese
            )
            
            # Converti a torch tensor
            if isinstance(embeddings, np.ndarray):
                embeddings = torch.from_numpy(embeddings).float()
                
            # IMPORTANTE: clone() per evitare errore inference mode
            # Gli embeddings SONAR sono creati in inference mode, 
            # ma servono per calcolare loss (che richiede gradients)
            embeddings = embeddings.to(self.device).clone().detach()
            
            # Ensure targets are also normalized (usually they are, but to be safe)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            return embeddings

def train_epoch(model: SONARFineTuner, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device: str) -> float:
    """Training loop per un epoch"""
    model.train()
    total_loss = 0.0
    
    progress = tqdm(dataloader, desc="Training")
    
    for batch in progress:
        features = batch["features"].to(device) # (B, max_len, 256) - con padding
        lengths = batch["lengths"].to(device)   # (B,) - lunghezze originali
        texts = batch["texts"]
        
        # Forward: features -> embeddings
        pred_embeddings = model(features, lengths=lengths) # (B, 1024)
        
        # Calcola target embeddings dai testi usando SONAR text encoder
        target_embeddings = model.encode_texts(texts) # (B, 1024)
        
        # === FIX 2: COSINE EMBEDDING LOSS ===
        # Instead of MSE, use Cosine Similarity for normalized embeddings
        # Loss = 1 - CosineSimilarity
        cosine_sim = F.cosine_similarity(pred_embeddings, target_embeddings)
        loss = 1.0 - cosine_sim.mean()
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # === FIX 3: GRADIENT MONITORING ===
        # Calculate gradient norm before clipping
        total_norm = 0.0
        for p in model.encoder.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Gradient clipping per stabilità
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar with loss and grad norm
        progress.set_postfix({
            "loss": f"{loss.item():.4f}",
            "grad": f"{total_norm:.2f}"
        })
        
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate(model: SONARFineTuner, dataloader: DataLoader, device: str) -> Tuple[float, List[Dict]]:
    """Evaluation: calcola Cosine Similarity media (BLEU richiede decoder)"""
    model.eval()
    
    total_cosine_sim = 0.0
    samples = []
    
    progress = tqdm(dataloader, desc="Evaluating")
    
    for batch in progress:
        features = batch["features"].to(device)
        lengths = batch["lengths"].to(device)
        texts = batch["texts"]
        video_ids = batch["video_ids"]
        
        # Forward
        embeddings = model(features, lengths=lengths)
        
        # Targets
        target_embeddings = model.encode_texts(texts)
        
        # Calculate Cosine Similarity
        cosine_sim = F.cosine_similarity(embeddings, target_embeddings)
        total_cosine_sim += cosine_sim.sum().item()
        
        # Store some samples
        if len(samples) < 20:
            for i in range(len(video_ids)):
                if len(samples) >= 20: break
                samples.append({
                    "video_id": video_ids[i],
                    "text": texts[i],
                    "cosine_sim": cosine_sim[i].item()
                })
    
    avg_cosine = total_cosine_sim / len(dataloader.dataset)
    
    # Note: We return avg_cosine as the "score" instead of BLEU
    # because we don't have a text decoder yet.
    return avg_cosine, samples

def main():
    parser = argparse.ArgumentParser(description="SONAR Fine-Tuning (IMPROVED)")
    
    # Paths
    parser.add_argument("--encoder_checkpoint", type=str, required=True, help="Path to SONAR encoder checkpoint (.pth)")
    parser.add_argument("--train_features", type=str, required=True, help="Directory con train features")
    parser.add_argument("--train_manifest", type=str, required=True, help="Train manifest (.tsv)")
    parser.add_argument("--val_features", type=str, required=True, help="Directory con val features")
    parser.add_argument("--val_manifest", type=str, required=True, help="Val manifest (.tsv)")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (increased for cosine loss)")
    parser.add_argument("--eval_every", type=int, default=1, help="Evaluate every N epochs")
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per split (for quick test)")
    
    # Model
    parser.add_argument("--freeze_decoder", action="store_true", default=True, help="Freeze decoder (recommended!)")
    parser.add_argument("--device", type=str, default="cuda")
    
    # Evaluation only
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate, no training")
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    
    # Dataset
    print("\n📂 Loading datasets...")
    train_dataset = How2SignDataset(args.train_features, args.train_manifest, max_samples=args.max_samples)
    val_dataset = How2SignDataset(args.val_features, args.val_manifest, max_samples=args.max_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
    # Model
    print("\n🔧 Building model...")
    model = SONARFineTuner(encoder_checkpoint=args.encoder_checkpoint, device=device, freeze_decoder=args.freeze_decoder)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Evaluation only
    if args.eval_only:
        print("\n📊 Evaluation only mode...")
        score, samples = evaluate(model, val_loader, device)
        
        print(f"\n✅ Avg Cosine Similarity: {score:.4f}")
        
        # Save results
        results_path = Path(args.output_dir) / "test_results.json"
        with open(results_path, "w") as f:
            json.dump({"cosine_sim": score, "samples": samples}, f, indent=2)
            
        print(f"💾 Results saved to {results_path}")
        return

    # Training
    print(f"\n🚀 Starting fine-tuning for {args.epochs} epochs...")
    print(f"📊 Train: {len(train_dataset)} samples")
    print(f"📊 Val: {len(val_dataset)} samples")
    print(f"📊 Batch size: {args.batch_size}")
    print(f"📊 Learning rate: {args.learning_rate}")
    print(f"🔒 Decoder frozen: {args.freeze_decoder}")
    
    best_score = -1.0
    metrics_history = []
    
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f"📉 Train Loss: {train_loss:.4f}")
        
        # Evaluate
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            score, samples = evaluate(model, val_loader, device)
            print(f"📊 Val Cosine Sim: {score:.4f}")
            
            # Save metrics
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_cosine_sim": score,
                "samples": samples[:5]
            }
            metrics_history.append(metrics)
            
            # Save predictions
            pred_path = Path(args.output_dir) / f"predictions_epoch{epoch:03d}.json"
            with open(pred_path, "w") as f:
                json.dump({"epoch": epoch, "cosine_sim": score, "samples": samples}, f, indent=2)
            
            # Save metrics
            metrics_path = Path(args.output_dir) / f"metrics_epoch{epoch:03d}.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            
            # Save best model
            if score > best_score:
                best_score = score
                best_path = Path(args.output_dir) / "best_encoder.pt"
                torch.save(model.encoder.state_dict(), best_path)
                print(f"💾 Best model saved (Cosine Sim: {score:.4f})")
                
    # Final summary
    print(f"\n{'='*60}")
    print(f"✅ Fine-Tuning completato!")
    print(f"{'='*60}")
    print(f"📊 Best Cosine Sim: {best_score:.4f}")
    print(f"💾 Models saved in: {args.output_dir}")
    
    # Save final config
    config_path = Path(args.output_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

if __name__ == "__main__":
    main()
