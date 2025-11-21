import os
import math
import argparse
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline,
)

##############################################
# 1. Positional encoding
##############################################


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:, :T, :]


##############################################
# 2. Video backbone + encoder
##############################################


class VideoBackboneResNet(nn.Module):
    """
    Backbone per-frame basato su ResNet50 pre-addestrata su ImageNet.
    Input:  (B, T, 3, H, W)
    Output: (B, T, 2048)
    """

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # fino ad avgpool
        self.feature_dim = 2048

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)  # (B*T, 2048, 1, 1)
        feats = feats.view(B, T, -1)  # (B, T, 2048)
        return feats


class VideoEncoder(nn.Module):
    """
    Encoder video completo:
    - backbone per-frame (ResNet)
    - Transformer temporale
    - proiezione nello spazio SONAR (dim 1024)
    """

    def __init__(
        self,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        output_dim=1024,
    ):
        super().__init__()
        self.backbone = VideoBackboneResNet()
        self.input_dim = self.backbone.feature_dim  # 2048

        self.feature_projection = nn.Linear(self.input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,  # FIX
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.output_projection = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, frames, lengths=None):
        """
        frames:  (B, T, 3, H, W)
        lengths: (B,)  lunghezze reali (senza padding) oppure None

        Ritorna:
        - video_embeddings: (B, 1024)
        """
        # 1. Backbone per-frame
        feats = self.backbone(frames)  # (B, T, D_in)

        # 2. Proiezione
        x = self.feature_projection(feats)  # (B, T, H)

        # 3. Positional encoding
        x = self.pos_encoder(x)  # (B, T, H)

        # 4. Mask di padding
        if lengths is not None:
            B, T, _ = x.shape
            device = x.device
            time_ids = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
            src_key_padding_mask = time_ids >= lengths.unsqueeze(1)  # (B, T)
        else:
            src_key_padding_mask = None

        # 5. Transformer temporale
        x = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # (B, T, H)

        # 6. Pooling temporale (media sui frame validi)
        if lengths is not None:
            mask = ~src_key_padding_mask  # True sui frame validi
            mask_f = mask.float()  # (B, T)
            x_sum = (x * mask_f.unsqueeze(-1)).sum(dim=1)  # (B, H)
            lengths_eff = mask_f.sum(dim=1).clamp(min=1.0)  # (B,)
            x_pooled = x_sum / lengths_eff.unsqueeze(-1)  # (B, H)
        else:
            x_pooled = x.mean(dim=1)  # (B, H)

        # 7. Proiezione nello spazio SONAR
        emb = self.output_projection(x_pooled)  # (B, 1024)
        emb = self.output_norm(emb)
        emb = torch.tanh(emb)  # opzionale, per restare in range [-1,1]

        return emb


##############################################
# 3. Dataset video + collate
##############################################


class SignVideoDataset(Dataset):
    """
    manifest TSV: video_rel_path<TAB>testo
    root_dir: cartella base dei video (manifest usa path relativi)
    """

    def __init__(self, manifest_path, root_dir, max_frames=64):
        self.samples = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                video_rel, text = line.split("\t", 1)
                video_path = os.path.join(root_dir, video_rel)
                self.samples.append((video_path, text))
        self.max_frames = max_frames

    def __len__(self):
        return len(self.samples)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            # fallback: frame nero
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)]

        # campionamento semplice a max_frames
        num_frames = min(self.max_frames, len(frames))
        indices = np.linspace(0, len(frames) - 1, num=num_frames, dtype=int)
        frames = [frames[i] for i in indices]

        frames_t = []
        for f in frames:
            f = cv2.resize(f, (224, 224))
            f = f.astype(np.float32) / 255.0
            # normalizza come ImageNet
            f = (f - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            f = np.transpose(f, (2, 0, 1))  # (C, H, W)
            frames_t.append(f)

        frames_t = np.stack(frames_t, axis=0)  # (T, C, H, W)
        length = frames_t.shape[0]
        return frames_t, length

    def __getitem__(self, idx):
        video_path, text = self.samples[idx]
        frames, length = self._load_video(video_path)
        return {
            "frames": torch.from_numpy(frames),  # (T, C, H, W)
            "length": length,
            "text": text,
        }


def collate_fn(batch):
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    max_T = lengths.max().item()
    B = len(batch)

    C, H, W = batch[0]["frames"].shape[1:]  # (T, C, H, W) -> C,H,W
    frames_padded = torch.zeros(B, max_T, C, H, W, dtype=torch.float32)
    texts = []

    for i, b in enumerate(batch):
        T = b["length"]
        frames_padded[i, :T] = b["frames"]
        texts.append(b["text"])

    return {
        "frames": frames_padded,  # (B, T, C, H, W)
        "lengths": lengths,  # (B,)
        "texts": texts,
    }


##############################################
# 4. SONAR FineTuner (encoder/decoder veri)
##############################################


class SONARFineTuner(nn.Module):
    def __init__(self, device="cuda", mode="sonar"):
        super().__init__()
        self.device = device

        # 1) Video encoder
        self.video_encoder = VideoEncoder(
            hidden_dim=512,
            num_layers=4,
            num_heads=8,
            dropout=0.1,
            output_dim=1024,
        )

        # 2) SONAR text pipelines
        print("\n📥 Loading SONAR Text Pipelines from sonar-space...")
        self.text_embedder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=torch.device(device),
        )
        self.text_decoder = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_encoder",
            device=torch.device(device),
        )
        print("✅ SONAR text encoder/decoder loaded.")

        self.mode = mode  # per ora solo "sonar"

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            embeddings = self.text_embedder.predict(
                texts,
                source_lang="eng_Latn",
            )
        if isinstance(embeddings, np.ndarray):
            embeddings = torch.from_numpy(embeddings).float()
        embeddings = embeddings.to(self.device).clone().detach()
        return embeddings

    def decode(self, embeddings: torch.Tensor, max_length: int = 512) -> List[str]:
        if embeddings.is_cuda:
            embeddings = embeddings.cpu()
        texts = self.text_decoder.predict(
            embeddings,
            target_lang="eng_Latn",
            max_seq_len=max_length,
        )
        return texts

    def forward(self, frames, lengths, texts=None):
        # 1) Video -> embedding
        video_embeddings = self.video_encoder(frames, lengths)  # (B, 1024)
        outputs = {"video_embeddings": video_embeddings}

        # 2) Target embeddings da testo (solo in training)
        if self.mode == "sonar" and texts is not None:
            text_embeddings = self.encode_texts(texts)  # (B, 1024)
            outputs["text_embeddings"] = text_embeddings

        return outputs

    def compute_loss(self, outputs):
        video_embeddings = outputs["video_embeddings"]
        text_embeddings = outputs["text_embeddings"]

        # 1) Cosine loss (direzione)
        pred_norm = nn.functional.normalize(video_embeddings, p=2, dim=1)
        target_norm = nn.functional.normalize(text_embeddings, p=2, dim=1)
        cosine_sim = (pred_norm * target_norm).sum(dim=1).mean()
        loss_cosine = 1.0 - cosine_sim

        # 2) Magnitude loss (scala)
        pred_mag = video_embeddings.norm(p=2, dim=1)
        target_mag = text_embeddings.norm(p=2, dim=1)
        loss_mag = nn.functional.mse_loss(pred_mag, target_mag)

        loss = loss_cosine + 1.0 * loss_mag

        return loss, {
            "loss": loss.item(),
            "cos_loss": loss_cosine.item(),
            "mag_loss": loss_mag.item(),
            "p_norm": pred_mag.mean().item(),
            "t_norm": target_mag.mean().item(),
            "cos_sim": cosine_sim.item(),
        }

    def generate(self, video_embeddings: torch.Tensor, max_len: int = 40) -> List[str]:
        return self.decode(video_embeddings, max_length=max_len)


##############################################
# 5. BLEU (semplice 1-gram)
##############################################


def simple_bleu(reference: List[str], hypothesis: List[str]) -> float:
    assert len(reference) == len(hypothesis)
    total_match = 0
    total_words = 0
    for ref, hyp in zip(reference, hypothesis):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        total_words += len(hyp_tokens)
        total_match += sum(1 for w in hyp_tokens if w in ref_tokens)
    if total_words == 0:
        return 0.0
    return 100.0 * total_match / total_words


##############################################
# 6. Training + valutazione
##############################################


def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_cos = 0.0
    total_p = 0.0
    total_t = 0.0
    n_batches = 0

    for batch in tqdm(dataloader, desc="Train"):
        frames = batch["frames"].to(device)  # (B, T, C, H, W)
        lengths = batch["lengths"].to(device)  # (B,)
        texts = batch["texts"]

        outputs = model(frames, lengths, texts=texts)
        loss, loss_dict = model.compute_loss(outputs)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_cos += loss_dict["cos_sim"]
        total_p += loss_dict["p_norm"]
        total_t += loss_dict["t_norm"]
        n_batches += 1

    avg_loss = total_loss / n_batches
    avg_cos = total_cos / n_batches
    avg_p = total_p / n_batches
    avg_t = total_t / n_batches

    return avg_loss, avg_cos, avg_p, avg_t


def evaluate(model, dataloader, device, print_examples=5):
    model.eval()
    all_ref = []
    all_hyp = []
    printed = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Eval")):
            frames = batch["frames"].to(device)
            lengths = batch["lengths"].to(device)
            texts = batch["texts"]

            outputs = model(frames, lengths, texts=None)
            video_embeddings = outputs["video_embeddings"]  # (B, 1024)
            preds = model.generate(video_embeddings, max_len=40)

            all_ref.extend(texts)
            all_hyp.extend(preds)

            if printed < print_examples:
                for r, p in zip(texts, preds):
                    print("\n[REF]", r)
                    print("[HYP]", p)
                    printed += 1
                    if printed >= print_examples:
                        break

    bleu = simple_bleu(all_ref, all_hyp)
    return bleu


##############################################
# 7. Main
##############################################


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", type=str, required=True)
    parser.add_argument("--val_manifest", type=str, required=True)
    parser.add_argument("--video_root", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--save_dir", type=str, default="checkpoints_e2e")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Dataset e DataLoader
    train_dataset = SignVideoDataset(
        args.train_manifest, args.video_root, max_frames=args.max_frames
    )
    val_dataset = SignVideoDataset(
        args.val_manifest, args.video_root, max_frames=args.max_frames
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = SONARFineTuner(device=device, mode="sonar").to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )

    best_bleu = 0.0

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        train_loss, cos_sim, p_norm, t_norm = train_one_epoch(
            model, train_loader, optimizer, device
        )
        print(
            f"Train loss: {train_loss:.4f} | "
            f"cos_sim: {cos_sim:.3f} | "
            f"p_norm: {p_norm:.2f} | t_norm: {t_norm:.2f}"
        )

        bleu = evaluate(model, val_loader, device, print_examples=5)
        print(f"Val BLEU (simple): {bleu:.2f}")

        if bleu > best_bleu:
            best_bleu = bleu
            ckpt_path = os.path.join(
                args.save_dir, f"model_epoch{epoch}_bleu{bleu:.2f}.pt"
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved new best model to {ckpt_path}")

    print(f"\nTraining finished. Best BLEU: {best_bleu:.2f}")


if __name__ == "__main__":
    main()
