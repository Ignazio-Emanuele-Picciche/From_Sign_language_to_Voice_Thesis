"""
Sign Language Tokenizer
========================

Tokenizer per caption ASL usando Byte-Pair Encoding (BPE).
Basato su tokenizers library (HuggingFace).

Usage:
    # Train tokenizer
    tokenizer = SignLanguageTokenizer()
    tokenizer.train_from_csv('data/processed/utterances_with_translations.csv')
    tokenizer.save('models/sign_to_text_tokenizer.json')

    # Load and use
    tokenizer = SignLanguageTokenizer.load('models/sign_to_text_tokenizer.json')
    tokens = tokenizer.encode("Hello my name is John")
    text = tokenizer.decode(tokens)
    
TUNING COMPLETO:
.venv/bin/python src/sign_to_text/tune.py \
    --n_trials 50 \
    --epochs 10 \
    --optimize bleu
"""

import json
from pathlib import Path
from typing import List, Optional, Union, Dict
import pandas as pd
from tokenizers import (
    Tokenizer,
    normalizers,
    pre_tokenizers,
    decoders,
    trainers,
    processors,
)
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace, Metaspace as MetaspacePreTokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing


class SignLanguageTokenizer:
    """
    Tokenizer per caption Sign Language.

    Features:
    - BPE (Byte-Pair Encoding) per handling parole OOV
    - Special tokens: [PAD], [UNK], [SOS], [EOS]
    - Normalizzazione lowercase
    - Vocab size: ~3500-4000
    """

    # Special tokens
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
    SOS_TOKEN = "[SOS]"
    EOS_TOKEN = "[EOS]"

    def __init__(self, vocab_size: int = 4000):
        """
        Args:
            vocab_size: Dimensione vocabulary (default 4000)
        """
        self.vocab_size = vocab_size

        # Crea tokenizer BPE
        self.tokenizer = Tokenizer(BPE(unk_token=self.UNK_TOKEN))

        # Normalizzazione: lowercase + rimuovi accenti
        self.tokenizer.normalizer = normalizers.Sequence(
            [NFD(), Lowercase(), StripAccents()]
        )

        # Pre-tokenization: Metaspace (aggiunge ▁ prima delle parole)
        self.tokenizer.pre_tokenizer = MetaspacePreTokenizer()

        # Decoder (rimuove ▁ e ripristina spazi)
        self.tokenizer.decoder = decoders.Metaspace()

        # IDs speciali
        self.pad_token_id = None
        self.unk_token_id = None
        self.sos_token_id = None
        self.eos_token_id = None

    def train_from_csv(
        self, csv_path: str, caption_column: str = "caption", min_frequency: int = 2
    ):
        """
        Allena tokenizer su caption da CSV.

        Args:
            csv_path: Path al CSV con caption
            caption_column: Nome colonna caption
            min_frequency: Frequenza minima per includere token
        """
        print(f"\n🔧 Training tokenizer...")
        print(f"   CSV: {csv_path}")
        print(f"   Vocab size: {self.vocab_size}")

        # Carica caption
        df = pd.read_csv(csv_path)
        df = df[~df[caption_column].isna()]
        captions = df[caption_column].tolist()

        print(f"   Caption: {len(captions)}")

        # Setup trainer
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=[
                self.PAD_TOKEN,
                self.UNK_TOKEN,
                self.SOS_TOKEN,
                self.EOS_TOKEN,
            ],
            show_progress=True,
        )

        # Train da lista stringhe
        self.tokenizer.train_from_iterator(captions, trainer=trainer)

        # Memorizza IDs special tokens
        self.pad_token_id = self.tokenizer.token_to_id(self.PAD_TOKEN)
        self.unk_token_id = self.tokenizer.token_to_id(self.UNK_TOKEN)
        self.sos_token_id = self.tokenizer.token_to_id(self.SOS_TOKEN)
        self.eos_token_id = self.tokenizer.token_to_id(self.EOS_TOKEN)

        # Abilita padding
        self.tokenizer.enable_padding(
            pad_id=self.pad_token_id, pad_token=self.PAD_TOKEN
        )

        # Abilita truncation
        self.tokenizer.enable_truncation(max_length=512)

        print(f"\n   ✓ Tokenizer trained!")
        print(f"   Vocab size: {self.tokenizer.get_vocab_size()}")
        print(f"   PAD ID: {self.pad_token_id}")
        print(f"   UNK ID: {self.unk_token_id}")
        print(f"   SOS ID: {self.sos_token_id}")
        print(f"   EOS ID: {self.eos_token_id}")

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
    ) -> List[int]:
        """
        Encode testo → token IDs.

        Args:
            text: Testo da encodare
            add_special_tokens: Se aggiungere [SOS] e [EOS]
            max_length: Max lunghezza (None = no limit)
            padding: Se fare padding a max_length

        Returns:
            Lista token IDs
        """
        # Encode base
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids

        # Aggiungi special tokens
        if add_special_tokens:
            ids = [self.sos_token_id] + ids + [self.eos_token_id]

        # Truncate
        if max_length and len(ids) > max_length:
            if add_special_tokens:
                # Mantieni [SOS] e [EOS]
                ids = (
                    [self.sos_token_id] + ids[1 : max_length - 1] + [self.eos_token_id]
                )
            else:
                ids = ids[:max_length]

        # Padding
        if padding and max_length and len(ids) < max_length:
            ids = ids + [self.pad_token_id] * (max_length - len(ids))

        return ids

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = True,
    ) -> List[List[int]]:
        """
        Encode batch di testi.

        Args:
            texts: Lista testi
            add_special_tokens: Se aggiungere [SOS]/[EOS]
            max_length: Max lunghezza
            padding: Se fare padding

        Returns:
            Lista di liste token IDs
        """
        return [
            self.encode(text, add_special_tokens, max_length, padding) for text in texts
        ]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs → testo.

        Args:
            token_ids: Lista token IDs
            skip_special_tokens: Se rimuovere special tokens

        Returns:
            Testo decodificato
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(
        self, batch_ids: List[List[int]], skip_special_tokens: bool = True
    ) -> List[str]:
        """Decode batch."""
        return [self.decode(ids, skip_special_tokens) for ids in batch_ids]

    def save(self, path: str):
        """
        Salva tokenizer su file.

        Args:
            path: Path file .json
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Salva tokenizer
        self.tokenizer.save(str(path))

        # Salva metadata
        metadata = {
            "vocab_size": self.vocab_size,
            "pad_token_id": self.pad_token_id,
            "unk_token_id": self.unk_token_id,
            "sos_token_id": self.sos_token_id,
            "eos_token_id": self.eos_token_id,
            "pad_token": self.PAD_TOKEN,
            "unk_token": self.UNK_TOKEN,
            "sos_token": self.SOS_TOKEN,
            "eos_token": self.EOS_TOKEN,
        }

        metadata_path = path.with_suffix(".metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n💾 Tokenizer salvato:")
        print(f"   Tokenizer: {path}")
        print(f"   Metadata: {metadata_path}")

    @classmethod
    def load(cls, path: str) -> "SignLanguageTokenizer":
        """
        Carica tokenizer da file.

        Args:
            path: Path file .json

        Returns:
            SignLanguageTokenizer
        """
        path = Path(path)

        # Carica metadata
        metadata_path = path.with_suffix(".metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Crea istanza
        tokenizer_obj = cls(vocab_size=metadata["vocab_size"])

        # Carica tokenizer
        tokenizer_obj.tokenizer = Tokenizer.from_file(str(path))

        # Ripristina IDs
        tokenizer_obj.pad_token_id = metadata["pad_token_id"]
        tokenizer_obj.unk_token_id = metadata["unk_token_id"]
        tokenizer_obj.sos_token_id = metadata["sos_token_id"]
        tokenizer_obj.eos_token_id = metadata["eos_token_id"]

        print(f"\n📂 Tokenizer caricato da: {path}")
        print(f"   Vocab size: {tokenizer_obj.tokenizer.get_vocab_size()}")

        return tokenizer_obj

    def get_vocab(self) -> Dict[str, int]:
        """Ritorna vocabulary completo."""
        return self.tokenizer.get_vocab()

    def get_vocab_size(self) -> int:
        """Ritorna dimensione vocabulary."""
        return self.tokenizer.get_vocab_size()

    def __len__(self) -> int:
        """Vocab size."""
        return self.tokenizer.get_vocab_size()


def train_tokenizer_from_splits(
    train_csv: str = "results/utterances_analysis/train_split.csv",
    output_path: str = "models/sign_to_text/tokenizer.json",
    vocab_size: int = 4000,
):
    """
    Utility per trainare tokenizer da train split.

    Args:
        train_csv: Path train CSV
        output_path: Path output tokenizer
        vocab_size: Vocab size
    """
    print(f"\n{'='*80}")
    print(f"🔧 TRAINING SIGN LANGUAGE TOKENIZER")
    print(f"{'='*80}")

    # Train
    tokenizer = SignLanguageTokenizer(vocab_size=vocab_size)
    tokenizer.train_from_csv(train_csv)

    # Test
    print(f"\n🧪 Test tokenization:")
    test_sentences = [
        "Hello my name is John",
        "I am learning sign language",
        "How are you today",
    ]

    for sent in test_sentences:
        ids = tokenizer.encode(sent)
        decoded = tokenizer.decode(ids)
        print(f"\n   Original: {sent}")
        print(f"   IDs:      {ids}")
        print(f"   Decoded:  {decoded}")

    # Save
    tokenizer.save(output_path)

    print(f"\n{'='*80}")
    print(f"✅ TOKENIZER TRAINING COMPLETATO!")
    print(f"{'='*80}")
    print(f"\n📁 File: {output_path}")
    print(f"📊 Vocab: {tokenizer.get_vocab_size()} tokens")
    print(f"\n🚀 Next: Crea dataset loader (SignLanguageDataset)")
    print(f"\n")

    return tokenizer


if __name__ == "__main__":
    # Train tokenizer
    train_tokenizer_from_splits(
        train_csv="results/utterances_analysis/train_split.csv",
        output_path="models/sign_to_text/tokenizer.json",
        vocab_size=4000,
    )
