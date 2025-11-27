"""
================================================================================
FINE-TUNING ROBERTA: MLFLOW, WEIGHTED METRICS & VAL LOSS MONITORING (MPS FIX)
================================================================================

MODELS: cardiffnlp/twitter-roberta-base-sentiment-latest
LOGGING: MLflow

CORREZIONI RISPETTO ALLA VERSIONE PRECEDENTE:
1. Fix per 'Placeholder storage... on MPS device': I pesi della loss vengono
   spostati dinamicamente sul device del modello.
2. Gestione corretta del device (CUDA vs MPS vs CPU).
3. Aggiornamento argomenti Trainer (processing_class al posto di tokenizer).

================================================================================
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)
import os
import mlflow

# --- CONFIGURAZIONE ---
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUTPUT_DIR = "models/finetuned_roberta_sentiment_mlflow"
EXPERIMENT_NAME = "Sentiment_Roberta_Experiment_ValLoss"

# Hyperparameters
MAX_LENGTH = 128
BATCH_SIZE = 256  # Ridotto a 16 per sicurezza su Mac/MPS
EPOCHS = 10
LEARNING_RATE = 1e-5

# File Paths
TRAIN_FILE_1 = (
    "data/train/asllrp_video_sentiment_data_with_neutral_0.34_without_golden.csv"
)
TRAIN_FILE_2 = "data/train/video_sentiment_data_with_neutral_0.34.csv"
VAL_FILE = "data/val/video_sentiment_data_with_neutral_0.34.csv"

# Mapping Label
label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}


# ------------------------------------------------------------------------------
# CLASS: CUSTOM TRAINER (Weighted Loss - MPS Safe)
# ------------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # --- FIX CRITICO PER MPS (Mac) ---
        # Ci assicuriamo che i pesi delle classi siano sullo stesso device del modello
        if self.class_weights.device != model.device:
            self.class_weights = self.class_weights.to(model.device)

        # Loss pesata
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


# ------------------------------------------------------------------------------
# DATA PROCESSING
# ------------------------------------------------------------------------------
def prepare_dataframe(df, source_name="Unknown"):
    """Pulisce e normalizza un DataFrame per il training."""
    initial_len = len(df)
    df = df.dropna(subset=["caption", "emotion"])

    # Normalizzazione stringhe
    df["emotion"] = df["emotion"].astype(str).str.upper().str.strip()

    # Mapping in interi
    df["label"] = df["emotion"].map(label2id)

    # Rimozione label non valide
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    final_len = len(df)
    print(
        f"[{source_name}] Righe: {initial_len} -> {final_len} (Dropped: {initial_len - final_len})"
    )
    return df


def load_datasets():
    print("\n--- 1. Caricamento Dataset ---")

    # 1. Training Set (Merge di due file)
    if not os.path.exists(TRAIN_FILE_1) or not os.path.exists(TRAIN_FILE_2):
        raise FileNotFoundError("File di training mancanti.")

    df_t1 = pd.read_csv(TRAIN_FILE_1)
    df_t2 = pd.read_csv(TRAIN_FILE_2)
    df_train_raw = pd.concat([df_t1, df_t2], ignore_index=True)

    print("Processamento Training Set...")
    df_train = prepare_dataframe(df_train_raw, source_name="TRAIN_MERGED")

    # 2. Validation Set
    if not os.path.exists(VAL_FILE):
        raise FileNotFoundError(f"File di validazione mancante: {VAL_FILE}")

    print("Processamento Validation Set...")
    df_val_raw = pd.read_csv(VAL_FILE)
    df_val = prepare_dataframe(df_val_raw, source_name="VALIDATION")

    print("\nDistribuzione Classi TRAIN:\n", df_train["emotion"].value_counts())
    print("\nDistribuzione Classi VAL:\n", df_val["emotion"].value_counts())

    return df_train, df_val


# ------------------------------------------------------------------------------
# METRICHE ESTESE
# ------------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, predictions)
    balanced_acc = balanced_accuracy_score(labels, predictions)
    f1_weighted = f1_score(labels, predictions, average="weighted")
    precision_weighted = precision_score(
        labels, predictions, average="weighted", zero_division=0
    )
    recall_weighted = recall_score(
        labels, predictions, average="weighted", zero_division=0
    )
    f1_macro = f1_score(labels, predictions, average="macro")

    return {
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "f1_weighted": f1_weighted,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
    }


# ------------------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------------------
def main():
    # Setup MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Rilevamento Device Corretto (Include MPS per Mac)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Caricamento Dati
    train_df, val_df = load_datasets()

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(val_df),
        }
    )

    # Tokenizer
    print(f"\n--- 2. Tokenizzazione ({MODEL_NAME}) ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(
            examples["caption"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    cols_to_remove = ["video_name", "caption", "emotion"]
    if "__index_level_0__" in tokenized_datasets["train"].column_names:
        cols_to_remove.append("__index_level_0__")

    tokenized_datasets = tokenized_datasets.remove_columns(cols_to_remove)
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    # Calcolo Pesi
    print("\n--- 3. Calcolo Pesi Classi (Weighted Loss) ---")
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df["label"]),
        y=train_df["label"],
    )
    # Inizializziamo il tensore (verrà spostato sul device corretto nel trainer)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    print(f"Pesi applicati (Neg, Neu, Pos): {class_weights}")

    # Modello
    print("\n--- 4. Inizializzazione Modello ---")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=id2label, label2id=label2id
    ).to(device)

    # Configurazione Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        report_to="mlflow",
        run_name=f"run_valloss_weighted",
        dataloader_pin_memory=False,  # Disabilita pin_memory per evitare warning su MPS
    )

    # Inizializzazione Trainer Personalizzato
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,  # Aggiornato da 'tokenizer' a 'processing_class'
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # Iniezione pesi nel trainer
    trainer.class_weights = class_weights_tensor

    # Training
    print("\n--- 5. Avvio Training (Monitoraggio su Val Loss) ---")
    trainer.train()

    # Salvataggio Finale
    print(f"\n--- 6. Salvataggio Miglior Modello in {OUTPUT_DIR} ---")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    metrics = trainer.evaluate()
    print("Metriche finali (Best Model):", metrics)
    mlflow.end_run()


if __name__ == "__main__":
    main()
