# =================================================================================================
# QWEN2.5-VL INFERENCE SCRIPT FOR GOLDEN LABELS
# =================================================================================================
#
# Questo script utilizza il modello Qwen2.5-VL-Instruct per l'inferenza sui golden labels
# passando direttamente i video invece delle sequenze di landmark.
#
# COMANDO PER ESEGUIRE:
# python src/models/three_classes/qwen_vl_golden_inference.py --model_size 2B --save_results --batch_size 10
#
# =================================================================================================

import torch
import numpy as np
import pandas as pd
import os
import sys
import argparse
import cv2
from PIL import Image
import transformers
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)
import logging
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Setup del Percorso di Base ---
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir)
)
sys.path.insert(0, BASE_DIR)

# Modelli disponibili (dal più piccolo al più grande)
AVAILABLE_MODELS = {
    "2B": "Qwen/Qwen2-VL-2B-Instruct",  # Versione precedente ma più leggera
    "3B": "Qwen/Qwen2.5-VL-3B-Instruct",  # Più piccolo della serie 2.5
    "7B": "Qwen/Qwen2.5-VL-7B-Instruct",  # Buon compromesso
    "32B": "Qwen/Qwen2.5-VL-32B-Instruct",  # Molto performante
    "72B": "Qwen/Qwen2.5-VL-72B-Instruct",  # Il più grande
}


class QwenVLGoldenLabelEvaluator:
    """
    Evaluator per golden labels usando modelli Qwen2.5-VL
    """

    def __init__(self, model_size="2B", device="auto", torch_dtype=torch.bfloat16):
        self.model_size = model_size
        self.model_name = AVAILABLE_MODELS[model_size]
        self.device = self._setup_device(device)
        self.torch_dtype = torch_dtype

        logger.info(f"Inizializzo QwenVLGoldenLabelEvaluator")
        logger.info(f"  Modello: {self.model_name}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Torch dtype: {self.torch_dtype}")

        # Carica il modello e i processori
        self._load_model()

        # Definisce il prompt per l'analisi del sentiment
        self.sentiment_prompt = self._create_sentiment_prompt()

        # Test di funzionamento
        logger.info("Testing model responsiveness...")
        self._test_model_response()

    def _setup_device(self, device):
        """Setup del device ottimale"""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device

    def _load_model(self):
        """Carica il modello e i processori"""
        try:
            logger.info("Caricamento modello e processori...")

            # Carica il modello
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device if self.device != "mps" else None,
                trust_remote_code=True,
                attn_implementation=(
                    "flash_attention_2" if self.device == "cuda" else "eager"
                ),
            )

            if self.device == "mps":
                self.model = self.model.to(self.device)

            # Carica il processore
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )

            logger.info("Modello caricato con successo!")

        except Exception as e:
            logger.error(f"Errore nel caricamento del modello: {e}")
            # Fallback a modello più piccolo se disponibile
            fallback_order = ["72B", "32B", "7B", "3B", "2B"]
            try:
                current_idx = fallback_order.index(self.model_size)
                if current_idx < len(fallback_order) - 1:
                    self.model_size = fallback_order[current_idx + 1]
                    logger.info(f"Tentativo con modello più piccolo: {self.model_size}")
                    self.model_name = AVAILABLE_MODELS[self.model_size]
                    self._load_model()
                else:
                    raise e
            except ValueError:
                # Se model_size non è nella lista, usa il più piccolo
                self.model_size = "2B"
                self.model_name = AVAILABLE_MODELS[self.model_size]
                self._load_model()

    def _create_sentiment_prompt(self):
        return """You are an expert in the field of emotions.
            Please focus on facial expressions, body language, environmental cues, and events in the video and predict the
            emotional state of the character. Please ignore the character's identity. We uniformly sample 10 frames per
            second from this clip. Please consider the temporal relationship between these frames.
            The video involves a person signing a sentence in ASL.



            Your task is to analyze the emotional sentiment of the person signing in this video.

            Focus ONLY on:
            - Facial expressions (eyebrows, mouth, eye movements)  
            - Body posture and shoulder positioning
            - Hand gesture intensity and fluidity
            - Overall emotional energy conveyed through signing

            The person may be expressing one of these emotions: joy, excited, surprise (positive), surprise (negative), worry, sadness, fear, disgust, frustration, anger, or a neutral state.

            Your job is to classify these into three categories:
            - Positive: joy, excited, surprise (positive) - shows happiness, enthusiasm, pleasant surprise.
            - Negative: surprise (negative), worry, sadness, fear, disgust, frustration, anger - shows distress, unpleasant emotions.
            - Neutral: ONLY if there is absolutely NO emotional expression whatsoever. This should be RARE in sign language.

            CRITICAL INSTRUCTIONS:
            1. Sign language is inherently expressive - most videos will show clear emotions
            2. Do NOT default to "Neutral" - actively look for positive or negative cues
            3. Even subtle facial expressions indicate emotion - classify them as Positive or Negative
            4. Only choose "Neutral" if you see a completely blank, expressionless face
            
            IMPORTANT: Completely ignore any text overlays, watermarks, background elements, or video interface components.
            
            Respond with exactly ONE word only: "Positive", "Negative", or "Neutral\""""

    def extract_frames_from_video(
        self, video_path, max_frames=8, target_size=(336, 336)
    ):
        """
        Estrae frame rappresentativi dal video per l'analisi
        MIGLIORATO: Focus sul centro dell'immagine per evitare interfacce
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Impossibile aprire il video: {video_path}")
                return None

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                logger.error(f"Video vuoto: {video_path}")
                return None

            # Salta i primi e ultimi frame per evitare artefatti
            skip_frames = max(1, total_frames // 20)  # Salta 5% all'inizio e fine
            start_frame = skip_frames
            end_frame = total_frames - skip_frames

            # Seleziona frame dalla parte centrale del video
            frame_indices = np.linspace(
                start_frame,
                end_frame,
                min(max_frames, end_frame - start_frame),
                dtype=int,
            )

            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if ret:
                    # Converte da BGR a RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # CROP CENTRALE per focalizzare sul signer
                    h, w = frame_rgb.shape[:2]
                    # Crop 80% centrale per eliminare watermark/interfacce
                    crop_w = int(w * 0.8)
                    crop_h = int(h * 0.8)
                    start_x = (w - crop_w) // 2
                    start_y = (h - crop_h) // 2
                    frame_rgb = frame_rgb[
                        start_y : start_y + crop_h, start_x : start_x + crop_w
                    ]

                    # Ridimensiona con dimensioni più grandi per dettaglio
                    if target_size:
                        frame_rgb = cv2.resize(frame_rgb, target_size)

                    # Converte a PIL Image
                    pil_frame = Image.fromarray(frame_rgb)
                    frames.append(pil_frame)

            cap.release()

            if not frames:
                logger.error(f"Nessun frame estratto da: {video_path}")
                return None

            logger.debug(f"Estratti {len(frames)} frame da {video_path} (cropped)")
            return frames

        except Exception as e:
            logger.error(f"Errore nell'estrazione frame da {video_path}: {e}")
            return None

    def analyze_video_sentiment(self, video_path):
        """
        Analizza il sentiment di un video usando Qwen2.5-VL
        """
        try:
            # Estrae frame dal video
            frames = self.extract_frames_from_video(video_path)
            if frames is None:
                return None, 0.5  # Return neutral confidence for failed videos

            # Prepara l'input per il modello
            # Usa più frame per una migliore analisi
            if len(frames) >= 3:
                # Usa 3 frame: inizio, metà, fine
                selected_frames = [frames[0], frames[len(frames) // 2], frames[-1]]
            else:
                selected_frames = frames

            # Per ora usa il frame centrale come principale, ma potremo usare tutti
            main_frame = frames[len(frames) // 2]  # Frame centrale

            # Crea il messaggio per il modello
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": main_frame,
                        },
                        {"type": "text", "text": self.sentiment_prompt},
                    ],
                }
            ]

            # Processa l'input
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Usa il processore correttamente per Qwen2.5-VL
            inputs = self.processor(text=text, images=main_frame, return_tensors="pt")

            # Muovi gli input al device
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].to(self.device)
                    if key == "pixel_values":
                        inputs[key] = inputs[key].to(dtype=self.torch_dtype)

            # Genera la risposta con retry per risposte inadeguate
            for attempt in range(3):  # Fino a 3 tentativi
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=15,  # Ridotto per risposte concise
                        do_sample=True,  # Sempre sampling per più varietà
                        temperature=1.2 + (attempt * 0.3),  # Temperature MOLTO più alta
                        top_p=0.95,  # Aumentato anche top_p
                        pad_token_id=self.processor.tokenizer.eos_token_id,
                    )

                # Decodifica tentativa
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]

                temp_output = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                # Se non contiene scuse/rifiuti, usa questa risposta
                if not any(
                    phrase in temp_output.lower()
                    for phrase in [
                        "i'm sorry",
                        "i cannot",
                        "more information",
                        "cannot provide",
                    ]
                ):
                    output_text = temp_output
                    break

                logger.debug(
                    f"Attempt {attempt + 1} failed, retrying with higher temperature"
                )

            else:
                # Tutti i tentativi falliti, usa l'ultimo
                output_text = temp_output
                logger.warning(f"All generation attempts failed for {video_path}")

            # La decodifica è già gestita nella retry logic sopra

            # Parse della risposta
            prediction, confidence = self._parse_prediction(output_text)

            # Log dettagliato per debugging
            logger.info(
                f"Video: {os.path.basename(video_path)} -> RAW: '{output_text.strip()}' -> PARSED: {prediction} (conf: {confidence:.3f})"
            )

            # Log extra per predizioni neutral (per debugging)
            if prediction == "Neutral":
                logger.warning(
                    f"⚠️  NEUTRAL detected for: {os.path.basename(video_path)}"
                )
                logger.warning(f"    Full raw output: '{output_text}'")

            return prediction, confidence

        except Exception as e:
            logger.error(f"Errore nell'analisi del video {video_path}: {e}")
            return None, 0.5

    def _parse_prediction(self, output_text):
        """
        Parse della predizione dal testo di output
        """
        output_clean = output_text.strip().lower()

        # Log SEMPRE per debugging completo
        logger.info(f"🔍 PARSING: '{output_text}' -> cleaned: '{output_clean}'")

        # Cerca parole chiave esatte prima
        if output_clean == "positive":
            logger.info("✅ EXACT MATCH: positive")
            return "Positive", 0.9
        elif output_clean == "negative":
            logger.info("✅ EXACT MATCH: negative")
            return "Negative", 0.9
        elif output_clean == "neutral":
            logger.warning("⚠️  EXACT MATCH: neutral")
            return "Neutral", 0.9
        # Poi cerca contenute
        elif "positive" in output_clean:
            logger.info("✅ FOUND 'positive' in text")
            return "Positive", 0.8
        elif "negative" in output_clean:
            logger.info("✅ FOUND 'negative' in text")
            return "Negative", 0.8
        elif "neutral" in output_clean:
            logger.warning("⚠️  FOUND 'neutral' in text")
            return "Neutral", 0.8
        else:
            # Se arriviamo qui, il modello non ha usato le keyword standard
            logger.warning(f"❌ NO STANDARD KEYWORD in: '{output_clean}'")

            # Fallback: analizza il contenuto per indizi
            positive_words = [
                "happy",
                "joy",
                "joyful",
                "smile",
                "smiling",
                "pleased",
                "satisfied",
                "good",
                "great",
                "wonderful",
                "excellent",
                "cheerful",
                "content",
                "glad",
                "excited",
            ]
            negative_words = [
                "sad",
                "angry",
                "mad",
                "upset",
                "worried",
                "frustrated",
                "disappointed",
                "bad",
                "terrible",
                "awful",
                "distressed",
                "unhappy",
                "miserable",
                "depressed",
                "fear",
                "disgust",
            ]

            pos_count = sum(1 for word in positive_words if word in output_clean)
            neg_count = sum(1 for word in negative_words if word in output_clean)

            logger.info(
                f"📊 Word analysis - Positive: {pos_count}, Negative: {neg_count}"
            )

            if pos_count > neg_count and pos_count > 0:
                logger.info(
                    f"🔶 FALLBACK to Positive (pos={pos_count} > neg={neg_count})"
                )
                return "Positive", 0.6
            elif neg_count > pos_count and neg_count > 0:
                logger.info(
                    f"🔶 FALLBACK to Negative (neg={neg_count} > pos={pos_count})"
                )
                return "Negative", 0.6
            else:
                logger.error(
                    f"⚠️  DEFAULTING TO NEUTRAL - No clear signal in: '{output_clean}'"
                )
                return "Neutral", 0.5

    def _test_model_response(self):
        """Test semplice per verificare che il modello risponda correttamente"""
        try:
            # Test con un prompt semplice
            test_prompt = "What color is the sky? Answer with one word."

            messages = [
                {"role": "user", "content": [{"type": "text", "text": test_prompt}]}
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = self.processor(text=text, return_tensors="pt")

            # Muovi al device
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            output = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            logger.info(f"Model test response: '{output.strip()}'")

        except Exception as e:
            logger.warning(f"Model test failed: {e}")


def load_golden_labels_data():
    """
    Carica i dati dei golden labels
    """
    # Percorsi
    processed_file = os.path.join(
        BASE_DIR, "data", "processed", "golden_label_sentiment_with_neutral.csv"
    )
    video_dir = os.path.join(
        BASE_DIR, "data", "raw", "ASLLRP", "batch_utterance_video_v3_1"
    )

    # Carica il CSV
    df = pd.read_csv(processed_file)

    # Verifica che i video esistano
    existing_videos = []
    for _, row in df.iterrows():
        video_path = os.path.join(video_dir, row["video_name"])
        if os.path.exists(video_path):
            existing_videos.append(
                {
                    "video_name": row["video_name"],
                    "video_path": video_path,
                    "true_label": row["emotion"],
                    "caption": row["caption"],
                }
            )
        else:
            logger.warning(f"Video non trovato: {video_path}")

    logger.info(f"Trovati {len(existing_videos)} video su {len(df)} golden labels")
    return existing_videos


def calculate_metrics(y_true, y_pred, labels):
    """
    Calcola tutte le metriche richieste
    """
    # Converti le etichette in indici numerici
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y_true_idx = [label_to_idx[label] for label in y_true]
    y_pred_idx = [label_to_idx[label] for label in y_pred]

    # Calcola le metriche
    accuracy = accuracy_score(y_true_idx, y_pred_idx)
    balanced_accuracy = balanced_accuracy_score(y_true_idx, y_pred_idx)
    f1_macro = f1_score(y_true_idx, y_pred_idx, average="macro")
    f1_weighted = f1_score(y_true_idx, y_pred_idx, average="weighted")
    precision_macro = precision_score(y_true_idx, y_pred_idx, average="macro")
    precision_weighted = precision_score(y_true_idx, y_pred_idx, average="weighted")
    recall_macro = recall_score(y_true_idx, y_pred_idx, average="macro")
    recall_weighted = recall_score(y_true_idx, y_pred_idx, average="weighted")

    # Calcola weighted accuracy personalizzata
    class_weights = np.bincount(y_true_idx) / len(y_true_idx)
    class_accuracies = []
    for i in range(len(labels)):
        mask = np.array(y_true_idx) == i
        if np.sum(mask) > 0:
            class_acc = accuracy_score(
                np.array(y_true_idx)[mask], np.array(y_pred_idx)[mask]
            )
            class_accuracies.append(class_acc)
        else:
            class_accuracies.append(0.0)
    weighted_accuracy = np.sum(
        [class_weights[i] * class_accuracies[i] for i in range(len(labels))]
    )

    return {
        "accuracy_standard": accuracy,
        "accuracy_balanced": balanced_accuracy,
        "accuracy_weighted": weighted_accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
    }


def print_results(metrics, y_true, y_pred, labels):
    """
    Stampa i risultati dettagliati
    """
    print("\n" + "=" * 80)
    print("RISULTATI QWEN2.5-VL SUI GOLDEN LABELS")
    print("=" * 80)

    print(f"\n=== METRICHE PRINCIPALI ===")
    print(
        f"Accuracy (Standard): {metrics['accuracy_standard']:.4f} ({metrics['accuracy_standard']*100:.2f}%)"
    )
    print(
        f"Accuracy (Balanced): {metrics['accuracy_balanced']:.4f} ({metrics['accuracy_balanced']*100:.2f}%)"
    )
    print(
        f"Accuracy (Weighted): {metrics['accuracy_weighted']:.4f} ({metrics['accuracy_weighted']*100:.2f}%)"
    )
    print(
        f"\nF1-Score (Macro): {metrics['f1_macro']:.4f} ({metrics['f1_macro']*100:.2f}%)"
    )
    print(
        f"F1-Score (Weighted): {metrics['f1_weighted']:.4f} ({metrics['f1_weighted']*100:.2f}%)"
    )
    print(
        f"\nPrecision (Macro): {metrics['precision_macro']:.4f} ({metrics['precision_macro']*100:.2f}%)"
    )
    print(
        f"Precision (Weighted): {metrics['precision_weighted']:.4f} ({metrics['precision_weighted']*100:.2f}%)"
    )
    print(
        f"\nRecall (Macro): {metrics['recall_macro']:.4f} ({metrics['recall_macro']*100:.2f}%)"
    )
    print(
        f"Recall (Weighted): {metrics['recall_weighted']:.4f} ({metrics['recall_weighted']*100:.2f}%)"
    )

    print(f"\nNumero totale di campioni: {len(y_true)}")

    # Distribuzione delle predizioni
    print(f"\nDistribuzione delle predizioni:")
    for label in labels:
        count = y_pred.count(label)
        percentage = count / len(y_pred) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    # Distribuzione delle etichette reali
    print(f"\nDistribuzione delle etichette reali:")
    for label in labels:
        count = y_true.count(label)
        percentage = count / len(y_true) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    # Confusion Matrix
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    y_true_idx = [label_to_idx[label] for label in y_true]
    y_pred_idx = [label_to_idx[label] for label in y_pred]

    print(f"\nMatrice di Confusione:")
    cm = confusion_matrix(y_true_idx, y_pred_idx)
    print("    Predicted")
    print("    " + "  ".join([f"{label:>8}" for label in labels]))
    for i, label in enumerate(labels):
        print(
            f"{label:>6} " + "  ".join([f"{cm[i][j]:>8}" for j in range(len(labels))])
        )

    # Classification Report
    print(f"\nClassification Report:")
    print(classification_report(y_true_idx, y_pred_idx, target_names=labels))


def save_results(results_data, metrics, model_size, save_dir):
    """
    Salva i risultati in file CSV
    """
    os.makedirs(save_dir, exist_ok=True)

    # Salva risultati per-sample
    results_df = pd.DataFrame(results_data)
    results_file = os.path.join(
        save_dir, f"qwen_vl_{model_size}_with_neutral_golden_results.csv"
    )
    results_df.to_csv(results_file, index=False)
    print(f"\nRisultati dettagliati salvati in: {results_file}")

    # Salva metriche aggregate
    metrics_df = pd.DataFrame(
        {
            "metric": list(metrics.keys()),
            "value": list(metrics.values()),
            "percentage": [v * 100 for v in metrics.values()],
            "description": [
                "Accuracy standard: % di predizioni corrette sul totale",
                "Accuracy bilanciata: accuracy robusta al class imbalance",
                "Accuracy pesata: accuracy che considera la distribuzione delle classi",
                "F1-Score macro: media non pesata dell'F1 per ogni classe",
                "F1-Score weighted: F1 pesato per il supporto di ogni classe",
                "Precision macro: media non pesata della precision per ogni classe",
                "Precision weighted: precision pesata per il supporto di ogni classe",
                "Recall macro: media non pesata del recall per ogni classe",
                "Recall weighted: recall pesato per il supporto di ogni classe",
            ],
        }
    )

    metrics_file = os.path.join(
        save_dir, f"qwen_vl_{model_size}_with_neutral__metrics_summary.csv"
    )
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metriche aggregate salvate in: {metrics_file}")


def main(args):
    """
    Funzione principale
    """
    print("=" * 80)
    print("QWEN2.5-VL GOLDEN LABELS INFERENCE")
    print("=" * 80)

    # Carica i dati
    print("\nCaricamento dati golden labels...")
    video_data = load_golden_labels_data()

    if not video_data:
        print("ERRORE: Nessun video trovato!")
        return

    print(f"Trovati {len(video_data)} video per l'analisi")

    # Limita il numero di video se specificato
    if args.max_videos:
        video_data = video_data[: args.max_videos]
        print(f"Processando solo i primi {len(video_data)} video")

    # Inizializza il valutatore
    print(f"\nInizializzazione modello Qwen2.5-VL ({args.model_size})...")
    evaluator = QwenVLGoldenLabelEvaluator(model_size=args.model_size, device="auto")

    # Esegue l'inferenza
    print("\nEsecuzione inferenza...")
    results_data = []
    y_true = []
    y_pred = []

    # Processa i video
    for i, video_info in enumerate(tqdm(video_data, desc="Analizzando video")):
        prediction, confidence = evaluator.analyze_video_sentiment(
            video_info["video_path"]
        )

        if prediction is not None:
            results_data.append(
                {
                    "video_name": video_info["video_name"],
                    "true_label": video_info["true_label"],
                    "predicted_label": prediction,
                    "confidence": confidence,
                    "caption": video_info["caption"],
                    "correct_prediction": video_info["true_label"] == prediction,
                }
            )

            y_true.append(video_info["true_label"])
            y_pred.append(prediction)

        # Salvataggio intermedio ogni 50 video (per evitare perdite)
        if (i + 1) % 50 == 0 and args.save_results:
            temp_df = pd.DataFrame(results_data)
            temp_file = os.path.join(
                BASE_DIR, "results", f"temp_qwen_results_{i+1}.csv"
            )
            os.makedirs(os.path.dirname(temp_file), exist_ok=True)
            temp_df.to_csv(temp_file, index=False)
            logger.info(f"Salvataggio intermedio: {temp_file}")

    if not results_data:
        print("ERRORE: Nessuna predizione valida ottenuta!")
        return

    # Calcola le metriche
    labels = sorted(list(set(y_true) | set(y_pred)))
    if "Neutral" not in labels:
        labels.append("Neutral")
    labels = sorted(labels)
    metrics = calculate_metrics(y_true, y_pred, labels)

    # Stampa i risultati
    print_results(metrics, y_true, y_pred, labels)

    # Salva i risultati
    if args.save_results:
        save_dir = os.path.join(BASE_DIR, "results")
        save_results(results_data, metrics, args.model_size, save_dir)

        # Rimuovi file temporanei
        for temp_file in os.listdir(save_dir):
            if temp_file.startswith("temp_qwen_results_"):
                os.remove(os.path.join(save_dir, temp_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inferenza Qwen2.5-VL sui Golden Labels"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="2B",
        choices=list(AVAILABLE_MODELS.keys()),
        help="Dimensione del modello Qwen2.5-VL da utilizzare",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,  # Qwen2.5-VL processa un video alla volta
        help="Batch size (attualmente non utilizzato, mantenuto per compatibilità)",
    )
    parser.add_argument(
        "--save_results", action="store_true", help="Salva i risultati in file CSV"
    )
    parser.add_argument(
        "--max_videos",
        type=int,
        default=None,
        help="Numero massimo di video da processare (per test rapidi)",
    )

    args = parser.parse_args()

    # Limita il numero di video se specificato
    if args.max_videos:
        print(f"MODALITÀ TEST: Processando solo i primi {args.max_videos} video")

    main(args)
