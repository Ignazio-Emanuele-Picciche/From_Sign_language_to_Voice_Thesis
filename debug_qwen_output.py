#!/usr/bin/env python3
"""
Script di debug per capire cosa sta realmente generando Qwen2.5-VL
"""

import torch
import cv2
import numpy as np
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simple_test():
    """Test semplice per vedere l'output raw del modello"""

    # Carica il modello più piccolo per test rapido
    model_name = "Qwen/Qwen2-VL-2B-Instruct"

    logger.info(f"Caricando modello: {model_name}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "mps" else torch.float32,
        device_map=None,
        trust_remote_code=True,
    )

    if device == "mps":
        model = model.to(device)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Crea un'immagine di test semplice
    # Immagine nera (dovrebbe essere neutra/negativa)
    black_image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    # Immagine bianca (dovrebbe essere neutra/positiva)
    white_image = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)

    # Prompt molto semplice
    simple_prompts = [
        "Look at this image. Is it positive or negative? Answer with one word only.",
        "Positive or negative?",
        "What is the emotion: positive or negative?",
    ]

    test_images = [("black", black_image), ("white", white_image)]

    for img_name, test_image in test_images:
        print(f"\n{'='*50}")
        print(f"TESTING {img_name.upper()} IMAGE")
        print("=" * 50)

        for i, prompt in enumerate(simple_prompts):
            print(f"\n--- Test {i+1}: {prompt} ---")

            # Crea messaggio
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": test_image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Processa
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=text, images=test_image, return_tensors="pt")

            # Sposta al device
            for key in inputs:
                if torch.is_tensor(inputs[key]):
                    inputs[key] = inputs[key].to(device)
                    if key == "pixel_values":
                        inputs[key] = inputs[key].to(
                            dtype=torch.bfloat16 if device == "mps" else torch.float32
                        )

            # Genera 5 volte con temperature diverse
            temperatures = [0.1, 0.5, 0.9, 1.2, 1.5]

            for temp in temperatures:
                print(f"\nTemperature {temp}:")

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=True,
                        temperature=temp,
                        top_p=0.9,
                        pad_token_id=processor.tokenizer.eos_token_id,
                    )

                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]

                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

                print(f"  Raw output: '{output_text}'")
                print(f"  Cleaned: '{output_text.strip().lower()}'")


def test_with_real_video():
    """Test con un video reale per vedere se il problema è nell'immagine"""

    # Percorso di un video di esempio
    video_dir = "/Users/ignazioemanuelepicciche/Documents/TESI Magistrale UCBM/Improved_EmoSign_Thesis/data/raw/ASLLRP/batch_utterance_video_v3_1"

    # Prendi il primo video disponibile
    video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")][:3]

    if not video_files:
        print("Nessun video trovato!")
        return

    # Carica modello
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if device == "mps" else torch.float32,
        device_map=None,
        trust_remote_code=True,
    )

    if device == "mps":
        model = model.to(device)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        print(f"\n{'='*60}")
        print(f"TESTING VIDEO: {video_file}")
        print("=" * 60)

        # Estrai frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (224, 224))
            pil_frame = Image.fromarray(frame_rgb)

            # Test con prompt molto semplici
            simple_prompts = [
                "Positive or negative?",
                "Happy or sad?",
                "Good or bad emotion?",
            ]

            for prompt in simple_prompts:
                print(f"\n--- Prompt: {prompt} ---")

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": pil_frame},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(text=text, images=pil_frame, return_tensors="pt")

                for key in inputs:
                    if torch.is_tensor(inputs[key]):
                        inputs[key] = inputs[key].to(device)
                        if key == "pixel_values":
                            inputs[key] = inputs[key].to(
                                dtype=(
                                    torch.bfloat16 if device == "mps" else torch.float32
                                )
                            )

                # Test con 3 temperature diverse
                for temp in [0.3, 0.7, 1.1]:
                    with torch.no_grad():
                        generated_ids = model.generate(
                            **inputs,
                            max_new_tokens=15,
                            do_sample=True,
                            temperature=temp,
                            top_p=0.9,
                            pad_token_id=processor.tokenizer.eos_token_id,
                        )

                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                    ]

                    output_text = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )[0]

                    print(f"  T={temp}: '{output_text.strip()}'")


if __name__ == "__main__":
    print("🔍 DEBUGGING QWEN2.5-VL OUTPUT")
    print("Questo script testa se il modello risponde correttamente a prompt semplici")

    print("\n1. Test con immagini sintetiche...")
    simple_test()

    print("\n2. Test con video reali...")
    test_with_real_video()

    print("\n✅ Debug completato!")
