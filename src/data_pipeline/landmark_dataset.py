# PANORAMICA DEL FLUSSO:
# Questo file è il primo passo fondamentale: funge da ponte tra i dati grezzi su disco
# (i nostri file JSON con i landmark) e il modello di machine learning.
# La classe `LandmarkDataset` ha il compito di:
# 1. Leggere i metadati (nomi dei video ed etichette delle emozioni).
# 2. Caricare i file JSON corrispondenti a un video.
# 3. Trasformare i dati dei landmark in un formato numerico standard (tensore).
# 4. Garantire che tutte le sequenze video abbiano la stessa lunghezza (padding/troncamento).
# In sintesi, prepara i dati per essere "digeriti" dal modello durante l'addestramento.

import torch
from torch.utils.data import (
    Dataset,
)  # Classe base per creare dataset personalizzati in PyTorch
import json  # Per leggere i file di dati in formato JSON
import os  # Per interagire con il sistema operativo, come costruire percorsi di file
import numpy as np  # Libreria per calcoli numerici, specialmente per la gestione di array (matrici)
import pandas as pd  # Libreria per la manipolazione e l'analisi di dati, qui usata per leggere il file CSV dei metadati
import logging

# Configura il logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Definiamo una classe Dataset personalizzata. PyTorch richiede che erediti da `Dataset`.
class LandmarkDataset(Dataset):
    # Il metodo __init__ viene eseguito una sola volta, quando si crea un'istanza della classe.
    # Prepara le strutture dati iniziali.
    def __init__(
        self,
        landmarks_dir,
        processed_file,
        max_seq_length=100,
        keypoint_type="pose_keypoints_2d",
        num_features=50,  # Aggiunto per forzare una dimensione fissa
    ):
        """
        Args:
            landmarks_dir (string or list): Directory o lista di directory che contengono le cartelle dei video.
            processed_file (string): Percorso del file CSV che contiene i metadati.
            max_seq_length (int): Lunghezza massima a cui standardizzare le sequenze.
            keypoint_type (string): Tipo di keypoint da estrarre.
            num_features (int): Numero fisso di feature per ogni frame.
        """
        # Memorizza i percorsi e i parametri passati
        if isinstance(landmarks_dir, str):
            self.landmarks_dirs = [landmarks_dir]
        else:
            self.landmarks_dirs = landmarks_dir

        if isinstance(processed_file, list):
            self.processed = pd.concat(
                [pd.read_csv(f) for f in processed_file], ignore_index=True
            )
        else:
            self.processed = pd.read_csv(processed_file)
        self.max_seq_length = max_seq_length
        self.keypoint_type = keypoint_type
        self.num_features = num_features  # Memorizza il numero di feature

        # Crea una mappatura da etichetta testuale a un indice numerico, ordinando le etichette
        # per garantire coerenza (es. 'Negative' -> 0, 'Positive' -> 1).
        self.labels = sorted(self.processed["emotion"].unique())
        self.label_map = {label: i for i, label in enumerate(self.labels)}

    # Il metodo __len__ deve restituire la dimensione totale del dataset.
    def __len__(self):
        return len(self.processed)

    # Il metodo __getitem__ è il cuore del Dataset.
    # Carica e restituisce un singolo campione di dati dato un indice `idx`.
    def __getitem__(self, idx):
        if self.processed.empty:
            raise IndexError("Dataset is empty")

        if idx >= len(self.processed):
            raise IndexError(
                f"Index {idx} is out of range for dataset with size {len(self.processed)}"
            )

        video_info = self.processed.iloc[idx]
        video_name = str(video_info["video_name"]).replace(".mp4", "")
        # print("\n\nNOME VIDEO:", video_name, "\n\n")
        # print("\n\nLANDMARKS DIR:", self.landmarks_dirs, "\n\n")
        label_str = video_info["emotion"]
        label = self.label_map[label_str]

        # 2. Cerca la cartella dei landmark del video in tutte le directory specificate
        video_dir = None
        for l_dir in self.landmarks_dirs:
            potential_dir = os.path.join(l_dir, video_name)
            if os.path.isdir(potential_dir):
                video_dir = potential_dir
                break

        if video_dir is None:
            # If the directory is not found, we raise a FileNotFoundError.
            # This is a critical error indicating a mismatch between the CSV and the filesystem.
            # Raising an error is better than returning a default item, as it makes the problem visible.
            raise FileNotFoundError(
                f"Directory not found for video {video_name} in any of the provided directories."
            )

        # Lista dei file JSON ordinati per frame
        try:
            json_files = sorted(
                [f for f in os.listdir(video_dir) if f.endswith(".json")]
            )
        except FileNotFoundError:
            # This case should theoretically be caught by the `video_dir is None` check,
            # but it's good practice to keep it for robustness.
            raise FileNotFoundError(
                f"Error reading directory: {video_dir}. It may have been deleted during execution."
            )

        # 3. Estrae e trasforma i keypoints da ogni JSON in un vettore di feature
        sequence = []
        for jf in json_files:
            frame_path = os.path.join(video_dir, jf)
            try:
                with open(frame_path, "r") as f:
                    frame_data = json.load(f)

                people = frame_data.get("people", [])
                if not people:
                    continue

                # Estrae i keypoint specificati da `keypoint_type`
                keypoints = people[0].get(self.keypoint_type, [])
                if not keypoints:
                    continue

                # Per OpenPose, i dati sono in formato [x1, y1, c1, x2, y2, c2, ...].
                # Rimuoviamo i punteggi di confidenza (c).
                # Per MediaPipe (come da extract_landmarks.py modificato), i dati sono già [x1, y1, z1, ...].
                # La logica di rimozione del terzo elemento funziona per entrambi i casi se z non è desiderato,
                # altrimenti va adattata. Qui assumiamo di volere solo x, y.
                if "pose" in self.keypoint_type:  # Tipico di OpenPose
                    flat_landmarks = [
                        coord for i, coord in enumerate(keypoints) if i % 3 != 2
                    ]
                else:  # Assume che non ci siano punteggi di confidenza (es. MediaPipe)
                    flat_landmarks = keypoints

                # Converti esplicitamente in float32 qui
                sequence.append(np.array(flat_landmarks, dtype=np.float32))

            except (json.JSONDecodeError, IndexError) as e:
                logging.warning(
                    f"Errore nel processare il file {frame_path}: {e}. Sarà saltato."
                )
                continue

        if not sequence:
            # If a video directory is found but contains no valid landmarks, this is also a data error.
            # Instead of returning a default item, we raise an error to signal that this sample is unusable.
            raise ValueError(
                f"No valid landmarks found for video {video_name}. The directory might be empty or contain corrupt files."
            )

        # Converte la lista di liste Python in un array NumPy per efficienza
        sequence = np.array(sequence, dtype=np.float32)

        # Standardizza la dimensione delle feature per ogni frame
        current_features = sequence.shape[1]
        if current_features > self.num_features:
            # Tronca le feature se sono troppe
            sequence = sequence[:, : self.num_features]
        elif current_features < self.num_features:
            # Aggiunge padding se le feature sono troppo poche
            padding_features = np.zeros(
                (sequence.shape[0], self.num_features - current_features),
                dtype=np.float32,
            )
            sequence = np.hstack((sequence, padding_features))

        # 4. Standardizza la lunghezza della sequenza (Padding o Troncamento)
        if len(sequence) > self.max_seq_length:
            sequence = sequence[: self.max_seq_length]
        else:
            padding = np.zeros(
                (self.max_seq_length - len(sequence), sequence.shape[1]),
                dtype=np.float32,
            )
            sequence = np.vstack((sequence, padding))

        # 5. Restituisce i dati come tensori PyTorch.
        return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.long)
