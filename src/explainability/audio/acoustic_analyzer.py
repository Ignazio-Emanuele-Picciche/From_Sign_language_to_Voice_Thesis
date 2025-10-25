"""
Acoustic Analyzer - Estrae features acustiche da file audio
"""

import parselmouth
import librosa
import numpy as np
from typing import Dict, Optional
import warnings

warnings.filterwarnings("ignore")


class AcousticAnalyzer:
    """
    Analizza file audio per estrarre features acustiche (prosody)
    """

    def __init__(self, audio_path: str):
        """
        Args:
            audio_path (str): Path al file audio da analizzare
        """
        self.audio_path = audio_path

        # Carica con Parselmouth (per pitch analysis)
        try:
            self.sound = parselmouth.Sound(audio_path)
        except Exception as e:
            raise ValueError(f"Errore nel caricamento audio con Parselmouth: {e}")

        # Carica con librosa (per altre analisi)
        try:
            self.y, self.sr = librosa.load(audio_path, sr=None)
        except Exception as e:
            raise ValueError(f"Errore nel caricamento audio con librosa: {e}")

    def extract_pitch_features(self) -> Dict[str, float]:
        """
        Estrae features relative al pitch (frequenza fondamentale F0)

        Returns:
            dict: {
                'mean_pitch_hz': float,      # Pitch medio in Hz
                'std_pitch_hz': float,        # Deviazione standard
                'min_pitch_hz': float,        # Pitch minimo
                'max_pitch_hz': float,        # Pitch massimo
                'range_pitch_hz': float       # Range (max - min)
            }
        """
        try:
            # Estrai pitch con Praat
            pitch = self.sound.to_pitch()
            pitch_values = pitch.selected_array["frequency"]

            # Rimuovi valori unvoiced (0.0)
            pitch_values = pitch_values[pitch_values != 0]

            if len(pitch_values) == 0:
                # Nessun pitch rilevato (audio silenzioso o troppo corto)
                return {
                    "mean_pitch_hz": 0.0,
                    "std_pitch_hz": 0.0,
                    "min_pitch_hz": 0.0,
                    "max_pitch_hz": 0.0,
                    "range_pitch_hz": 0.0,
                }

            return {
                "mean_pitch_hz": float(np.mean(pitch_values)),
                "std_pitch_hz": float(np.std(pitch_values)),
                "min_pitch_hz": float(np.min(pitch_values)),
                "max_pitch_hz": float(np.max(pitch_values)),
                "range_pitch_hz": float(np.max(pitch_values) - np.min(pitch_values)),
            }
        except Exception as e:
            print(f"Errore nell'estrazione pitch: {e}")
            return {
                "mean_pitch_hz": 0.0,
                "std_pitch_hz": 0.0,
                "min_pitch_hz": 0.0,
                "max_pitch_hz": 0.0,
                "range_pitch_hz": 0.0,
            }

    def extract_rate_features(self) -> Dict[str, float]:
        """
        Estrae features relative alla velocità di eloquio (speaking rate)

        Returns:
            dict: {
                'duration_sec': float,           # Durata totale in secondi
                'syllables_estimated': int,      # Sillabe stimate
                'speaking_rate_syll_sec': float  # Sillabe al secondo
            }
        """
        # Durata
        duration = librosa.get_duration(y=self.y, sr=self.sr)

        # Stima sillabe usando onset detection (approssimazione)
        # Ogni onset ≈ una sillaba
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr)
        syllables = len(onsets)

        # Speaking rate
        speaking_rate = syllables / duration if duration > 0 else 0.0

        return {
            "duration_sec": float(duration),
            "syllables_estimated": int(syllables),
            "speaking_rate_syll_sec": float(speaking_rate),
        }

    def extract_energy_features(self) -> Dict[str, float]:
        """
        Estrae features relative all'energia/volume

        Returns:
            dict: {
                'mean_energy_db': float,   # Energia media in dB
                'max_energy_db': float,    # Energia massima
                'energy_range_db': float   # Range energia
            }
        """
        # Calcola RMS energy
        rms = librosa.feature.rms(y=self.y)[0]

        # Converti in dB
        db = librosa.amplitude_to_db(rms, ref=np.max)

        return {
            "mean_energy_db": float(np.mean(db)),
            "max_energy_db": float(np.max(db)),
            "energy_range_db": float(np.max(db) - np.min(db)),
        }

    def extract_voice_quality_features(self) -> Dict[str, float]:
        """
        Estrae features di qualità vocale (opzionale, più avanzato)

        Returns:
            dict: {
                'jitter': float,    # Perturbazione del pitch
                'shimmer': float,   # Perturbazione dell'ampiezza
                'hnr_db': float     # Harmonics-to-Noise Ratio
            }
        """
        try:
            # Jitter (variabilità locale del pitch)
            jitter = parselmouth.praat.call(
                self.sound, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3
            )

            # Shimmer (variabilità locale dell'ampiezza)
            shimmer = parselmouth.praat.call(
                self.sound, "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6
            )

            # HNR (Harmonics-to-Noise Ratio)
            harmonicity = self.sound.to_harmonicity()
            hnr = parselmouth.praat.call(harmonicity, "Get mean", 0.0, 0.0)

            return {
                "jitter": float(jitter) if not np.isnan(jitter) else 0.0,
                "shimmer": float(shimmer) if not np.isnan(shimmer) else 0.0,
                "hnr_db": float(hnr) if not np.isnan(hnr) else 0.0,
            }
        except Exception as e:
            print(f"Errore nell'estrazione voice quality: {e}")
            return {"jitter": 0.0, "shimmer": 0.0, "hnr_db": 0.0}

    def get_all_features(self, include_voice_quality: bool = False) -> Dict[str, float]:
        """
        Estrae tutte le features acustiche

        Args:
            include_voice_quality (bool): Se True, include jitter/shimmer/HNR

        Returns:
            dict: Dizionario completo con tutte le features
        """
        features = {}

        # Pitch
        features.update(self.extract_pitch_features())

        # Rate
        features.update(self.extract_rate_features())

        # Energy
        features.update(self.extract_energy_features())

        # Voice quality (opzionale)
        if include_voice_quality:
            features.update(self.extract_voice_quality_features())

        return features

    def print_summary(self, include_voice_quality: bool = False):
        """
        Stampa un summary delle features estratte
        """
        features = self.get_all_features(include_voice_quality)

        print(f"\n{'='*60}")
        print(f"ACOUSTIC ANALYSIS: {self.audio_path}")
        print(f"{'='*60}")

        print(f"\nPITCH FEATURES:")
        print(f"  Mean Pitch:  {features['mean_pitch_hz']:.2f} Hz")
        print(f"  Pitch Range: {features['range_pitch_hz']:.2f} Hz")
        print(f"  Pitch Std:   {features['std_pitch_hz']:.2f} Hz")

        print(f"\nRATE FEATURES:")
        print(f"  Duration:       {features['duration_sec']:.2f} sec")
        print(f"  Syllables:      {features['syllables_estimated']}")
        print(f"  Speaking Rate:  {features['speaking_rate_syll_sec']:.2f} syll/sec")

        print(f"\nENERGY FEATURES:")
        print(f"  Mean Energy: {features['mean_energy_db']:.2f} dB")
        print(f"  Max Energy:  {features['max_energy_db']:.2f} dB")

        if include_voice_quality:
            print(f"\nVOICE QUALITY:")
            print(f"  Jitter:  {features['jitter']:.4f}")
            print(f"  Shimmer: {features['shimmer']:.4f}")
            print(f"  HNR:     {features['hnr_db']:.2f} dB")


if __name__ == "__main__":
    # Test del modulo
    import sys

    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        print("Usage: python acoustic_analyzer.py <audio_file.mp3>")
        print("Usando file di test...")
        audio_file = "test_tts_output/baseline.mp3"

    try:
        analyzer = AcousticAnalyzer(audio_file)
        analyzer.print_summary(include_voice_quality=True)
    except Exception as e:
        print(f"Errore: {e}")
