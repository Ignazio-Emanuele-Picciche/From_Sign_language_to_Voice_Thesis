"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              TTS GENERATOR - CORE GENERAZIONE AUDIO CON BARK                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 DESCRIZIONE:
    Modulo principale per la generazione di audio emotivo usando Bark TTS.
    Questo è il CUORE della pipeline TTS: coordina tutti i componenti
    (emotion_mapper, emotion_tag_optimizer) e effettua la sintesi vocale
    vera e propria producendo file WAV ad alta qualità.

🎯 SCOPO PRINCIPALE:
    Trasformare predizioni emotive da ViViT in audio espressivo:

    INPUT:
        - Emozione predetta: "Positive" | "Negative" | "Neutral"
        - Confidence score: 0.0-1.0 (certezza della predizione)
        - Testo da pronunciare: caption del sign language
        - Parametri opzionali: speaker, tag, ottimizzazioni

    OUTPUT:
        - File audio WAV a 24kHz (standard Bark)
        - Prosodia emotiva naturale
        - Qualità paragonabile a voce umana

🏗️ ARCHITETTURA BARK:

    Bark è un modello generativo transformer-based (GPT-like) sviluppato
    da Suno AI. Architettura multi-stage:

    1. TEXT → SEMANTIC TOKENS (GPT encoder)
       └─> Converte testo in rappresentazione semantica

    2. SEMANTIC → COARSE ACOUSTIC (GPT generator)
       └─> Genera feature acustiche a bassa risoluzione

    3. COARSE → FINE ACOUSTIC (refinement)
       └─> Raffina feature per qualità alta

    4. FINE ACOUSTIC → WAVEFORM (EnCodec decoder)
       └─> Converte in forma d'onda audio 24kHz

    Ogni stage è controllabile tramite "history_prompt" (speaker voice)
    e temperature (variabilità/creatività).

🔧 FUNZIONI PRINCIPALI:

    1. generate_emotional_audio()
       ╔════════════════════════════════════════════════════╗
       ║  FUNZIONE PIÙ IMPORTANTE - USA QUESTA!             ║
       ╚════════════════════════════════════════════════════╝

       Genera audio emotivo completo con tutti i controlli:

       Parametri chiave:
       • emotion: "Positive" | "Negative" | "Neutral"
       • confidence: 0.0-1.0 (influenza scelta tag)
       • video_name: identificatore per naming file
       • output_dir: dove salvare WAV
       • caption: testo da pronunciare (se None, usa template)
       • use_emotional_tags: abilita/disabilita tag ([laughs], etc.)
       • alternative_speaker: 0-2 per variare voce
       • alternative_tag: 0-N per variare tag emotivo
       • confidence_based_tags: adapta tag a confidence
       • optimize_tag_placement: usa posizionamento intelligente
       • preload: carica modelli in RAM (più veloce)

       Output: path al file WAV generato

       Workflow interno:
       1. Pre-carica modelli Bark (opzionale)
       2. Ottiene config Bark da emotion_mapper
       3. Seleziona speaker (con alternatives)
       4. Seleziona tag emotivo (basato su confidence o manuale)
       5. Genera/ottiene testo da pronunciare
       6. Ottimizza posizionamento tag (emotion_tag_optimizer)
       7. Chiama Bark per generare audio
       8. Salva WAV file
       9. Ritorna path

    2. generate_baseline_audio()
       └─> Genera audio NEUTRALE per confronti/baseline
           Usa speaker neutro (v2/en_speaker_9)
           Nessun tag emotivo
           Temperature basse (0.5)
           Utile per ablation studies

    3. preload_bark_models()
       └─> PRE-CARICA modelli Bark in RAM

           Perché serve:
           - Prima generazione: ~60-90 secondi (caricamento + sintesi)
           - Successive generazioni: ~15-30 secondi (solo sintesi)

           Trade-off:
           + Velocità: 4x più veloce dopo preload
           - RAM: richiede ~10GB memoria

           Quando usare:
           ✅ Batch processing (molti audio)
           ✅ Demo interattive
           ❌ Una tantum (spreco memoria)
           ❌ Server con poca RAM

🎨 CONTROLLO ESPRESSIVITÀ:

    Bark offre controllo fine tramite TEMPERATURE:

    Temperature basse (0.3-0.5):
    • Voce più consistente e prevedibile
    • Meno variazioni prosodiche
    • Meglio per neutral/informative

    Temperature medie (0.6-0.7):
    • Bilanciamento espressività/consistenza
    • Default per positive/negative
    • Sweet spot per uso generale

    Temperature alte (0.8-1.0):
    • Massima espressività e variabilità
    • Rischio: artefatti, inconsistenze
    • Solo per sperimentazione

📊 STRATEGIE TAG EMOTIVI:

    Il modulo supporta 3 modalità per selezione tag:

    1. CONFIDENCE-BASED (default, consigliato)
       └─> Tag adapta a certezza predizione
           High conf (>90%) → tag forte ([laughs])
           Med conf (70-90%) → tag moderato ([chuckles])
           Low conf (<70%) → nessun tag

    2. ALTERNATIVE INDEX (manuale)
       └─> Specifica quale tag dalla lista alternatives
           alternative_tag=0 → primary
           alternative_tag=1 → prima alternativa
           alternative_tag=2 → seconda alternativa, etc.

    3. CUSTOM TAG (via emotion_tag_optimizer)
       └─> Passa tag custom direttamente
           Utile per sperimentazione

💡 INNOVAZIONI RISPETTO A BARK VANILLA:

    Questo modulo estende Bark con:

    1. SISTEMA MULTI-TAG
       └─> Bark di default supporta tag, ma nessun sistema di selezione
           Qui: scelta intelligente basata su emotion + confidence

    2. TAG PLACEMENT OPTIMIZATION
       └─> Bark mette tag dove li scrivi nel testo
           Qui: analisi linguistica per posizionamento ottimale

    3. EMOTION-AWARE CONFIGURATION
       └─> Bark richiede config manuale
           Qui: mapping automatico emozione → speaker + temperature

    4. CONFIDENCE ADAPTATION
       └─> Novità assoluta: intensità tag ∝ certezza predizione

    5. SPEAKER ALTERNATIVES
       └─> Sistema per variare voce mantenendo coerenza emotiva

🔄 PIPELINE COMPLETA:

    Da video sign language ad audio emotivo:

    Video (.mp4)
      ↓ [ViViT Model]
    Emotion + Confidence (es: "Positive", 0.92)
      ↓ [emotion_mapper.py]
    Bark Config (speaker: v2/en_speaker_6, tag: [laughs], temp: 0.7)
      ↓ [get_tts_text()]
    Testo grezzo ("This video shows positive emotion")
      ↓ [emotion_tag_optimizer.py]
    Testo ottimizzato ("[laughs] This video shows [laughs] positive emotion")
      ↓ [Bark TTS - 4 stage generation]
    Audio WAV 24kHz
      ↓ [scipy.io.wavfile.write]
    File salvato (video_001_positive.wav)

⚙️ DIPENDENZE CRITICHE:

    Bark TTS:
    - bark: Modello principale (pip install git+https://github.com/suno-ai/bark.git)
    - transformers: Backend per transformer models
    - encodec: Audio codec per waveform generation

    Audio processing:
    - scipy: Salvataggio WAV files
    - numpy: Manipolazione array audio

    Moduli custom:
    - emotion_mapper: Configurazione Bark per emozioni
    - emotion_tag_optimizer: Posizionamento intelligente tag
    - text_templates: Generazione testo template (parent module)
    - pytorch_patch: Fix compatibilità PyTorch 2.6+

🎯 PARAMETRI OTTIMALI TROVATI:

    Dopo sperimentazione, configurazione migliore:

    POSITIVE:
        speaker: v2/en_speaker_6
        tag: [laughs] (high conf) / [chuckles] (med conf)
        temperature: 0.7
        placement: inizio + dopo pause

    NEGATIVE:
        speaker: v2/en_speaker_3
        tag: [sighs] (high/med conf) / [clears throat] (low conf)
        temperature: 0.6
        placement: metà frase + verso fine

    NEUTRAL:
        speaker: v2/en_speaker_9
        tag: nessuno (o [clears throat] se >80 char)
        temperature: 0.5
        placement: solo inizio se necessario

📈 PERFORMANCE:

    Tempi di generazione (Apple M1 Pro, 16GB RAM):
    - Prima generazione (con caricamento): ~60-90 sec
    - Con preload: ~15-30 sec per audio
    - Audio tipico: 3-8 secondi di durata

    Qualità audio:
    - Sample rate: 24kHz (Bark standard)
    - Bit depth: 16-bit PCM
    - Mono channel
    - File size: ~500KB per 10 sec audio

🧪 TESTING:

    Il modulo include blocco __main__ per testing end-to-end:
    - Pre-carica modelli
    - Genera baseline neutrale
    - Genera audio per ogni emozione
    - Salva file per ascolto manuale

    Esegui: python tts_generator.py
    Output: test_baseline_bark.wav, test_video_positive.wav, etc.

🔗 INTEGRAZIONE:

    Usato da:
    - Script principali: test_bark_*.py per sperimentazione
    - Pipeline evaluation: run_how2sign_evaluation.sh
    - Demo: demo_tag_optimization.py

    Import standard:
    from src.tts.bark.tts_generator import generate_emotional_audio

⚠️ LIMITAZIONI:

    1. VELOCITÀ
       └─> Bark è lento (~30 sec per frase su CPU)
           Soluzione: usa GPU se disponibile, preload modelli

    2. MEMORIA
       └─> Modelli occupano ~10GB RAM quando caricati
           Soluzione: non preload se memoria limitata

    3. LINGUE
       └─> Bark supporta multi-lingua ma mapping è solo EN
           Soluzione: estendi EMOTION_BARK_MAPPING per altre lingue

    4. TAG SUPPORT
       └─> Non tutti tag funzionano con tutti speaker
           Soluzione: testa combinazioni, usa quelle validate

    5. DETERMINISMO
       └─> Output non deterministico anche con seed (Bark limitation)
           Soluzione: genera multiple versioni, scegli migliore

📚 RIFERIMENTI:
    - Bark GitHub: https://github.com/suno-ai/bark
    - Bark paper: https://arxiv.org/abs/2301.03298
    - Tag emotivi: docs/BARK_EMOTIONAL_TAGS.md
    - Esperimenti prosody: docs/2_tts_prosody_optimization_report.md
    - Pipeline completa: docs/BARK_TTS_PIPELINE.md

💭 NOTE IMPLEMENTATIVE:

    - generate_audio() è wrapper attorno a pipeline Bark completa
    - SAMPLE_RATE (24000 Hz) è standard Bark, non modificare
    - history_prompt determina voce ma non è fine-tunable
    - temperature applica a text_temp E waveform_temp (entrambi)
    - WAV format scelto per compatibilità universale

🚀 ESTENSIONI FUTURE:

    Possibili miglioramenti:
    - [ ] Support per GPU acceleration (CUDA)
    - [ ] Batch generation (multiple audio in parallelo)
    - [ ] Caching di speaker embeddings
    - [ ] Fine-tuning su voci custom
    - [ ] Real-time streaming generation
    - [ ] Multi-language support completo
    - [ ] Emotion blending (mix di emozioni)

👤 AUTORE: Ignazio Emanuele Picciche
📅 DATA: Novembre 2025
🎓 PROGETTO: Tesi Magistrale - EmoSign con Bark TTS
🎵 COMPONENTE: Core TTS Generation Engine
"""

import os
import numpy as np
from typing import Optional
import warnings

# Applica patch per compatibilità PyTorch 2.9+
try:
    from . import pytorch_patch
except ImportError:
    pass  # Patch non disponibile, continua comunque

# Bark imports
try:
    from bark import SAMPLE_RATE, generate_audio, preload_models
    from scipy.io.wavfile import write as write_wav

    BARK_AVAILABLE = True
except ImportError:
    BARK_AVAILABLE = False
    warnings.warn(
        "Bark non installato. Installa con: pip install git+https://github.com/suno-ai/bark.git"
    )

from .emotion_mapper import (
    map_emotion_to_bark_prompt,
    get_bark_speaker,
    get_emotional_tag,
    get_alternative_emotional_tags,
)
from .emotion_tag_optimizer import optimize_emotional_text, get_alternative_tags

# Importa text_templates dal modulo parent
import sys
from pathlib import Path

parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from text_templates import get_tts_text, get_baseline_text


# Flag per pre-caricare i modelli (velocizza la generazione)
MODELS_PRELOADED = False


def preload_bark_models():
    """
    Pre-carica i modelli Bark in memoria
    Questo velocizza la generazione successiva ma richiede ~10GB di RAM
    """
    global MODELS_PRELOADED

    if not BARK_AVAILABLE:
        raise RuntimeError("Bark non è installato!")

    if not MODELS_PRELOADED:
        print("Caricamento modelli Bark in memoria...")
        preload_models()
        MODELS_PRELOADED = True
        print("✅ Modelli Bark caricati!")


def generate_emotional_audio(
    emotion: str,
    confidence: float,
    video_name: str,
    output_dir: str,
    caption: str = None,
    use_emotional_tags: bool = True,
    alternative_speaker: int = 0,
    alternative_tag: int = 0,
    confidence_based_tags: bool = True,
    preload: bool = False,
    optimize_tag_placement: bool = True,
) -> str:
    """
    Genera audio emotivo per un video analizzato usando Bark

    Args:
        emotion (str): Emozione predetta ('Positive', 'Negative', 'Neutral')
        confidence (float): Confidenza della predizione (0.0-1.0 o 0-100)
        video_name (str): Nome del video (per naming del file)
        output_dir (str): Directory dove salvare l'audio
        caption (str, optional): Testo originale del sign language da pronunciare
        use_emotional_tags (bool): Se True, usa tag emotivi come [laughs], [sighs]
        alternative_speaker (int): Indice speaker alternativo (0-2)
        alternative_tag (int): 🆕 Indice tag emotivo alternativo (0=default, 1+=alternatives)
        confidence_based_tags (bool): 🆕 Se True, sceglie tag basandosi su confidenza
        preload (bool): Se True, pre-carica i modelli Bark (prima generazione più lenta, successive veloci)
        optimize_tag_placement (bool): Se True, posiziona tag in modo intelligente nel testo

    Returns:
        str: Path completo del file audio generato

    Example:
        >>> # Tag default basato su confidenza
        >>> generate_emotional_audio('Positive', 0.92, 'video_001', 'results/tts_audio')

        >>> # Tag alternativo specifico
        >>> generate_emotional_audio('Positive', 0.92, 'video_001', 'results/tts_audio',
        ...                          alternative_tag=1)  # Usa [chuckles] invece di [laughs]
    """
    if not BARK_AVAILABLE:
        raise RuntimeError(
            "Bark non è installato! Installa con: pip install git+https://github.com/suno-ai/bark.git"
        )

    # Pre-carica modelli se richiesto
    if preload:
        preload_bark_models()

    # Crea directory se non esiste
    os.makedirs(output_dir, exist_ok=True)

    # Ottieni configurazione Bark per l'emozione
    bark_config = map_emotion_to_bark_prompt(emotion, use_emotional_tags)

    # Ottieni speaker (con possibilità di usare alternative)
    history_prompt = get_bark_speaker(emotion, alternative_speaker)

    # 🆕 Determina quale tag emotivo usare
    emotional_tag = ""
    if use_emotional_tags:
        if confidence_based_tags and alternative_tag == 0:
            # Usa tag basato su confidenza
            emotional_tag = get_emotional_tag(emotion, confidence=confidence)
        else:
            # Usa tag alternativo specifico o default
            emotional_tag = get_emotional_tag(emotion, alternative=alternative_tag)

    # Genera testo
    if caption:
        # Usa il caption originale del sign language
        text = caption
    else:
        # Genera testo basato sull'emozione
        text = get_tts_text(emotion, confidence, video_name, caption=None)

    # Applica ottimizzazione posizionamento tag emotivi
    if use_emotional_tags and emotional_tag:
        if optimize_tag_placement:
            # Usa posizionamento intelligente
            text = optimize_emotional_text(
                text=text,
                emotion=emotion,
                use_tags=True,
                custom_tag=emotional_tag,
            )
        else:
            # Metodo tradizionale: tag all'inizio
            text = f"{emotional_tag} {text}"

    # Path output (Bark genera WAV)
    safe_video_name = video_name.replace("/", "_").replace("\\", "_")
    output_filename = f"{safe_video_name}_{emotion.lower()}.wav"
    output_path = os.path.join(output_dir, output_filename)

    # Genera audio con Bark
    print(f"Generazione audio con Bark...")
    print(f"  Testo: {text[:80]}..." if len(text) > 80 else f"  Testo: {text}")
    print(f"  Speaker: {history_prompt}")
    print(f"  Tag emotivo: {emotional_tag if emotional_tag else '(nessuno)'}")
    print(f"  Temperature: {bark_config['temperature']}")

    audio_array = generate_audio(
        text,
        history_prompt=history_prompt,
        text_temp=bark_config["temperature"],
        waveform_temp=bark_config["temperature"],
    )

    # Salva audio come WAV (Bark usa sample rate 24kHz)
    write_wav(output_path, SAMPLE_RATE, audio_array)

    print(f"  ✅ Audio salvato: {output_path}")

    return output_path


def generate_baseline_audio(output_path: str, preload: bool = False) -> str:
    """
    Genera audio baseline neutrale (senza modulazione emotiva)
    Serve come riferimento per comparazioni

    Args:
        output_path (str): Path completo dove salvare il file
        preload (bool): Se True, pre-carica i modelli Bark

    Returns:
        str: Path del file audio generato

    Example:
        >>> generate_baseline_audio('results/tts_audio/baseline/baseline_neutral.wav')
        'results/tts_audio/baseline/baseline_neutral.wav'
    """
    if not BARK_AVAILABLE:
        raise RuntimeError("Bark non è installato!")

    # Pre-carica modelli se richiesto
    if preload:
        preload_bark_models()

    # Crea directory se non esiste
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Usa speaker neutro
    neutral_speaker = "v2/en_speaker_9"

    # Testo baseline
    text = get_baseline_text()

    # Genera audio
    print(f"Generazione audio baseline con Bark...")
    audio_array = generate_audio(
        text,
        history_prompt=neutral_speaker,
        text_temp=0.5,
        waveform_temp=0.5,
    )

    # Salva audio come WAV
    write_wav(output_path, SAMPLE_RATE, audio_array)

    print(f"✅ Baseline salvato: {output_path}")

    return output_path


if __name__ == "__main__":
    # Test del modulo
    print("=" * 70)
    print("TEST TTS GENERATOR - BARK")
    print("=" * 70)

    if not BARK_AVAILABLE:
        print("❌ Bark non installato!")
        print("   Installa con: pip install git+https://github.com/suno-ai/bark.git")
        exit(1)

    # Test 1: Pre-carica modelli
    print("\n1. Pre-caricamento modelli Bark...")
    preload_bark_models()

    # Test 2: Genera baseline
    print("\n2. Generazione audio baseline...")
    baseline_path = generate_baseline_audio("test_baseline_bark.wav", preload=True)

    # Test 3: Genera audio emotivo
    print("\n3. Generazione audio emotivo...")

    positive_path = generate_emotional_audio(
        emotion="Positive",
        confidence=0.92,
        video_name="test_video",
        output_dir=".",
        caption="This is a test of positive emotion with Bark TTS",
        preload=True,
    )
    print(f"   ✅ Audio Positive salvato in: {positive_path}")

    negative_path = generate_emotional_audio(
        emotion="Negative",
        confidence=0.85,
        video_name="test_video",
        output_dir=".",
        caption="This is a test of negative emotion with Bark TTS",
        preload=True,
    )
    print(f"   ✅ Audio Negative salvato in: {negative_path}")

    print("\n✅ Test completato! Ascolta i file .wav generati per verificare.")
    print("   Bark genera audio molto più espressivo di edge-tts!")
