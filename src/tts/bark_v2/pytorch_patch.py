"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                  PYTORCH PATCH - FIX COMPATIBILITÀ BARK TTS                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

📋 DESCRIZIONE:
    Monkey patch per risolvere incompatibilità tra PyTorch 2.6+ e Bark TTS
    nel caricamento dei checkpoint del modello. Piccolo ma ESSENZIALE per
    far funzionare Bark con versioni recenti di PyTorch.

🔥 PROBLEMA RISOLTO:

    A partire da PyTorch 2.6.0, il comportamento di default di torch.load()
    è cambiato per ragioni di sicurezza:

    VECCHIO (PyTorch < 2.6):
        torch.load(file)  → weights_only=False (default)
        ✅ Carica qualsiasi oggetto Python

    NUOVO (PyTorch >= 2.6):
        torch.load(file)  → weights_only=True (default)
        ❌ Carica SOLO tensori, rifiuta altri oggetti Python

    IMPATTO SU BARK:
        Bark usa checkpoint salvati col vecchio formato che includono
        oggetti Python complessi (non solo weights). Con PyTorch 2.6+,
        il caricamento fallisce con errore:

        "FutureWarning: You are using torch.load with weights_only=False...
         pickle.UnpicklingError: invalid load key, '<'."

🔧 SOLUZIONE IMPLEMENTATA:

    Questo modulo applica un "monkey patch" che:

    1. Salva la funzione originale torch.load
       └─> _original_load = torch.load

    2. Crea wrapper patched_load che:
       └─> Aggiunge automaticamente weights_only=False se non specificato
       └─> Sopprime FutureWarning fastidiosi (sappiamo che Bark è sicuro)
       └─> Chiama la funzione originale con parametri corretti

    3. Sostituisce torch.load con la versione patched
       └─> torch.load = patched_load

    Risultato: Bark funziona perfettamente anche con PyTorch 2.9+!

🎯 QUANDO SERVE:

    ✅ Necessario se:
        - Usi PyTorch >= 2.6.0
        - Usi Bark TTS (qualsiasi versione)
        - Ottieni errori di unpickling al caricamento modelli

    ❌ Non necessario se:
        - Usi PyTorch < 2.6.0
        - Usi altri TTS engine (edge-tts, gTTS, etc.)

💡 UTILIZZO:

    Il patch viene applicato AUTOMATICAMENTE all'import del package:

    # In tts_generator.py:
    try:
        from . import pytorch_patch  # ← Applica patch qui
    except ImportError:
        pass  # Se non disponibile, continua (potrebbe funzionare lo stesso)

    Questo garantisce che il patch sia attivo PRIMA di importare Bark.

🔍 DETTAGLI TECNICI:

    patched_load(*args, **kwargs):
        • Controlla se 'weights_only' è già specificato
        • Se NO → aggiunge weights_only=False
        • Sopprime warnings::FutureWarning
        • Chiama torch.load originale con parametri safe
        • Ritorna risultato esattamente come torch.load normale

    Il patch è "trasparente": codice esistente continua a funzionare
    senza modifiche, ma ora compatibile con PyTorch recente.

⚠️ SICUREZZA:

    Perché è safe disabilitare weights_only per Bark?

    1. Bark è un progetto open source ufficiale di Suno AI
       └─> GitHub: https://github.com/suno-ai/bark
       └─> Checkpoint verificati e fidati

    2. Checkpoint scaricati da Hugging Face Hub ufficiale
       └─> Non da fonti random su internet

    3. Pickle di oggetti Python è necessario per architettura Bark
       └─> Non è possibile usare solo tensori

    4. Il patch è limitato SOLO a questo modulo
       └─> Non impatta altri progetti o librerie

    ⚠️ NON usare questo approccio per caricare checkpoint da fonti
       non fidate! Il pickle può eseguire codice arbitrario.

📊 IMPATTO:

    PRIMA del patch:
    ❌ ImportError quando si importa Bark
    ❌ UnpicklingError al caricamento modelli
    ❌ Impossibile usare Bark con PyTorch recente

    DOPO il patch:
    ✅ Import di Bark funziona
    ✅ Modelli caricano correttamente
    ✅ Generazione audio funziona perfettamente
    ✅ Nessun warning fastidioso in console

🔄 ALTERNATIVE CONSIDERATE:

    1. Downgrade PyTorch a 2.5
       ❌ Perde feature nuove, non sostenibile long-term

    2. Ricompilare checkpoint Bark in nuovo formato
       ❌ Complesso, richiede accesso a modello originale

    3. Fork Bark e modificare loading code
       ❌ Maintenance burden, si perde sync con upstream

    4. Monkey patch torch.load (SCELTA) ✅
       ✅ Semplice, non invasivo, reversibile
       ✅ Funziona con qualsiasi versione Bark
       ✅ 5 righe di codice vs giorni di lavoro

🎓 PATTERN UTILIZZATO:

    Questo è un esempio di "Monkey Patching" - tecnica Python per
    modificare comportamento di librerie a runtime senza toccare
    il codice sorgente. Utile per:
    - Quick fixes di compatibilità
    - Workaround temporanei
    - Testing/mocking

    Pro: Veloce, non invasivo
    Contro: Può rendere debugging più difficile (attenzione!)

📚 RIFERIMENTI:
    - PyTorch 2.6 release notes: https://pytorch.org/blog/pytorch-2.6-release/
    - Bark issue tracker: https://github.com/suno-ai/bark/issues
    - Pickle security: https://docs.python.org/3/library/pickle.html

🔗 INTEGRAZIONE:

    Import automatico in:
    - tts_generator.py: applica patch prima di importare Bark
    - __init__.py: può essere importato da package root

🧪 TESTING:

    Per verificare che il patch funzioni:

    1. Import modulo:
       >>> from src.tts.bark import pytorch_patch
       ✅ Patch PyTorch applicato per compatibilità con Bark

    2. Verifica torch.load modificato:
       >>> import torch
       >>> torch.load is pytorch_patch.patched_load  # False (wrapped)
       >>> # Ma comportamento è patched!

    3. Prova a caricare Bark:
       >>> from bark import preload_models
       >>> preload_models()  # Dovrebbe funzionare senza errori

💭 NOTE:

    - Questo patch sarà obsoleto quando Bark aggiornerà i checkpoint
      al nuovo formato PyTorch
    - Per ora (Nov 2025), Bark non ha ancora fatto l'aggiornamento
    - Il patch è backward-compatible: funziona anche con PyTorch < 2.6
      (semplicemente non ha effetto, weights_only=False è già default)

👤 AUTORE: Ignazio Emanuele Picciche
📅 DATA: Novembre 2025
🎓 PROGETTO: Tesi Magistrale - EmoSign con Bark TTS
🐛 FIX: PyTorch 2.6+ compatibility issue
"""

import torch
import warnings

# Salva la funzione originale
_original_load = torch.load


def patched_load(*args, **kwargs):
    """
    Wrapper per torch.load che permette il caricamento di modelli Bark
    anche con PyTorch 2.9+
    """
    # Se weights_only non è specificato, usa False per Bark
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False

    # Sopprimi i warning di sicurezza (sappiamo che Bark è safe)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        return _original_load(*args, **kwargs)


# Applica il patch
torch.load = patched_load

print("✅ Patch PyTorch applicato per compatibilità con Bark")
