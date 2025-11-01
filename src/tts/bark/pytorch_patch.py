"""
Patch per Bark - Fix compatibilità con PyTorch 2.9+

PyTorch 2.6+ ha cambiato il default di weights_only=True in torch.load,
ma Bark usa checkpoint con il vecchio formato. Questo patch risolve il problema.
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
