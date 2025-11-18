#!/usr/bin/env python3
"""
Script per applicare fix automatici a train_sonar_finetuning.py

Fix applicati:
1. Normalizzazione L2 output encoder
2. Cosine Loss invece di MSE
3. Gradient norm monitoring
4. Logging avanzato
5. Validation metrics estese

Uso:
    python apply_training_fixes.py

Crea: train_sonar_finetuning_IMPROVED.py
"""

from pathlib import Path
import shutil
from datetime import datetime


def apply_fixes():
    """Applica fix allo script di training"""

    print("=" * 70)
    print("🔧 APPLICAZIONE FIX AUTOMATICI")
    print("=" * 70)

    original_script = Path("train_sonar_finetuning.py")

    if not original_script.exists():
        print(f"\n❌ Script non trovato: {original_script}")
        print(f"   Assicurati di essere nella directory corretta!")
        return False

    print(f"\n📄 Lettura script originale...")
    with open(original_script, "r", encoding="utf-8") as f:
        content = f.read()

    # Crea backup con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = Path(f"train_sonar_finetuning_BACKUP_{timestamp}.py")

    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"   ✅ Backup creato: {backup_path}")

    fixes_applied = 0

    # ========================================================================
    # FIX 1: Normalizzazione Output Encoder
    # ========================================================================
    print(f"\n🔧 Fix 1: Normalizzazione output encoder...")

    old_forward = """        # Media temporale (esclude padding)
        if lengths is not None:
            for i in range(B):
                features_avg[i] = features[i, :lengths[i], :].mean(dim=0)
        else:
            features_avg = features.mean(dim=1)  # [B, D]

        return features_avg"""

    new_forward = """        # Media temporale (esclude padding)
        if lengths is not None:
            for i in range(B):
                features_avg[i] = features[i, :lengths[i], :].mean(dim=0)
        else:
            features_avg = features.mean(dim=1)  # [B, D]

        # NORMALIZZAZIONE L2 (risolve problemi di scala)
        features_avg = torch.nn.functional.normalize(features_avg, p=2, dim=1)

        return features_avg"""

    if old_forward in content:
        content = content.replace(old_forward, new_forward)
        print(f"   ✅ Normalizzazione L2 applicata")
        fixes_applied += 1
    else:
        print(f"   ⚠️  Pattern non trovato (potrebbe essere già applicato)")

    # ========================================================================
    # FIX 2: Cosine Loss invece di MSE
    # ========================================================================
    print(f"\n🔧 Fix 2: Cosine Loss invece di MSE...")

    old_loss = """        # Loss: MSE tra embeddings
        loss = torch.nn.functional.mse_loss(embeddings, target_embeddings)"""

    new_loss = """        # Loss: 1 - Cosine Similarity (migliore per embeddings normalizzati)
        # Normalizza target (encoder già normalizzato)
        target_embeddings_norm = torch.nn.functional.normalize(target_embeddings, p=2, dim=1)
        
        # Cosine similarity: dot product di vettori normalizzati
        cosine_sim = (embeddings * target_embeddings_norm).sum(dim=1).mean()
        
        # Loss: 1 - similarity (range [0, 2], ottimo = 0)
        loss = 1.0 - cosine_sim"""

    if old_loss in content:
        content = content.replace(old_loss, new_loss)
        print(f"   ✅ Cosine Loss implementato")
        fixes_applied += 1
    else:
        print(f"   ⚠️  Pattern non trovato (potrebbe essere già applicato)")

    # ========================================================================
    # FIX 3: Gradient Monitoring
    # ========================================================================
    print(f"\n🔧 Fix 3: Gradient monitoring...")

    old_backward = """        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        self.optimizer.step()"""

    new_backward = """        loss.backward()
        
        # MONITORING: Calcola norma gradiente
        total_norm = 0.0
        for p in self.encoder.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), max_norm=1.0)
        self.optimizer.step()"""

    if old_backward in content:
        content = content.replace(old_backward, new_backward)
        print(f"   ✅ Gradient monitoring aggiunto")
        fixes_applied += 1
    else:
        print(f"   ⚠️  Pattern non trovato (potrebbe essere già applicato)")

    # ========================================================================
    # FIX 4: Logging Avanzato
    # ========================================================================
    print(f"\n🔧 Fix 4: Logging avanzato...")

    old_logging = """            pbar.set_postfix({"loss": f"{loss.item():.4f}"})"""

    new_logging = """            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "grad_norm": f"{total_norm:.4f}",
                "cosine_sim": f"{cosine_sim.item():.4f}"
            })"""

    if old_logging in content:
        content = content.replace(old_logging, new_logging)
        print(f"   ✅ Logging migliorato")
        fixes_applied += 1
    else:
        print(f"   ⚠️  Pattern non trovato (potrebbe essere già applicato)")

    # ========================================================================
    # FIX 5: Evaluation Metrics Estese
    # ========================================================================
    print(f"\n🔧 Fix 5: Evaluation metrics...")

    old_eval_log = """        log_entry = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_bleu": bleu_score,
        }"""

    new_eval_log = """        # Calcola cosine similarity media (1 - loss se usiamo cosine loss)
        val_cosine = 1.0 - avg_loss if avg_loss < 2.0 else 0.0
        
        log_entry = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_bleu": bleu_score,
            "val_cosine_sim": val_cosine,
        }"""

    if old_eval_log in content:
        content = content.replace(old_eval_log, new_eval_log)
        print(f"   ✅ Metrics estese")
        fixes_applied += 1
    else:
        print(f"   ⚠️  Pattern non trovato (potrebbe essere già applicato)")

    # Salva versione migliorata
    improved_path = Path("train_sonar_finetuning_IMPROVED.py")
    with open(improved_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n" + "=" * 70)
    print(f"✅ FIX APPLICATI: {fixes_applied}/5")
    print(f"=" * 70)

    print(f"\n📁 Files generati:")
    print(f"   Original: train_sonar_finetuning.py")
    print(f"   Backup:   {backup_path}")
    print(f"   Improved: {improved_path}")

    if fixes_applied >= 3:
        print(f"\n✅ Script migliorato creato con successo!")
        print(f"\n📊 Modifiche principali:")
        print(f"   • Normalizzazione L2 output encoder → scala corretta")
        print(f"   • Cosine Loss (1 - similarity) → loss interpretabile")
        print(f"   • Gradient monitoring → rileva collapse/esplosione")
        print(f"   • Logging esteso → loss, grad_norm, cosine_sim")
        print(f"   • Metrics validation → BLEU + cosine similarity")

        print(f"\n💡 Prossimi passi:")
        print(f"   1. Test veloce:")
        print(f"      python {improved_path} --epochs 5 --max_samples 50 \\")
        print(f"             --output_dir checkpoints/test_improved")
        print(f"\n   2. Se funziona, sostituisci l'originale:")
        print(f"      mv train_sonar_finetuning.py train_sonar_finetuning_OLD.py")
        print(f"      mv {improved_path} train_sonar_finetuning.py")
        print(f"\n   3. Rilancia full training (50 epochs)")
    else:
        print(f"\n⚠️  Solo {fixes_applied}/5 fix applicati")
        print(f"   Verifica che lo script originale sia nella versione corretta")
        print(f"   Oppure i fix potrebbero essere già stati applicati!")

    print(f"\n" + "=" * 70)

    return fixes_applied >= 3


if __name__ == "__main__":
    success = apply_fixes()
    exit(0 if success else 1)
