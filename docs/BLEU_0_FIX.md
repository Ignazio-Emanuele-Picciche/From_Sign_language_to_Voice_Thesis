# 🚨 BLEU 0.00% Fix: Magnitude Mismatch

## 📉 Il Problema
Il training precedente ha mostrato:
- **Loss in calo** (0.8 → 0.6) ✅
- **Cosine Similarity in aumento** (0.2 → 0.4) ✅
- **BLEU fisso a 0.00%** ❌

**Diagnosi:**
Abbiamo forzato la normalizzazione degli embeddings a 1 (`norm=1`).
Tuttavia, il decoder SONAR si aspetta embeddings con una magnitudo molto più alta (tipicamente `sqrt(1024) ≈ 32`).
Fornendo vettori con norma 1, il decoder riceveva segnali "invisibili" (30 volte più piccoli del normale), generando output vuoti o spazzatura.

---

## 🛠️ La Soluzione (Già Applicata)

Ho modificato `train_sonar_finetuning.py` per:

1.  **Rimuovere la normalizzazione forzata** in `forward()`.
    -   Ora il modello può produrre vettori di qualsiasi lunghezza.

2.  **Aggiornare la Loss Function**:
    -   **Cosine Loss**: Continua a ottimizzare la *direzione* (semantica).
    -   **Magnitude Loss**: Aggiunto un termine MSE sulle norme per insegnare al modello la *scala* corretta.
    
    ```python
    loss = loss_cosine + 0.01 * loss_mag
    ```

---

## 🚀 Cosa Fare Ora

1.  **Aggiorna lo script su Colab**:
    ```bash
    cd /content/drive/MyDrive/How2Sign_SONAR
    git pull origin dev
    ```

2.  **Riavvia il Training (Da Zero)**:
    Esegui la stessa cella di prima.

3.  **Cosa Aspettarsi**:
    -   **Loss**: Potrebbe essere diversa (perché ora include il termine magnitudo).
    -   **Sim**: Dovrebbe continuare a salire.
    -   **BLEU**: Dovrebbe finalmente schiodarsi da 0.00%!

---

**Commit:** `fix: Remove forced normalization to fix BLEU 0%`
