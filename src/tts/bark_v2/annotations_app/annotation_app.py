"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             EMOSIGN ANNOTATION TOOL - MULTI USER EDITION                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import os
import random
import csv
from datetime import datetime

# --- CONFIGURAZIONE ---
ASSIGNMENT_DIR = "assignments"
AUDIO_DIR = "output_audio_emosign"
OUTPUT_ANNOTATIONS = "data/human_annotations.csv"


# --- FUNZIONI UTILI ---
def load_user_task(username):
    """Carica il file specifico per l'utente selezionato"""
    filename = f"task_{username.lower()}.csv"
    filepath = os.path.join(ASSIGNMENT_DIR, filename)

    if not os.path.exists(filepath):
        return pd.DataFrame()  # Ritorna vuoto se file non esiste

    df_task = pd.read_csv(filepath)

    # --- FILTRO GIÀ FATTI ---
    # Controlliamo nel file dei risultati se l'utente ha già fatto questi video
    if os.path.exists(OUTPUT_ANNOTATIONS):
        try:
            df_done = pd.read_csv(OUTPUT_ANNOTATIONS)
            # Filtriamo solo le righe fatte da QUESTO utente
            user_done = df_done[df_done["annotator_id"] == username]
            done_ids = user_done["video_name"].unique()

            # Rimuoviamo i video già completati dalla lista da fare
            df_task = df_task[~df_task["video_name"].isin(done_ids)]
        except Exception as e:
            st.warning(f"Errore lettura progressi: {e}")

    return df_task


def save_annotation(data):
    file_exists = os.path.exists(OUTPUT_ANNOTATIONS)
    with open(OUTPUT_ANNOTATIONS, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


# --- INTERFACCIA STREAMLIT ---
st.set_page_config(page_title="EmoSign Annotator", page_icon="👥")

# --- SIDEBAR: LOGIN ---
with st.sidebar:
    st.header("🔐 Login")
    user = st.selectbox("Chi sta annotando?", ["Seleziona...", "Luca", "Daniele"])

    if user == "Seleziona...":
        st.warning("Seleziona il tuo nome per iniziare.")
        st.stop()

    st.success(f"Ciao {user}!")

    # Caricamento Dati Utente
    df = load_user_task(user)
    remaining = len(df)
    st.metric("Video Rimanenti", remaining)

    if st.button("🔄 Ricarica Dati"):
        st.rerun()

# --- MAIN LOGIC ---
st.title(f"📝 EmoSign Blind Test ({user})")

if df.empty:
    st.balloons()
    st.success(f"🎉 Grande {user}! Hai completato tutto il tuo set.")
    st.stop()

# 1. Gestione Stato Campione Corrente
if (
    "current_sample" not in st.session_state
    or st.session_state.get("current_user") != user
):
    # Se cambia utente o non c'è campione, ne pesca uno nuovo
    sample = df.sample(1).iloc[0]
    st.session_state.current_sample = sample
    st.session_state.current_user = user
    # Randomizza modalità
    st.session_state.mode = random.choice(["TEXT_ONLY", "AUDIO_ONLY"])

row = st.session_state.current_sample
mode = st.session_state.mode

# --- DISPLAY AREA ---
st.divider()

if mode == "TEXT_ONLY":
    st.info("📖 **LEGGI IL TESTO**")
    st.markdown(f"### *“{row['original_caption']}”*")

elif mode == "AUDIO_ONLY":
    st.warning("🎧 **ASCOLTA L'AUDIO** (Non leggere la caption!)")
    audio_path = os.path.join(AUDIO_DIR, row["filename"])
    if os.path.exists(audio_path):
        st.audio(audio_path, format="audio/wav")
    else:
        st.error(f"File audio mancante: {row['filename']}")

st.divider()

# --- FORM ---
with st.form("annotation_form"):
    st.write("Valuta il **Sentiment** percepito:")

    cols = st.columns(7)
    labels = [
        "-3\n(Molto Neg)",
        "-2",
        "-1",
        "0\n(Neutro)",
        "+1",
        "+2",
        "+3\n(Molto Pos)",
    ]
    for col, label in zip(cols, labels):
        col.caption(label)

    rating = st.slider("Score", -3, 3, 0, label_visibility="collapsed")

    audio_bad = False
    if mode == "AUDIO_ONLY":
        st.write("---")
        audio_bad = st.checkbox("🚩 Audio Difettoso (Glitch/Rumore)")

    if st.form_submit_button("Salva e Prossimo ➡️", use_container_width=True):
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "annotator_id": user,  # <--- Salviamo chi l'ha fatto
            "video_name": row["video_name"],
            "original_sentiment": row["Sentiment"],
            "presentation_mode": mode,
            "human_rating": rating,
            "is_audio_bad": audio_bad,
            "wer_score": row["WER"],
        }
        save_annotation(data)
        st.success("Salvato!")

        # Reset per caricare il prossimo
        del st.session_state.current_sample
        st.rerun()
