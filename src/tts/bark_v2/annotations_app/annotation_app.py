"""
╔══════════════════════════════════════════════════════════════════════════════╗
║             EMOSIGN ANNOTATION TOOL - MULTI USER (RADIO BUTTONS)             ║
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
OUTPUT_ANNOTATIONS = "human_annotations.csv"


# --- FUNZIONI UTILI ---
def load_user_task(username):
    filename = f"task_{username.lower()}.csv"
    filepath = os.path.join(ASSIGNMENT_DIR, filename)

    if not os.path.exists(filepath):
        return pd.DataFrame()

    df_task = pd.read_csv(filepath)

    if os.path.exists(OUTPUT_ANNOTATIONS):
        try:
            df_done = pd.read_csv(OUTPUT_ANNOTATIONS)
            user_done = df_done[df_done["annotator_id"] == username]
            done_ids = user_done["video_name"].unique()
            df_task = df_task[~df_task["video_name"].isin(done_ids)]
        except Exception:
            pass

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

# --- CSS PER INGRANDIRE I RADIO BUTTON ---
# Questo piccolo trucco CSS rende i pulsanti più grandi e distanziati per cliccare meglio
st.markdown(
    """
<style>
div[role="radiogroup"] > label > div:first-child {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding-right: 10px;
}
div[role="radiogroup"] {
    gap: 15px;
    justify-content: center;
}
</style>
""",
    unsafe_allow_html=True,
)

# --- SIDEBAR: LOGIN ---
with st.sidebar:
    st.header("🔐 Login")
    user = st.selectbox("Chi sta annotando?", ["Seleziona...", "Luca", "Daniele"])

    if user == "Seleziona...":
        st.warning("Seleziona il tuo nome per iniziare.")
        st.stop()

    st.success(f"Ciao {user}!")
    df = load_user_task(user)
    remaining = len(df)
    st.metric("Video Rimanenti", remaining)

    if st.button("🔄 Ricarica"):
        st.rerun()

# --- MAIN LOGIC ---
st.title(f"📝 EmoSign ({user})")

if df.empty:
    st.balloons()
    st.success("Hai finito tutto! 🎉")
    st.stop()

if (
    "current_sample" not in st.session_state
    or st.session_state.get("current_user") != user
):
    sample = df.sample(1).iloc[0]
    st.session_state.current_sample = sample
    st.session_state.current_user = user
    st.session_state.mode = random.choice(["TEXT_ONLY", "AUDIO_ONLY"])

row = st.session_state.current_sample
mode = st.session_state.mode

# --- DISPLAY AREA ---
st.divider()
if mode == "TEXT_ONLY":
    st.info("📖 **LEGGI**")
    st.markdown(f"### *“{row['original_caption']}”*")
elif mode == "AUDIO_ONLY":
    st.warning("🎧 **ASCOLTA** (Non leggere!)")
    audio_path = os.path.join(AUDIO_DIR, row["filename"])
    if os.path.exists(audio_path):
        st.audio(audio_path, format="audio/wav")
    else:
        st.error(f"Audio mancante: {row['filename']}")
st.divider()

# --- FORM DI VALUTAZIONE ---
with st.form("annotation_form"):
    st.write("##### Qual è il Sentiment?")

    # MAPPING LABEL PER VISUALIZZAZIONE
    # Usiamo una lista ordinata per i radio button
    options = [-3, -2, -1, 0, 1, 2, 3]

    # Funzione per mostrare etichette carine
    def format_option(opt):
        if opt == -3:
            return "🔴 -3 (Molto Neg)"
        if opt == -2:
            return "-2"
        if opt == -1:
            return "-1"
        if opt == 0:
            return "⚪ 0 (Neutro)"
        if opt == 1:
            return "+1"
        if opt == 2:
            return "+2"
        if opt == 3:
            return "🟢 +3 (Molto Pos)"
        return str(opt)

    # RADIO BUTTON ORIZZONTALE
    # index=3 imposta il valore di default a "0" (che è il quarto elemento della lista)
    rating = st.radio(
        "Seleziona un valore:",
        options=options,
        format_func=format_option,
        horizontal=True,
        index=3,
        label_visibility="collapsed",
    )

    st.write("")  # Spaziatura

    audio_bad = False
    if mode == "AUDIO_ONLY":
        st.write("---")
        audio_bad = st.checkbox("🚩 Audio Difettoso (Glitch/Rumore)")

    # Pulsante grande per inviare
    if st.form_submit_button("SALVA E PROSSIMO ➡️", use_container_width=True):
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "annotator_id": user,
            "video_name": row["video_name"],
            "original_sentiment": row["Sentiment"],
            "presentation_mode": mode,
            "human_rating": rating,
            "is_audio_bad": audio_bad,
            "wer_score": row["WER"],
        }
        save_annotation(data)
        st.success("Salvato!")
        del st.session_state.current_sample
        st.rerun()
