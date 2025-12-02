"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        EMOSIGN ANNOTATOR - FULL CROSS EVALUATION (TEXT + AUDIO)              ║
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
def get_pending_tasks(username):
    """
    Genera la lista di TUTTI i task (Testo E Audio) e rimuove quelli già fatti.
    Ritorna:
      1. df_metadata: Il dataframe completo per recuperare caption/sentiment
      2. pending_list: Lista di tuple (video_name, mode) da fare
    """
    # 1. Carica i video assegnati all'utente
    filename = f"task_{username.lower()}.csv"
    filepath = os.path.join(ASSIGNMENT_DIR, filename)

    if not os.path.exists(filepath):
        return pd.DataFrame(), []

    df_metadata = pd.read_csv(filepath)
    # Imposta video_name come indice per lookup veloce
    df_metadata.set_index("video_name", inplace=False)

    # 2. Genera TUTTI i task possibili (2 per ogni video)
    all_tasks = set()
    for video in df_metadata["video_name"].unique():
        all_tasks.add((video, "TEXT_ONLY"))
        all_tasks.add((video, "AUDIO_ONLY"))

    # 3. Controlla cosa è già stato fatto
    if os.path.exists(OUTPUT_ANNOTATIONS):
        try:
            df_done = pd.read_csv(OUTPUT_ANNOTATIONS)
            # Filtra per utente corrente
            user_done = df_done[df_done["annotator_id"] == username]

            # Crea un set di task completati
            for _, row in user_done.iterrows():
                done_tuple = (row["video_name"], row["presentation_mode"])
                if done_tuple in all_tasks:
                    all_tasks.remove(done_tuple)
        except Exception:
            pass  # Se il file è corrotto o vuoto, rifacciamo tutto (safe)

    # Converte il set rimasto in una lista per poter fare random.choice
    pending_list = list(all_tasks)
    return df_metadata, pending_list


def save_annotation(data):
    file_exists = os.path.exists(OUTPUT_ANNOTATIONS)
    with open(OUTPUT_ANNOTATIONS, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)


# --- INTERFACCIA STREAMLIT ---
st.set_page_config(page_title="EmoSign Annotator", page_icon="⚖️")

# CSS Radio Button Orizzontali Grandi
st.markdown(
    """
<style>
div[role="radiogroup"] > label > div:first-child {
    background-color: #f0f2f6;
    border-radius: 8px;
    padding: 10px;
    margin-right: 5px;
}
div[role="radiogroup"] {
    gap: 10px;
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
        st.warning("Seleziona il tuo nome.")
        st.stop()

    st.success(f"Ciao {user}!")

    # Caricamento Logica Task
    df_meta, pending_tasks = get_pending_tasks(user)

    remaining = len(pending_tasks)
    st.metric("Valutazioni Rimanenti", remaining)
    st.caption("(Include sia Testo che Audio per ogni video)")

    if st.button("🔄 Ricarica"):
        st.rerun()

# --- MAIN LOGIC ---
st.title(f"📝 EmoSign ({user})")

if not pending_tasks:
    st.balloons()
    st.success("🎉 Hai completato TUTTE le valutazioni (Audio e Testo)!")
    st.stop()

# --- SELEZIONE RANDOM DEL TASK ---
# Se non c'è un task attivo o l'utente è cambiato, ne pesciamo uno nuovo dal calderone
if (
    "current_task" not in st.session_state
    or st.session_state.get("current_user") != user
):
    # Random shuffle assicura che non capitino vicini
    selected_task = random.choice(pending_tasks)
    st.session_state.current_task = selected_task
    st.session_state.current_user = user

# Estrai i dati del task corrente
video_id, mode = st.session_state.current_task

# Recupera i metadati dal dataframe usando il video_id
try:
    # df_meta potrebbe avere indice numerico o video_name, usiamo filtro sicuro
    row_data = df_meta[df_meta["video_name"] == video_id].iloc[0]
except IndexError:
    st.error("Errore nel recupero dati video. Prova a ricaricare.")
    st.stop()

# --- DISPLAY AREA ---
st.divider()

if mode == "TEXT_ONLY":
    st.info("📖 **LEGGI IL TESTO** (Valuta l'emozione scritta)")
    # Mostriamo testo grande
    st.markdown(
        f"<h3 style='text-align: center;'>“{row_data['original_caption']}”</h3>",
        unsafe_allow_html=True,
    )

elif mode == "AUDIO_ONLY":
    st.warning("🎧 **ASCOLTA L'AUDIO** (Valuta l'emozione della voce)")
    st.caption("Non leggere la caption originale, concentrati sul tono.")

    audio_path = os.path.join(AUDIO_DIR, row_data["filename"])
    if os.path.exists(audio_path):
        st.audio(audio_path, format="audio/wav")
    else:
        st.error(f"File audio mancante: {row_data['filename']}")

st.divider()

# --- FORM DI VALUTAZIONE ---
with st.form("annotation_form"):
    st.write(
        f"Che **Sentiment** percepisci da questo {('testo' if mode=='TEXT_ONLY' else 'audio')}?"
    )

    # Opzioni Radio
    options = [-3, -2, -1, 0, 1, 2, 3]

    def format_opt(x):
        if x == -3:
            return "🔴 -3 (Neg)"
        if x == 0:
            return "⚪ 0 (Neutro)"
        if x == 3:
            return "🟢 +3 (Pos)"
        return str(x)

    rating = st.radio(
        "Sentiment:",
        options=options,
        format_func=format_opt,
        horizontal=True,
        index=3,  # Default su 0
        label_visibility="collapsed",
    )

    st.write("")

    # Checkbox solo se audio
    audio_bad = False
    if mode == "AUDIO_ONLY":
        st.write("---")
        audio_bad = st.checkbox("🚩 Audio Difettoso (Glitch/Rumore)")

    # Submit
    if st.form_submit_button("SALVA E PROSSIMO ➡️", use_container_width=True):
        data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "annotator_id": user,
            "video_name": video_id,
            "original_sentiment": row_data["Sentiment"],
            "presentation_mode": mode,
            "human_rating": rating,
            "is_audio_bad": audio_bad,
            "wer_score": row_data["WER"],
        }

        save_annotation(data)
        st.success("Salvato!")

        # Rimuovi task dallo stato per forzare nuovo pescaggio random
        del st.session_state.current_task
        st.rerun()
