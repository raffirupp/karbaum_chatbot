import streamlit as st
import numpy as np
import os
import pickle
import json
from datetime import datetime
from openai import OpenAI

# -----------------------------------
# 1️⃣ Setup
# -----------------------------------
st.set_page_config(page_title="Karriere-Coach Chatbot", layout="wide")
st.title("💬 Karriere-Coach Chatbot (im Stil von Dr. Markus Karbaum)")

api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=api_key)

# -----------------------------------
# 2️⃣ Artikel laden
# -----------------------------------
@st.cache_data
def lade_artikel():
    with open("articles.json", "r", encoding="utf-8") as f:
        return json.load(f)

artikel = lade_artikel()

# -----------------------------------
# 3️⃣ Embeddings erzeugen & cachen
# -----------------------------------
def erzeuge_embeddings(artikel, _client, batch_size=20):
    texte = []
    embeddings = []
    titles = []
    urls = []

    for a in artikel:
        text = f"{a['title']}\n\n{a['content']}"
        texte.append(text)
        titles.append(a['title'])
        urls.append(a['url'])

    progress_bar = st.progress(0)
    total = len(texte)

    for i in range(0, total, batch_size):
        batch = texte[i:i+batch_size]
        response = _client.embeddings.create(
            model="text-embedding-ada-002",
            input=batch
        )
        for emb in response.data:
            embeddings.append(emb.embedding)
        progress_bar.progress(min((i + batch_size) / total, 1.0))

    progress_bar.empty()
    return texte, np.array(embeddings), titles, urls


def lade_oder_erzeuge_embeddings(force_neu=False):
    cache_file = "embeddings_cache.pkl"
    timestamp_file = "embeddings_timestamp.txt"

    if not force_neu and os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            texte, embeddings, titles, urls = pickle.load(f)
        with open(timestamp_file, "r") as f:
            timestamp = f.read()
        return texte, embeddings, titles, urls, timestamp

    texte, embeddings, titles, urls = erzeuge_embeddings(artikel, client)

    with open(cache_file, "wb") as f:
        pickle.dump((texte, embeddings, titles, urls), f)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(timestamp_file, "w") as f:
        f.write(timestamp)

    return texte, embeddings, titles, urls, timestamp


# -----------------------------------
# 4️⃣ Hilfsfunktion: semantische Suche
# -----------------------------------
def finde_relevante_texte(prompt, texte, embeddings, client, top_k=3):
    """Finde die relevantesten Artikel zum Nutzerprompt."""
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=[prompt]
    )
    query_emb = np.array(response.data[0].embedding)
    ähnlichkeiten = np.dot(embeddings, query_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
    )
    beste_indices = np.argsort(ähnlichkeiten)[-top_k:][::-1]
    relevante_texte = [texte[i] for i in beste_indices]
    relevante_urls = [urls[i] for i in beste_indices]
    return relevante_texte, relevante_urls


# -----------------------------------
# 5️⃣ Layout mit zwei Spalten
# -----------------------------------
col1, col2 = st.columns([1, 3], gap="large")

# ---------- Linke Spalte ----------
with col1:
    st.markdown("""
    <div style='background-color:#f0f2f6; padding:18px; border-radius:10px;'>
        <h3>ℹ️ Über dieses Projekt</h3>
        <p>
        Dieser Chatbot basiert auf den öffentlichen Blogartikeln von 
        <a href='https://karrierecoaching.eu/' target='_blank'>Dr. Markus Karbaum</a> 
        und nutzt KI, um Fragen rund um Karriere, Führung und Bewerbung 
        anhand dieser Texte zu beantworten.
        </p>
        <p>
        <b>Hinweis:</b> Dieses Tool ersetzt kein persönliches Coaching 
        und dient ausschließlich Demonstrations- und Lernzwecken.
        </p>
        <hr>
        <p style='font-size: 0.9em;'>
        📬 Bei Fragen oder Anmerkungen gerne an 
        <a href='mailto:raffael.ruppert@sciencespo.fr'>Raffael Ruppert</a>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧠 Daten & Embeddings")

    if st.button("Embeddings aktualisieren"):
        with st.spinner("Die Wissensbasis wird aktualisiert – bitte etwas Geduld ..."):
            texte, embeddings, titles, urls, timestamp = lade_oder_erzeuge_embeddings(force_neu=True)
            st.success(f"✅ Embeddings wurden neu erstellt am {timestamp}")
    else:
        if os.path.exists("embeddings_cache.pkl"):
            texte, embeddings, titles, urls, timestamp = lade_oder_erzeuge_embeddings()
            st.info(f"Embeddings aus Cache geladen (Stand: {timestamp})")
        else:
            st.warning("Noch keine Embeddings vorhanden. Klicke auf den Button, um sie zu erstellen.")


# ---------- Rechte Spalte ----------
with col2:
    st.markdown("""
    <style>
    .chat-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .user-msg {
        background-color: #e3f2fd;
        padding: 10px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .coach-msg {
        background-color: #ede7f6;
        padding: 10px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .send-btn {
        background-color: #4a4efc;
        color: white;
        border: none;
        border-radius: 50%;
        width: 42px;
        height: 42px;
        font-size: 20px;
        cursor: pointer;
    }
    .send-btn:hover {
        background-color: #3236cc;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------- Session States ----------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "prompt_from_suggestion" not in st.session_state:
        st.session_state.prompt_from_suggestion = None
    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    # ---------- Eingabefeld + Button nebeneinander ----------
    col_input, col_button = st.columns([6, 1])
    with col_input:
        user_input = st.text_area(
            "💬 Ihre Frage an den Karriere-Coach:",
            height=80,
            placeholder="z. B. Wie plane ich eine Solo-Selbständigkeit?",
            label_visibility="collapsed",
            key="user_input",
        )
    with col_button:
        if st.button("➡️", help="Frage absenden", use_container_width=True):
            st.session_state.submitted = True

    # Falls Vorschlag angeklickt wurde
    if st.session_state.prompt_from_suggestion:
        user_input = st.session_state.prompt_from_suggestion
        st.session_state.prompt_from_suggestion = None
        st.session_state.submitted = True

    # ---------- Anfrage an Modell ----------
    if st.session_state.submitted and user_input.strip():
        relevante_infos, relevante_urls = finde_relevante_texte(user_input, texte, embeddings, client)

        system_prompt = (
            "Du bist ein erfahrener Karriere-Coach im Stil von Dr. Markus Karbaum.\n"
            "Sprich in der Sie-Form, bleibe ruhig, professionell und empathisch.\n"
            "Gib praxisnahe, kurze und konkrete Empfehlungen.\n"
            "Wenn du keine sichere Antwort weißt, sag das offen.\n"
            "Beende deine Antwort optional mit passenden Blogvorschlägen (mit 2). Überprüfe bitte, dass es diese auch wirklich von Dr. Markus Karbaum in der JSON gibt und erfinde keine."
        )

        previous_dialogue = "\n".join(
            [f"{role}: {msg}" for role, msg in st.session_state.chat_history if role in ("User", "Coach")]
        )
        kontext = "\n\n".join([f"- {t}" for t in relevante_infos])

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{previous_dialogue}\n\nAktuelle Frage: {user_input}\n\nRelevante Artikel:\n{kontext}"}
            ]
        )

        antwort = completion.choices[0].message.content.strip()
        st.session_state.chat_history.append(("User", user_input))
        st.session_state.chat_history.append(("Coach", antwort))
        st.session_state.chat_history.append(("Links", relevante_urls))
        st.session_state.submitted = False
        st.rerun()

    # ---------- Chatverlauf ----------
    st.markdown("### 🧠 Verlauf")
    for role, msg in reversed(st.session_state.chat_history):
        if role == "User":
            st.markdown(f"<div class='user-msg'><b>Sie:</b> {msg}</div>", unsafe_allow_html=True)
        elif role == "Coach":
            st.markdown(f"<div class='coach-msg'><b>Dr. Karbaum-Bot:</b> {msg}</div>", unsafe_allow_html=True)
        elif role == "Links":
            urls = msg
            if urls:
                links_html = "<br>".join([f"🔗 <a href='{u}' target='_blank'>{u}</a>" for u in urls])
                st.markdown(f"<div class='chat-box'><b>Zur weiteren Lektüre:</b><br>{links_html}</div>", unsafe_allow_html=True)

    # ---------- Themenvorschläge ----------
    st.markdown("### 💡 Themenvorschläge")
    beispiele = [
        "Wie gehe ich mit Arbeitsplatzverlust um?",
        "Wie entwickle ich Führungskompetenzen?",
        "Wie kann ich meine Selbstvermarktung verbessern?",
        "Wie plane ich den Wiedereinstieg nach einer Pause?"
    ]
    for b in beispiele:
        if st.button(b):
            st.session_state.prompt_from_suggestion = b
            st.rerun()

    # ---------- Verlauf löschen ----------
    if st.button("🧹 Verlauf löschen"):
        st.session_state.chat_history = []
        st.session_state.submitted = False
        st.rerun()
