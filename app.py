"""
app.py
──────
RAG Chatbot — Streamlit Cloud ready.
Run locally : streamlit run app.py
Deploy      : push to GitHub → connect on share.streamlit.io
"""

import os
from dotenv import load_dotenv
import streamlit as st
from groq import Groq
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from transformers.utils import logging as transformers_logging
from huggingface_hub import login

# ── Load .env for local development ──────────────────────────
load_dotenv()

# ── API key — works both locally and on Streamlit Cloud ───────
def get_api_key_from_env() -> str:
    try:
        return st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        return os.getenv("GROQ_API_KEY", "")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Context-Aware RAG Chatbot")
st.caption("LangChain · FAISS · Groq · File Upload · Tone · Multilingual")

# ── Constants ─────────────────────────────────────────────────
TONES = {
    "Professional": "You are a professional assistant. Use formal, precise language. Be concise and structured.",
    "Casual":       "You are a friendly assistant. Use simple, conversational language. Be warm and approachable.",
    "Technical":    "You are a technical expert. Use domain-specific terminology. Include details and be thorough.",
    "Simple":       "You are a teacher explaining to a beginner. Use very simple language, short sentences, and analogies.",
    "Socratic":     "You are a Socratic tutor. Guide the user to the answer with thoughtful questions and hints.",
}

LANGUAGES = {
    "English":      "Respond in English.",
    "Roman Urdu":   (
        "Respond in Roman Urdu only. "
        "Roman Urdu means writing Urdu words using English/Latin alphabet letters. "
        "For example: 'Yeh bohat acha hai', 'Mujhe samajh aa gaya', 'Theek hai'. "
        "CRITICAL: Do NOT use any Urdu script (اردو), Arabic script, or any non-Latin characters. "
        "Every single word must be written using only English alphabet (a-z, A-Z). "
        "Write exactly as Pakistanis type Urdu in WhatsApp or SMS messages."
    ),
    "Arabic":       "Respond in Arabic (العربية).",
    "French":       "Respond in French.",
    "Spanish":      "Respond in Spanish.",
    "German":       "Respond in German.",
    "Chinese":      "Respond in Simplified Chinese (中文).",
    "Hindi":        "Respond in Hindi (हिन्दी).",
    "Portuguese":   "Respond in Portuguese.",
    "Turkish":      "Respond in Turkish.",
}

# ── Fetch available Groq models ───────────────────────────────
@st.cache_data(show_spinner=False, ttl=300)
def fetch_groq_models(api_key: str) -> list:
    try:
        client = Groq(api_key=api_key)
        models = client.models.list()
        chat_models = sorted([
            m.id for m in models.data
            if not any(x in m.id for x in ["whisper", "tts", "vision", "guard"])
        ])
        return chat_models if chat_models else ["llama-3.1-8b-instant"]
    except Exception:
        return [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "llama3-8b-8192",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ]

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    env_key = get_api_key_from_env()
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=env_key,
        help="Free key at https://console.groq.com"
    )

    st.divider()

    st.subheader("🧠 Model")
    if api_key:
        with st.spinner("Fetching models..."):
            available_models = fetch_groq_models(api_key)
        default_idx = next(
            (i for i, m in enumerate(available_models) if "llama-3.1-8b-instant" in m), 0
        )
        selected_model = st.selectbox(
            "Choose model", available_models, index=default_idx,
            help="Fetched live from your Groq account."
        )
        st.caption(f"_`{selected_model}`_")
    else:
        selected_model = "llama-3.1-8b-instant"
        st.info("Enter API key to load available models.")

    st.divider()

    st.subheader("🎭 Tone")
    selected_tone = st.selectbox("Choose tone", list(TONES.keys()), index=0)
    st.caption(f"_{TONES[selected_tone]}_")

    st.divider()

    st.subheader("🌍 Language")
    selected_language = st.selectbox("Choose language", list(LANGUAGES.keys()), index=0)
    if selected_language == "Roman Urdu":
        st.caption("_Latin letters only — WhatsApp style Urdu_")

    st.divider()

    st.subheader("📂 Knowledge Base")
    upload_mode = st.radio(
        "Source",
        ["Default (knowledge_base.txt)", "Upload your own file"],
        index=0
    )
    uploaded_file = None
    if upload_mode == "Upload your own file":
        uploaded_file = st.file_uploader(
            "Upload PDF or TXT", type=["pdf", "txt"]
        )
        if uploaded_file:
            st.success(f"✅ {uploaded_file.name}")

    st.divider()

    st.subheader("🔍 Retrieval")
    st.metric("Candidates fetched", 20)
    st.metric("Used for answer", 5)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages      = []
        st.session_state.chat_history  = ""
        st.session_state.vectorstore   = None
        st.session_state.loaded_file   = None
        st.rerun()

    st.caption("Task 4 — DevelopersHub Internship")

# ── Guard ─────────────────────────────────────────────────────
if not api_key:
    st.warning("👈 Enter your Groq API key in the sidebar to start.")
    st.markdown("Get a **free** key at [console.groq.com](https://console.groq.com)")
    st.stop()

# ── Helpers ───────────────────────────────────────────────────
def load_documents(uploaded_file=None) -> list:
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            docs = []
            file_bytes = uploaded_file.read()
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page_num, page in enumerate(doc):
                    text = page.get_text()
                    docs.append(Document(
                        page_content=text,
                        metadata={"source": uploaded_file.name, "page": page_num + 1}
                    ))
            return docs
        text = uploaded_file.read().decode("utf-8")
        return [Document(page_content=text, metadata={"source": uploaded_file.name})]
    with open("knowledge_base.txt", "r", encoding="utf-8") as f:
        text = f.read()
    return [Document(page_content=text, metadata={"source": "knowledge_base.txt"})]


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    transformers_logging.disable_progress_bar()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_vectorstore(documents: list, embeddings) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)


def build_chain(api_key, model, tone, language):
    llm = ChatGroq(model=model, temperature=0.2, max_tokens=768, api_key=api_key)

    roman_extra = (
        "\nREMINDER: Every word MUST use only English/Latin letters (a-z). "
        "No Urdu script at all. Example: 'Yeh bohat acha explanation hai.'"
        if language == "Roman Urdu" else ""
    )

    system = f"""{TONES[tone]}
{LANGUAGES[language]}{roman_extra}

Use ONLY the context below to answer. If the answer is not in the context, say so clearly.

Context:
{{context}}

Chat History:
{{chat_history}}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Question: {question}")
    ])

    chain = (
        {
            "context": lambda x: x["context"],
            "question": lambda x: x["question"],
            "chat_history": lambda x: x.get("chat_history", ""),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def get_top_chunks(vectorstore, query: str, fetch_k=20, top_k=5):
    results = vectorstore.similarity_search_with_score(query, k=fetch_k)
    results.sort(key=lambda x: x[1])
    return results[:top_k]


# ── Session state ─────────────────────────────────────────────
for key in ["messages", "chat_history", "vectorstore", "loaded_file"]:
    if key not in st.session_state:
        if key == "messages":
            st.session_state[key] = []
        elif key == "chat_history":
            st.session_state[key] = ""
        else:
            st.session_state[key] = None

# ── Rebuild vectorstore on file change ───────────────────────
current_file = uploaded_file.name if uploaded_file else "knowledge_base.txt"
if st.session_state.loaded_file != current_file:
    with st.spinner(f"Processing '{current_file}'..."):
        emb = load_embeddings()
        docs = load_documents(uploaded_file)
        st.session_state.vectorstore = build_vectorstore(docs, emb)
        st.session_state.loaded_file = current_file
        st.session_state.messages     = []
        st.session_state.chat_history = ""

# ── Chain ─────────────────────────────────────────────────────
qa_chain = build_chain(
    api_key, selected_model, selected_tone, selected_language
)

# ── Settings banner ───────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.info(f"📄 **Source:** {current_file}")
c2.info(f"🧠 **Model:** {selected_model}")
c3.info(f"🎭 **Tone:** {selected_tone}")
c4.info(f"🌍 **Language:** {selected_language}")
st.divider()

# ── Chat history display ──────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 Top 5 source chunks"):
                for i, (snippet, score) in enumerate(message["sources"]):
                    st.caption(f"**Chunk {i+1}** — score: `{score}`")
                    st.text(snippet + "...")
                    if i < len(message["sources"]) - 1:
                        st.divider()

# ── Input ─────────────────────────────────────────────────────
placeholder = {
    "Roman Urdu": "Kuch bhi poochein...",
    "Arabic":     "اكتب سؤالك هنا...",
    "French":     "Posez votre question...",
    "Spanish":    "Escribe tu pregunta...",
    "Turkish":    "Sorunuzu yazın...",
}.get(selected_language, f"Ask anything... (answering in {selected_language})")

if prompt := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Fetching 20 → top 5 → {selected_model}..."):
            top_results = get_top_chunks(st.session_state.vectorstore, prompt)
            context = "\n\n".join(doc.page_content for doc, _ in top_results)
            top_snippets = [(doc.page_content[:120], round(float(score), 4)) for doc, score in top_results]
            answer = st.write_stream(qa_chain.stream({
                "question": prompt,
                "context": context,
                "chat_history": st.session_state.chat_history,
            }))

        with st.expander("📄 Top 5 source chunks used"):
            for i, (snippet, score) in enumerate(top_snippets):
                st.caption(f"**Chunk {i+1}** — score: `{score}`")
                st.text(snippet + "...")
                if i < len(top_snippets) - 1:
                    st.divider()

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "sources": top_snippets
    })
    st.session_state.chat_history += f"Human: {prompt}\nAssistant: {answer}\n"
