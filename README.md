# RAG Chatbot — Context-Aware Q&A with PDF Support

**🔗 Live App:** [https://rag-chatbot-mr3ansar.streamlit.app/](https://rag-chatbot-mr3ansar.streamlit.app/)

A multilingual, tone-aware RAG chatbot built with LangChain, FAISS, Groq, and Streamlit. Upload PDFs or use the built-in knowledge base — the app retrieves relevant chunks and generates grounded answers.

## Features

- **PDF / TXT upload** — load any document as a knowledge base
- **Fast PDF parsing** — uses PyMuPDF (fitz) in-memory, no disk writes
- **Streaming responses** — token-by-token output via `st.write_stream`
- **Multilingual** — English, Roman Urdu, Arabic, French, Spanish, German, Chinese, Hindi, Portuguese, Turkish
- **Tone selection** — Professional, Casual, Technical, Simple, Socratic
- **Groq LLM** — free, fast inference (`llama-3.1-8b-instant` and others)
- **FAISS vector store** — sub-millisecond similarity search on 384-dim embeddings

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API keys
# Create .streamlit/secrets.toml (gitignored) with:
# GROQ_API_KEY = "gsk_..."
# HF_TOKEN = "hf_..."

# 3. Run
streamlit run app.py
```

Get a free Groq key at [console.groq.com](https://console.groq.com).

## Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → connect repo
3. In **Settings → Secrets**, add:
   ```
   GROQ_API_KEY = "gsk_..."
   HF_TOKEN = "hf_..."
   ```
4. Deploy

## Project Structure

```
├── app.py                  # Streamlit application
├── knowledge_base.txt      # Default knowledge base (AI/ML topics)
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata (uv)
├── .streamlit/
│   └── config.toml         # Theme & server settings
└── .gitignore
```

## Tech Stack

LangChain · FAISS · sentence-transformers · PyMuPDF · Groq · Streamlit
