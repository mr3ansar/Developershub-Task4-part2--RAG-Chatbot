# Task 4: Context-Aware Chatbot Using RAG

## Objective
Build a conversational chatbot that remembers context and retrieves answers from a vectorized document store using LangChain and Retrieval-Augmented Generation (RAG).

---

## Dataset
**Custom AI/ML Knowledge Base** (`knowledge_base.txt`)
- 15 topic paragraphs covering: AI, ML, Deep Learning, NLP, Transformers, LLMs, RAG, FAISS, LangChain, Embeddings, Groq, Streamlit
- Swap this file with any text/PDF corpus to adapt the chatbot to any domain

---

## Methodology / Approach

### 1. Document Processing
- Loaded custom knowledge base text file
- Split into overlapping chunks using `RecursiveCharacterTextSplitter`
- chunk_size=500, chunk_overlap=50

### 2. Document Embedding & Vector Store
- Embedded chunks using `sentence-transformers/all-MiniLM-L6-v2` (384-dim vectors)
- Indexed with FAISS for fast similarity search
- Saved index locally to `./faiss_index/`

### 3. RAG Chain with Memory
- Retriever fetches top-3 most relevant chunks per query
- `ConversationBufferMemory` stores full chat history
- `ConversationalRetrievalChain` wires retriever + memory + LLM together

### 4. LLM — Groq (Free)
- Model: `llama-3.1-8b-instant` via Groq API
- Temperature: 0.2 for factual, grounded responses
- Get a free API key at: https://console.groq.com

### 5. Deployment
- Interactive Streamlit chatbot with chat history display
- Source chunk expander shows which document section was used
- Clear chat button resets memory and conversation

---

## Key Observations
- RAG grounds LLM responses in real documents, significantly reducing hallucinations
- Conversation memory enables multi-turn dialogue and follow-up questions
- The pipeline is fully modular — swap the corpus, embedding model, or LLM independently
- FAISS enables sub-millisecond retrieval on the local index

---

## How to Run

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/developershub-aiml-internship.git
cd task4-rag-chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up your API key
cp .env.example .env
# Edit .env and add your Groq API key

# 4. Run the notebook (builds + tests the pipeline)
jupyter notebook rag_chatbot.ipynb

# 5. Launch the Streamlit app
streamlit run app.py
```

### Get a Free Groq API Key
1. Go to https://console.groq.com
2. Sign up (free)
3. Create an API key
4. Add it to your `.env` file or paste in the Streamlit sidebar

---

## Project Structure

```
task4-rag-chatbot/
├── rag_chatbot.ipynb       <- Pipeline notebook
├── app.py                  <- Streamlit chatbot
├── knowledge_base.txt      <- Custom corpus
├── faiss_index/            <- Saved vector store (generated on first run)
├── .env.example            <- API key template
├── requirements.txt
└── README.md
```

---

## Skills Demonstrated
- Conversational AI development with LangChain
- Document embedding and FAISS vector search
- Retrieval-Augmented Generation (RAG)
- Conversation memory for multi-turn dialogue
- LLM integration and Streamlit deployment
