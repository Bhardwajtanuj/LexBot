# ⚖️ LexBot — Legal Document Assistant

A Streamlit chatbot that helps users understand legal documents, look up recent case law, and get plain-English explanations of complex legal language.

Built for the **NeoStats AI Engineer Case Study**.

---

## Features

- **RAG (Retrieval-Augmented Generation)** — Upload PDFs, TXT, or DOCX files; LexBot indexes them and answers questions grounded in your actual documents.
- **Live web search** — Automatically searches the web for recent laws, verdicts, and regulations using Serper.dev.
- **Concise / Detailed response modes** — Toggle in the sidebar.
- **Multi-provider LLM support** — Works with Groq (default), OpenAI, and Google Gemini.

---

## Local Setup

```bash
git clone https://github.com/your-username/lexbot.git
cd lexbot

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Copy and fill in your API keys
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml with your actual keys

streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## API Keys You Need

| Key | Where to get it | Required? |
|-----|----------------|-----------|
| `GROQ_API_KEY` | https://console.groq.com | Yes (default LLM) |
| `OPENAI_API_KEY` | https://platform.openai.com | Optional |
| `GEMINI_API_KEY` | https://aistudio.google.com | Optional |
| `SERPER_API_KEY` | https://serper.dev | Optional (web search) |

---

## Project Structure

```
project/
├── config/
│   ├── __init__.py
│   └── config.py          ← API keys, settings (reads from env / Streamlit secrets)
├── models/
│   ├── __init__.py
│   ├── llm.py             ← OpenAI / Groq / Gemini wrappers
│   └── embeddings.py      ← sentence-transformers embedding model
├── utils/
│   ├── __init__.py
│   ├── rag_utils.py       ← Document ingestion, chunking, vector search
│   ├── web_search.py      ← Serper.dev web search
│   └── prompt_utils.py    ← Message list builder
├── .streamlit/
│   ├── config.toml        ← Theme
│   └── secrets.toml.example
├── app.py                 ← Main Streamlit app
├── requirements.txt
└── README.md
```

---



---

## Disclaimer

LexBot provides informational responses only. Nothing it says constitutes formal legal advice. Always consult a qualified attorney for legal decisions.
