import os

def _get(key: str, fallback: str = "") -> str:
    """
    Try Streamlit secrets first (works on Streamlit Cloud),
    then fall back to environment variables.
    """
    try:
        import streamlit as st
        return st.secrets.get(key, os.environ.get(key, fallback))
    except Exception:
        return os.environ.get(key, fallback)

# ── LLM providers ─────────────────────────────────────────────────────────────
OPENAI_API_KEY   = _get("OPENAI_API_KEY")
GROQ_API_KEY     = _get("GROQ_API_KEY")
GEMINI_API_KEY   = _get("GEMINI_API_KEY")

# ── Web Search (Serper.dev) ───────────────────────────────────────────────────
SERPER_API_KEY   = _get("SERPER_API_KEY")

# ── Embedding / vector store ──────────────────────────────────────────────────
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # runs fully offline, no key needed
CHUNK_SIZE       = 500
CHUNK_OVERLAP    = 50
TOP_K_RESULTS    = 4

# ── App behaviour ──────────────────────────────────────────────────────────────
DEFAULT_LLM      = "groq"           # "openai" | "groq" | "gemini"
DEFAULT_GROQ_MODEL   = "llama3-8b-8192"
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
MAX_HISTORY      = 10               # turns kept in memory
