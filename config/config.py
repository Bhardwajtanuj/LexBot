# config/config.py
import os

def _get(key: str, fallback: str = "") -> str:
    # called at runtime not import time so st.secrets is ready
    try:
        import streamlit as st
        val = st.secrets.get(key, None)
        if val:
            return str(val)
    except Exception:
        pass
    return os.environ.get(key, fallback)

# key getter functions — called fresh on every LLM request
def get_groq_key():    return _get("GROQ_API_KEY")
def get_openai_key():  return _get("OPENAI_API_KEY")
def get_gemini_key():  return _get("GEMINI_API_KEY")
def get_serper_key():  return _get("SERPER_API_KEY")

# embeddings — runs offline, no key needed
EMBEDDING_MODEL      = "all-MiniLM-L6-v2"
CHUNK_SIZE           = 500
CHUNK_OVERLAP        = 50
TOP_K_RESULTS        = 4

# app defaults
DEFAULT_LLM          = "groq"
DEFAULT_GROQ_MODEL   = "llama3-8b-8192"
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
MAX_HISTORY          = 10
