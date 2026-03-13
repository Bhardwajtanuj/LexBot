"""
utils/rag_utils.py
Document ingestion → chunking → FAISS vector store → retrieval.
"""

import os
import pickle
import logging
from pathlib import Path

import numpy as np
from config.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS

logger = logging.getLogger(__name__)

STORE_PATH = Path("docs_store/index.pkl")


# ── text extraction ────────────────────────────────────────────────────────────
def extract_text_from_file(file) -> str:
    """Accept a Streamlit UploadedFile and return raw text."""
    try:
        name = file.name.lower()
        if name.endswith(".pdf"):
            import pdfplumber
            text_parts = []
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        text_parts.append(t)
            return "\n".join(text_parts)
        if name.endswith(".txt"):
            return file.read().decode("utf-8", errors="ignore")
        if name.endswith(".docx"):
            import docx
            doc = docx.Document(file)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return ""
    except Exception as e:
        logger.error(f"Text extraction failed for {file.name}: {e}")
        return ""


# ── chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 30]


# ── build / update vector store ───────────────────────────────────────────────
def build_vector_store(texts: list[str]) -> dict:
    """
    Build a simple in-memory store: {"chunks": [...], "embeddings": np.ndarray}
    We use numpy dot-product similarity instead of FAISS so there are no
    binary dependency headaches on Streamlit Cloud.
    """
    from models.embeddings import embed_texts
    all_chunks = []
    for text in texts:
        all_chunks.extend(chunk_text(text))
    embeddings = embed_texts(all_chunks)
    store = {"chunks": all_chunks, "embeddings": np.array(embeddings)}
    _save_store(store)
    return store


def _save_store(store: dict):
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STORE_PATH, "wb") as f:
        pickle.dump(store, f)


def load_vector_store() -> dict | None:
    try:
        if STORE_PATH.exists():
            with open(STORE_PATH, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
    return None


# ── retrieval ─────────────────────────────────────────────────────────────────
def retrieve_relevant_chunks(query: str, store: dict, top_k: int = TOP_K_RESULTS) -> list[str]:
    """Cosine-similarity search, returns top_k chunk strings."""
    try:
        from models.embeddings import embed_query
        q_vec = embed_query(query)
        embs  = store["embeddings"]               # shape (N, D)
        # normalise
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        e_norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)
        scores = e_norm @ q_norm
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [store["chunks"][i] for i in top_idx]
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []


def build_rag_context(chunks: list[str]) -> str:
    if not chunks:
        return ""
    joined = "\n\n---\n\n".join(chunks)
    return f"Relevant context from uploaded documents:\n\n{joined}"
