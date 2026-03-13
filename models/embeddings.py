"""
models/embeddings.py
Load the sentence-transformer embedding model once and reuse it.
"""

from config.config import EMBEDDING_MODEL

_model = None


def get_embedding_model():
    """Lazy-load so Streamlit doesn't re-init it on every rerun."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            raise RuntimeError(f"Could not load embedding model '{EMBEDDING_MODEL}': {e}")
    return _model


def embed_texts(texts: list[str]) -> list:
    """Return a list of numpy arrays, one per text."""
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def embed_query(text: str):
    """Embed a single query string."""
    return embed_texts([text])[0]
