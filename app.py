"""
app.py  —  LexBot: Legal Document Assistant
Streamlit chatbot with RAG, live web search, and concise/detailed response modes.
"""

import streamlit as st
import logging

from config.config import DEFAULT_LLM, MAX_HISTORY
from models.llm import get_response
from models.embeddings import get_embedding_model
from utils import (
    extract_text_from_file,
    build_vector_store,
    load_vector_store,
    retrieve_relevant_chunks,
    build_rag_context,
    web_search,
    format_search_results,
    should_search,
    build_messages,
    trim_history,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexBot — Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── session state defaults ─────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages":     [],
        "vector_store": None,
        "doc_names":    [],
        "provider":     DEFAULT_LLM,
        "mode":         "detailed",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/scales.png",
        width=64,
    )
    st.title("LexBot ⚖️")
    st.caption("Your AI-powered legal document assistant")
    st.divider()

    # LLM provider
    st.subheader("🤖 Model")
    provider = st.selectbox(
        "LLM Provider",
        ["groq", "openai", "gemini"],
        index=["groq", "openai", "gemini"].index(st.session_state.provider),
        key="provider_select",
    )
    st.session_state.provider = provider

    # Response mode
    st.subheader("📝 Response Mode")
    mode = st.radio(
        "How detailed should answers be?",
        ["concise", "detailed"],
        index=0 if st.session_state.mode == "concise" else 1,
        horizontal=True,
    )
    st.session_state.mode = mode

    st.divider()

    # Document upload
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "PDF, TXT, or DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        help="Upload legal documents for the chatbot to reference.",
    )

    if uploaded_files:
        new_names = [f.name for f in uploaded_files]
        if new_names != st.session_state.doc_names:
            with st.spinner("Reading and indexing documents…"):
                try:
                    texts = [extract_text_from_file(f) for f in uploaded_files]
                    texts = [t for t in texts if t.strip()]
                    if texts:
                        # warm up the embedding model once
                        get_embedding_model()
                        store = build_vector_store(texts)
                        st.session_state.vector_store = store
                        st.session_state.doc_names    = new_names
                        st.success(f"Indexed {len(texts)} document(s) ✓")
                    else:
                        st.warning("Couldn't extract text from the uploaded files.")
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
                    logger.error(e)

    if st.session_state.doc_names:
        st.caption("Indexed documents:")
        for name in st.session_state.doc_names:
            st.markdown(f"• {name}")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "LexBot provides informational responses only — not formal legal advice. "
        "Consult a qualified attorney for legal decisions."
    )


# ── main area ──────────────────────────────────────────────────────────────────
st.title("⚖️ LexBot — Legal Document Assistant")
st.caption(
    "Ask about contract clauses, legal terms, recent case law, or upload your own documents."
)

# render existing chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# chat input
user_input = st.chat_input("Ask a legal question or describe a document issue…")

if user_input:
    # show user bubble immediately
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # ── RAG retrieval ──────────────────────────────────────────────────────────
    rag_context = ""
    if st.session_state.vector_store:
        try:
            chunks = retrieve_relevant_chunks(user_input, st.session_state.vector_store)
            rag_context = build_rag_context(chunks)
        except Exception as e:
            logger.error(f"RAG retrieval error: {e}")

    # ── web search (conditional) ───────────────────────────────────────────────
    web_context = ""
    search_results = []
    if should_search(user_input, st.session_state.messages):
        try:
            search_results = web_search(user_input)
            web_context    = format_search_results(search_results)
        except Exception as e:
            logger.error(f"Web search error: {e}")

    # ── build message list + call LLM ──────────────────────────────────────────
    history  = trim_history(st.session_state.messages[:-1], MAX_HISTORY)
    messages = build_messages(
        user_query  = user_input,
        history     = history,
        mode        = st.session_state.mode,
        rag_context = rag_context,
        web_context = web_context,
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                reply, error = get_response(
                    messages  = messages,
                    provider  = st.session_state.provider,
                )
                if error:
                    reply = (
                        f"⚠️ The model returned an error: `{error}`\n\n"
                        "Check that your API key is set correctly in the sidebar or `.env`."
                    )
            except Exception as e:
                reply = f"⚠️ Unexpected error: {e}"
                logger.error(e)

        st.markdown(reply)

        # show source snippets in an expander if RAG was used
        if rag_context:
            with st.expander("📎 Document context used", expanded=False):
                st.markdown(rag_context[:1500] + ("…" if len(rag_context) > 1500 else ""))

        # show web sources if search was triggered
        if search_results:
            with st.expander("🌐 Web sources", expanded=False):
                for r in search_results:
                    st.markdown(f"**{r['title']}**  \n{r['snippet']}  \n[{r['link']}]({r['link']})")

    st.session_state.messages.append({"role": "assistant", "content": reply})
