"""
models/llm.py
Thin wrappers around three LLM providers.
All functions return (response_text: str, error: str | None).
"""

from config.config import (
    OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY,
    DEFAULT_GROQ_MODEL, DEFAULT_OPENAI_MODEL, DEFAULT_GEMINI_MODEL,
)


# ── OpenAI ─────────────────────────────────────────────────────────────────────
def call_openai(messages: list[dict], model: str = DEFAULT_OPENAI_MODEL) -> tuple[str, str | None]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content.strip(), None
    except Exception as e:
        return "", str(e)


# ── Groq ───────────────────────────────────────────────────────────────────────
def call_groq(messages: list[dict], model: str = DEFAULT_GROQ_MODEL) -> tuple[str, str | None]:
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content.strip(), None
    except Exception as e:
        return "", str(e)


# ── Gemini ─────────────────────────────────────────────────────────────────────
def call_gemini(messages: list[dict], model: str = DEFAULT_GEMINI_MODEL) -> tuple[str, str | None]:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        m = genai.GenerativeModel(model)
        # Gemini expects a plain string or a history-style list
        history = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})
        chat = m.start_chat(history=history[:-1])
        resp = chat.send_message(history[-1]["parts"][0])
        return resp.text.strip(), None
    except Exception as e:
        return "", str(e)


# ── Dispatcher ─────────────────────────────────────────────────────────────────
def get_response(
    messages: list[dict],
    provider: str = "groq",
    model: str | None = None,
) -> tuple[str, str | None]:
    """
    provider: "openai" | "groq" | "gemini"
    model:    pass None to use the default for that provider.
    """
    provider = provider.lower()
    if provider == "openai":
        return call_openai(messages, model or DEFAULT_OPENAI_MODEL)
    if provider == "groq":
        return call_groq(messages, model or DEFAULT_GROQ_MODEL)
    if provider == "gemini":
        return call_gemini(messages, model or DEFAULT_GEMINI_MODEL)
    return "", f"Unknown provider: {provider}"
