# models/llm.py
# one function per provider, all return (text, error)
# imports happen inside each fn so unused providers don't need to be installed
# keys are fetched fresh on every call via getter functions

from config.config import (
    get_groq_key, get_openai_key, get_gemini_key,
    DEFAULT_GROQ_MODEL, DEFAULT_OPENAI_MODEL, DEFAULT_GEMINI_MODEL,
)


def call_openai(messages: list[dict], model: str = DEFAULT_OPENAI_MODEL) -> tuple[str, str | None]:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=get_openai_key())
        resp = client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content.strip(), None
    except Exception as e:
        return "", str(e)


def call_groq(messages: list[dict], model: str = DEFAULT_GROQ_MODEL) -> tuple[str, str | None]:
    try:
        from groq import Groq
        key = get_groq_key()
        if not key:
            return "", "GROQ_API_KEY is not set. Add it in Streamlit Cloud → Settings → Secrets."
        client = Groq(api_key=key)
        resp = client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content.strip(), None
    except Exception as e:
        return "", str(e)


def call_gemini(messages: list[dict], model: str = DEFAULT_GEMINI_MODEL) -> tuple[str, str | None]:
    try:
        import google.generativeai as genai
        genai.configure(api_key=get_gemini_key())
        m = genai.GenerativeModel(model)
        # gemini uses "model" instead of "assistant" for the role name
        history = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})
        chat = m.start_chat(history=history[:-1])
        resp = chat.send_message(history[-1]["parts"][0])
        return resp.text.strip(), None
    except Exception as e:
        return "", str(e)


def get_response(
    messages: list[dict],
    provider: str = "groq",
    model: str | None = None,
) -> tuple[str, str | None]:
    # route to the right provider
    provider = provider.lower()
    if provider == "openai":
        return call_openai(messages, model or DEFAULT_OPENAI_MODEL)
    if provider == "groq":
        return call_groq(messages, model or DEFAULT_GROQ_MODEL)
    if provider == "gemini":
        return call_gemini(messages, model or DEFAULT_GEMINI_MODEL)
    return "", f"unknown provider: {provider}"
