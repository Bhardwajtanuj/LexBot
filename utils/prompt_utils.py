"""
utils/prompt_utils.py
Builds the message list that gets sent to any LLM provider.
"""

SYSTEM_TEMPLATE = """You are LexBot, an intelligent legal document assistant built for law firms, paralegals, and individuals navigating legal questions.

Your job is to:
- Answer questions about legal documents the user uploads
- Explain legal concepts in plain language
- Flag potential issues or clauses that deserve attention
- Search for up-to-date legal information when relevant

Tone: professional but approachable. No jargon without explanation.
Response mode: {mode}

{rag_context}

{web_context}

Ground rules:
- Always remind users that your answers are informational, not formal legal advice.
- If context from documents is provided, prefer that over general knowledge.
- If web results are provided, cite the source at the end of your answer.
- Never make up case names, statutes, or citations."""


def build_messages(
    user_query: str,
    history: list[dict],
    mode: str = "detailed",
    rag_context: str = "",
    web_context: str = "",
) -> list[dict]:
    """
    history: list of {"role": "user"|"assistant", "content": "..."}
    mode:    "concise" | "detailed"
    """
    mode_instruction = (
        "Keep your response SHORT and to the point — 3-5 sentences max."
        if mode == "concise"
        else "Give a thorough, well-structured answer with context and explanation where helpful."
    )

    system_content = SYSTEM_TEMPLATE.format(
        mode=mode_instruction,
        rag_context=rag_context,
        web_context=web_context,
    ).strip()

    messages = [{"role": "system", "content": system_content}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_query})
    return messages


def trim_history(history: list[dict], max_turns: int = 10) -> list[dict]:
    """Keep only the last N turns to avoid exceeding context limits."""
    return history[-(max_turns * 2):]
