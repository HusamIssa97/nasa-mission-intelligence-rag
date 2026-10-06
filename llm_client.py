from typing import Dict, List

from openai import OpenAI

from openai_config import resolve_openai_base_url


SYSTEM_PROMPT = """You are a NASA mission expert assistant specializing in Apollo 11, Apollo 13, and the Challenger missions.

Guidelines:
- Base your answers ONLY on the retrieved context provided below. Cite sources when possible (mission, document/source name).
- If the context is insufficient or does not contain the answer, clearly say you do not have enough information in the provided documents.
- Do not invent facts, dates, crew names, or technical details that are not supported by the context.
- Be clear, concise, and helpful. Prefer factual, sourced statements over speculation.
- When conversation history is present, stay consistent with prior turns while still grounding new claims in the latest context.
"""


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-3.5-turbo",
) -> str:
    """Generate response using OpenAI with retrieved context and conversation history."""

    # System prompt positions the assistant as a NASA expert who cites sources
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Attach retrieved context as a dedicated system message when available
    if context and context.strip():
        messages.append(
            {
                "role": "system",
                "content": (
                    "Use the following retrieved NASA document context to answer the user. "
                    "Cite sources from this context when relevant.\n\n"
                    f"{context}"
                ),
            }
        )
    else:
        messages.append(
            {
                "role": "system",
                "content": (
                    "No retrieved document context was provided for this turn. "
                    "Tell the user you lack sufficient source material and avoid unsupported claims."
                ),
            }
        )

    # Keep only role + content turns; skip malformed entries
    for turn in conversation_history or []:
        role = turn.get("role")
        content = turn.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": user_message})

    client_kwargs = {"api_key": openai_key}
    base_url = resolve_openai_base_url(openai_key)
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2,
        max_tokens=800,
    )

    return response.choices[0].message.content.strip()
