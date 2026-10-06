"""Shared OpenAI / Vocareum client configuration."""

from __future__ import annotations

import os
from typing import Optional

VOCAREUM_BASE_URL = "https://openai.vocareum.com/v1"


def resolve_openai_base_url(api_key: Optional[str] = None) -> Optional[str]:
    """Return Vocareum proxy URL for voc- keys; otherwise None (default OpenAI)."""
    override = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    if override:
        return override

    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("CHROMA_OPENAI_API_KEY") or ""
    if key.startswith("voc"):
        return VOCAREUM_BASE_URL
    return None
