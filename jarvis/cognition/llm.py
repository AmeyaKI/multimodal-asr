"""Local Ollama LLM with optional Gemini fallback."""

from __future__ import annotations

import json
import re
from typing import Any

from jarvis.config import get_settings


def get_chat_model(model: str | None = None, temperature: float = 0.2):
    settings = get_settings()
    if settings.jarvis_llm_fallback == "gemini" and settings.gemini_api_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                google_api_key=settings.gemini_api_key,
                temperature=temperature,
            )
        except ImportError:
            pass
    from langchain_ollama import ChatOllama

    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=model or settings.ollama_model,
        temperature=temperature,
    )


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


async def ainvoke_json(prompt: str, system: str, model: str | None = None) -> dict[str, Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_chat_model(model)
    messages = [SystemMessage(content=system), HumanMessage(content=prompt)]
    try:
        resp = await llm.ainvoke(messages)
        content = resp.content if hasattr(resp, "content") else str(resp)
        return extract_json(str(content))
    except Exception:
        return {}
