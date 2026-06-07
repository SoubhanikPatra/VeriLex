"""
OpenAI fallback backend for the LLM adapter.

This backend uses the official `openai` Python package to query a chat
completion endpoint. It expects `OPENAI_API_KEY` to be available in the
environment (the project already loads `.env` in `config/settings.py`).

This is intentionally small: the adapter will only call this backend when
Groq (or earlier backends) fail. The implementation converts a single
prompt string into a single user message for OpenAI.
"""
from __future__ import annotations

import os
from typing import Any

import openai

from config import settings


class Backend:
    """OpenAI backend wrapper exposing `invoke(prompt: str) -> str`.

    This class is lightweight and intended for use as a fallback.
    """

    def __init__(self):
        # Configure the OpenAI API key if present in the environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        openai.api_key = api_key
        # Model can be overridden via OPENAI_MODEL in config
        self.model = settings.OPENAI_MODEL

    def invoke(self, prompt: str) -> str:
        """Call OpenAI chat completion and return the assistant content.

        Note: the adapter handles timeouts and retries externally.
        """
        # Build a single-user message payload following the Chat API shape.
        # We keep the payload minimal because the adapter flattens message
        # sequences into a single textual prompt before calling backends.
        messages = [{"role": "user", "content": prompt}]

        # Call the OpenAI Chat Completions endpoint. The adapter will handle
        # retries and timeouts; if OpenAI raises a network or API error this
        # method will propagate and the adapter can perform fallback logic.
        resp = openai.chat.completions.create(model=self.model, messages=messages)

        # Parse and return the assistant content from the response. If the
        # returned structure differs the exception will propagate upwards so
        # the adapter can retry or try another backend.
        try:
            return resp.choices[0].message.content
        except Exception as exc:
            raise RuntimeError(f"OpenAI response parsing failed: {exc}")
