"""
Groq backend for the LLM adapter.

This module provides a thin wrapper around `langchain_groq.ChatGroq` so the
adapter can call a consistent `Backend.invoke(prompt: str) -> str` method.
"""
from __future__ import annotations

from typing import Any

from langchain_groq import ChatGroq

from config import settings


class Backend:
    """Backend wrapper around Groq's Chat client.

    The wrapper keeps a minimal interface (`invoke`) used by the adapter.
    """

    def __init__(self):
        # Use model and temperature from global settings
        self.model = settings.CHAT_MODEL
        self.temperature = settings.LLM_TEMPERATURE

    def invoke(self, prompt: str) -> str:
        """Invoke Groq synchronously and return the textual response.

        Note: the adapter will run this method inside a thread and apply
        timeouts/retries externally. Backend implementations should raise
        exceptions on failure to allow the adapter to retry or fallback.
        """
        # Instantiate the Groq client with configured model/temperature.
        # Keep this call lightweight; the adapter enforces retries/timeouts
        # around this synchronous method.
        client = ChatGroq(model=self.model, temperature=self.temperature)

        # `ChatGroq.invoke` expects a list of message objects (HumanMessage
        # compatible). We pass a single message as a dict with a `content` key
        # which the LangChain wrapper accepts via duck typing. The adapter
        # flattens message lists into a single prompt before calling this.
        response = client.invoke([{"content": prompt}])

        # Return the textual content of the model response. On errors the
        # underlying client will raise which lets the adapter perform retries
        # or fallbacks.
        return response.content
