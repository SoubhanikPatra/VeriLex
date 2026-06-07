"""
utils/llm.py
----------------
LLM adapter layer. This module exposes a simple `get_llm_client()` factory
that returns an `LLMClient` instance. The client will attempt configured
backends in order, applying retries and exponential backoff per-backend.

Purpose:
- Provide a single place to make LLM calls so the rest of the codebase
  doesn't directly depend on a particular provider (Groq/OpenAI/etc).
- Provide retries, backoff, timeouts, and graceful fallback to other backends.

The adapter uses `tenacity` for retry/backoff semantics and will run backend
invocations inside a thread with a timeout to avoid blocking the main thread
indefinitely.
"""
from __future__ import annotations

import importlib
import concurrent.futures
import time
from typing import List, Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

from config import settings


class LLMError(Exception):
    """Generic LLM adapter error."""


class LLMClient:
    """Simple LLM client that tries configured backends in order.

    Methods:
    - invoke(prompt: str) -> str
    - invoke_messages(messages: List[Any]) -> str
    """

    def __init__(self):
        # Pre-resolve backend classes based on names in config for quick lookup.
        # The `LLM_BACKENDS` setting is an ordered list of backend identifiers
        # (e.g. ["groq", "openai"]). For each name we attempt to import a
        # module at `utils.llm_backends.<name>_backend` which must expose a
        # `Backend` class implementing `invoke(prompt: str) -> str`.
        #
        # This approach keeps backend implementations isolated and lets the
        # adapter fallback between them when errors occur.
        self.backends = []
        for name in settings.LLM_BACKENDS:
            name = name.strip()
            if not name:
                continue
            # Map backend name to a python module in utils.llm_backends
            module_path = f"utils.llm_backends.{name}_backend"
            try:
                module = importlib.import_module(module_path)
                # Each backend module must expose a `Backend` class
                backend_cls = getattr(module, "Backend")
                # Store tuple (name, class) for later instantiation
                self.backends.append((name, backend_cls))
            except Exception as exc:
                # If a backend is missing or fails to import we don't crash the
                # whole adapter; we log and move on so the system can still run
                # with other available backends.
                print(f"⚠️ Skipping unavailable LLM backend '{name}': {exc}")

    def _invoke_with_retries(self, backend_name: str, backend_instance, prompt: str) -> str:
        """Invoke a single backend with configured retries and backoff.

        This method is wrapped with tenacity retry semantics; any exception
        raised by the backend will trigger retries up to `LLM_RETRIES`.
        """

        # Configure tenacity retry behaviour from settings:
        # - `wait_exponential` implements exponential backoff between attempts
        # - `stop_after_attempt` limits the number of retry attempts
        wait = wait_exponential(multiplier=settings.LLM_BACKOFF["initial"], max=settings.LLM_BACKOFF["max"])
        stop = stop_after_attempt(settings.LLM_RETRIES)

        @retry(stop=stop, wait=wait, retry=retry_if_exception_type(Exception))
        def _call():
            # Run the backend call inside a short-lived thread and enforce a
            # per-call timeout. Running inside a thread gives us a portable way
            # to abort slow network/blocking calls without relying on signal
            # handlers (which don't work on Windows) or backend-specific
            # timeout knobs.
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(backend_instance.invoke, prompt)
                try:
                    # If the backend doesn't respond within `LLM_TIMEOUT` we
                    # cancel the future and raise a timeout error so `tenacity`
                    # can perform a retry if configured.
                    return future.result(timeout=settings.LLM_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    raise LLMError(f"Timeout after {settings.LLM_TIMEOUT}s for backend {backend_name}")

        try:
            start = time.time()
            result = _call()
            elapsed = time.time() - start
            print(f"✅ LLM backend '{backend_name}' succeeded in {elapsed:.2f}s")
            return result
        except RetryError as re:
            # All attempts exhausted for this backend
            # tenacity's RetryError wraps the last attempt exception; unwrap
            last_exc = re.last_attempt.exception()
            raise LLMError(f"Backend '{backend_name}' failed after retries: {last_exc}")

    def invoke(self, prompt: str) -> str:
        """Invoke available backends with fallback.

        Tries each configured backend in order. If a backend fails (after
        its configured retries), the adapter logs and moves to the next.
        If all backends fail, an LLMError is raised.
        """
        if not self.backends:
            # Prevent silent failures if the configuration is empty or wrong
            raise LLMError("No LLM backends available (check settings.LLM_BACKENDS)")

        last_error = None
        # Try each backend in order. The first backend to successfully return a
        # result wins. If a backend fails (raises an exception) we log and move
        # on to the next one so the system can still operate using a fallback.
        for name, backend_cls in self.backends:
            try:
                backend = backend_cls()
                # Each backend exposes an `invoke(prompt: str) -> str` method
                return self._invoke_with_retries(name, backend, prompt)
            except Exception as exc:
                # Log the error and continue to fallback backends
                print(f"⚠️ LLM backend '{name}' error: {exc}")
                last_error = exc
                continue

        # If we reach this point, all backends failed — raise an informative
        # exception so callers can handle the failure appropriately.
        raise LLMError(f"All LLM backends failed. Last error: {last_error}")

    def invoke_messages(self, messages: List[Any]) -> str:
        """Convenience wrapper accepting a list of message objects.

        The adapter extracts textual content from the provided message objects
        and concatenates them into a single prompt string. This keeps the
        rest of the codebase free from backend-specific message types.
        """
        # Extract content from message-like objects (support HumanMessage or dict)
        parts = []
        for m in messages:
            # Support different message shapes: LangChain's `HumanMessage`, a
            # dict with a `content` key, or a plain string. This keeps the
            # adapter flexible for different calling code without forcing them
            # to construct specific message types.
            if hasattr(m, "content"):
                parts.append(m.content)
            elif isinstance(m, dict) and "content" in m:
                parts.append(m["content"])
            else:
                parts.append(str(m))

        # Join messages with blank lines to preserve readable separation in
        # the flattened prompt that gets sent to provider backends.
        joined = "\n\n".join(parts)
        return self.invoke(joined)


_singleton: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Return a singleton LLMClient instance.

    This is a small convenience so callers can `from utils.llm import get_llm_client`
    and re-use the same client across modules without re-importing backend
    modules repeatedly.
    """
    global _singleton
    if _singleton is None:
        _singleton = LLMClient()
    return _singleton
