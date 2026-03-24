"""
OpenAI wrapper with automatic context injection.

Usage::

    from opencontext.wrappers.openai_wrapper import OpenContextOpenAI

    client = OpenContextOpenAI(
        user_id="alice",
        openai_api_key="sk-...",   # or set OPENAI_API_KEY env var
        db_path="opencontext.db",
    )

    response = client.chat(messages=[
        {"role": "user", "content": "What Python libraries do you recommend for me?"}
    ])
    print(response["choices"][0]["message"]["content"])
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..context_manager import ContextManager
from ..models import MemoryType


class OpenContextOpenAI:
    """
    Drop-in replacement for the OpenAI Chat Completions API that automatically
    injects relevant context from the memory store.

    Requires the ``openai`` package::

        pip install openai

    The wrapper intercepts every ``chat()`` call to:
    1. Query relevant memories for the current user query.
    2. Inject them as a system message.
    3. Call the real OpenAI API.
    4. Extract new information from the response and persist it.
    """

    def __init__(
        self,
        user_id: str,
        openai_api_key: Optional[str] = None,
        db_path: str = "opencontext.db",
        model: str = "gpt-3.5-turbo",
        session_id: Optional[str] = None,
        context_manager: Optional[ContextManager] = None,
        **openai_kwargs: Any,
    ) -> None:
        self.user_id = user_id
        self.model = model
        self.session_id = session_id
        self.context_manager = context_manager or ContextManager(db_path=db_path)
        self._openai_kwargs = openai_kwargs

        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required: pass openai_api_key= or set OPENAI_API_KEY."
            )

        try:
            import openai as _openai  # type: ignore
            self._client = _openai.OpenAI(api_key=api_key, **openai_kwargs)
        except ImportError as exc:
            raise ImportError(
                "openai package is required: pip install openai"
            ) from exc

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        inject_context: bool = True,
        record: bool = True,
        **kwargs: Any,
    ) -> Any:
        """
        Send a chat request with automatic context injection and memory recording.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            model: Override the default model.
            session_id: Override the session id.
            inject_context: If False, skips context injection (raw pass-through).
            record: If False, skips saving the conversation turn to memory.
            **kwargs: Forwarded to ``openai.chat.completions.create``.

        Returns:
            The raw OpenAI ``ChatCompletion`` object.
        """
        sid = session_id or self.session_id

        if inject_context:
            messages = self.context_manager.augment_messages(
                user_id=self.user_id,
                messages=messages,
                session_id=sid,
            )

        response = self._client.chat.completions.create(
            model=model or self.model,
            messages=messages,
            **kwargs,
        )

        if record and sid:
            assistant_content = ""
            if response.choices:
                assistant_content = response.choices[0].message.content or ""
            full_messages = list(messages) + [
                {"role": "assistant", "content": assistant_content}
            ]
            self.context_manager.record_conversation(
                user_id=self.user_id,
                session_id=sid,
                messages=full_messages,
            )

        return response

    def add_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        importance: float = 0.8,
    ) -> None:
        """Manually add a memory for the current user."""
        self.context_manager.add_memory(
            self.user_id, content, memory_type=memory_type, importance=importance
        )

    def get_memories(self, memory_type: Optional[MemoryType] = None) -> List[Any]:
        """Retrieve memories for the current user."""
        return self.context_manager.get_memories(
            self.user_id, memory_type=memory_type
        )
