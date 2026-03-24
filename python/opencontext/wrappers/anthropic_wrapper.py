"""
Anthropic (Claude) wrapper with automatic context injection.

Usage::

    from opencontext.wrappers.anthropic_wrapper import OpenContextAnthropic

    client = OpenContextAnthropic(
        user_id="alice",
        anthropic_api_key="sk-ant-...",
        db_path="opencontext.db",
    )

    response = client.chat(messages=[
        {"role": "user", "content": "Which languages do you recommend for me?"}
    ])
    print(response.content[0].text)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..context_manager import ContextManager
from ..models import MemoryType


class OpenContextAnthropic:
    """
    Drop-in replacement for the Anthropic Messages API with context injection.

    Requires the ``anthropic`` package::

        pip install anthropic
    """

    def __init__(
        self,
        user_id: str,
        anthropic_api_key: Optional[str] = None,
        db_path: str = "opencontext.db",
        model: str = "claude-3-haiku-20240307",
        max_tokens: int = 1024,
        session_id: Optional[str] = None,
        context_manager: Optional[ContextManager] = None,
    ) -> None:
        self.user_id = user_id
        self.model = model
        self.max_tokens = max_tokens
        self.session_id = session_id
        self.context_manager = context_manager or ContextManager(db_path=db_path)

        api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required: pass anthropic_api_key= or set ANTHROPIC_API_KEY."
            )

        try:
            import anthropic as _anthropic  # type: ignore
            self._client = _anthropic.Anthropic(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required: pip install anthropic"
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
        Send a messages request with automatic context injection.

        Anthropic separates ``system`` from ``messages``.  This wrapper
        automatically extracts any leading system message and passes it in
        the ``system`` parameter.
        """
        sid = session_id or self.session_id

        if inject_context:
            messages = self.context_manager.augment_messages(
                user_id=self.user_id,
                messages=messages,
                session_id=sid,
            )

        # Anthropic API: extract system message separately
        system_content: Optional[str] = None
        anthropic_messages: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = (system_content or "") + msg.get("content", "")
            else:
                anthropic_messages.append(msg)

        create_kwargs: Dict[str, Any] = dict(
            model=model or self.model,
            max_tokens=kwargs.pop("max_tokens", self.max_tokens),
            messages=anthropic_messages,
            **kwargs,
        )
        if system_content:
            create_kwargs["system"] = system_content

        response = self._client.messages.create(**create_kwargs)

        if record and sid:
            assistant_content = ""
            if response.content:
                assistant_content = response.content[0].text
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
