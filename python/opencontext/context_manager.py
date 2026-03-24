"""Core context manager: the main entry point for openContext."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from .database import Database
from .extractor import extract_from_messages, generate_summary
from .filter import build_context_prompt, filter_memories
from .models import (
    ContextResult,
    Memory,
    MemoryType,
    Message,
    MessageRole,
    Session,
    UserProfile,
)


class ContextManager:
    """
    Manages AI context across multiple sessions for a given user.

    Usage::

        cm = ContextManager(db_path="myapp.db")

        # Augment messages before sending to LLM
        augmented = cm.augment_messages(user_id="alice", messages=[
            {"role": "user", "content": "I prefer Python and I'm a backend dev."}
        ])

        # After getting LLM response, record the conversation
        cm.record_conversation(
            user_id="alice",
            session_id="session-1",
            messages=augmented + [{"role": "assistant", "content": "Got it!"}],
        )
    """

    def __init__(
        self,
        db_path: str = "opencontext.db",
        max_context_memories: int = 10,
        min_relevance_score: float = 0.05,
        auto_extract: bool = True,
        auto_summarize: bool = True,
    ) -> None:
        self.db = Database(db_path)
        self.max_context_memories = max_context_memories
        self.min_relevance_score = min_relevance_score
        self.auto_extract = auto_extract
        self.auto_summarize = auto_summarize

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def augment_messages(
        self,
        user_id: str,
        messages: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        include_user_profile: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Inject relevant context into *messages* before sending to an LLM.

        The last user message is used as the relevance query.  A ``system``
        message carrying the injected context is prepended (or merged into an
        existing system message if one is already present).

        Returns a new list – the original list is not mutated.
        """
        if not messages:
            return messages

        # Find the most recent user turn as the relevance query
        query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                query = msg.get("content", "")
                break

        context_result = self.get_context(
            user_id=user_id,
            query=query,
            session_id=session_id,
            include_user_profile=include_user_profile,
        )

        if not context_result.injected_system_prompt:
            return list(messages)

        augmented = list(messages)
        if augmented and augmented[0].get("role") == "system":
            augmented[0] = {
                **augmented[0],
                "content": augmented[0]["content"] + "\n\n" + context_result.injected_system_prompt,
            }
        else:
            augmented.insert(
                0,
                {"role": "system", "content": context_result.injected_system_prompt},
            )
        return augmented

    def get_context(
        self,
        user_id: str,
        query: str,
        session_id: Optional[str] = None,
        include_user_profile: bool = True,
    ) -> ContextResult:
        """
        Retrieve relevant memories and build a context prompt for *query*.
        """
        all_memories = self.db.get_memories(user_id, limit=200)
        scored = filter_memories(
            query=query,
            memories=all_memories,
            max_memories=self.max_context_memories,
            min_score=self.min_relevance_score,
            required_types=[MemoryType.FACT],
        )

        selected_memories = [m for m, _ in scored]
        scores = {m.id: s for m, s in scored}

        # Mark accessed
        for memory in selected_memories:
            self.db.update_memory_access(memory.id)

        profile = self.db.get_user_profile(user_id) if include_user_profile else None
        context_prompt = build_context_prompt(scored)

        return ContextResult(
            memories=selected_memories,
            user_profile=profile,
            injected_system_prompt=context_prompt,
            relevance_scores=scores,
        )

    def record_conversation(
        self,
        user_id: str,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> Session:
        """
        Persist a conversation turn and extract new memories from it.

        Call this *after* you receive the LLM response to keep the memory
        store up to date.
        """
        parsed_messages = [
            Message(
                role=MessageRole(m["role"]),
                content=m.get("content", ""),
                metadata={k: v for k, v in m.items() if k not in ("role", "content")},
            )
            for m in messages
            if m.get("role") in ("user", "assistant", "system")
        ]

        session = self.db.get_session(session_id)
        if session is None:
            session = Session(id=session_id, user_id=user_id)
        session.messages = parsed_messages
        session.updated_at = datetime.utcnow()
        self.db.save_session(session)

        if self.auto_extract:
            new_memories = extract_from_messages(parsed_messages, session_id=session_id)
            for mem in new_memories:
                self.db.save_memory(user_id, mem)
            self._update_user_profile(user_id)

        if self.auto_summarize and len(parsed_messages) >= 4:
            summary = generate_summary(parsed_messages, session_id=session_id)
            if summary:
                self.db.save_memory(user_id, summary)

        return session

    # ------------------------------------------------------------------ #
    # Memory management
    # ------------------------------------------------------------------ #

    def add_memory(
        self,
        user_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.FACT,
        importance: float = 0.7,
        tags: Optional[List[str]] = None,
    ) -> Memory:
        """Manually add a memory for a user."""
        memory = Memory(
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags or [],
        )
        self.db.save_memory(user_id, memory)
        return memory

    def get_memories(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100,
    ) -> List[Memory]:
        """Retrieve memories for a user."""
        return self.db.get_memories(user_id, memory_type=memory_type, limit=limit)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        return self.db.delete_memory(memory_id)

    def search_memories(self, user_id: str, query: str, limit: int = 20) -> List[Memory]:
        """Full-text search over a user's memories."""
        return self.db.search_memories(user_id, query, limit=limit)

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Return the aggregated user profile."""
        return self.db.get_user_profile(user_id)

    def get_sessions(self, user_id: str, limit: int = 50) -> List[Session]:
        """Return recent sessions for a user."""
        return self.db.get_sessions(user_id, limit=limit)

    def clear_user_data(self, user_id: str) -> None:
        """Delete all data for a user (GDPR / privacy support)."""
        memories = self.db.get_memories(user_id, limit=10_000)
        for mem in memories:
            self.db.delete_memory(mem.id)
        sessions = self.db.get_sessions(user_id, limit=10_000)
        for session in sessions:
            self.db.delete_session(session.id)
        profile = self.db.get_user_profile(user_id)
        if profile:
            empty = UserProfile(user_id=user_id)
            self.db.save_user_profile(empty)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _update_user_profile(self, user_id: str) -> None:
        """Rebuild the cached UserProfile from stored memories."""
        profile = self.db.get_user_profile(user_id) or UserProfile(user_id=user_id)

        skills = self.db.get_memories(user_id, memory_type=MemoryType.SKILL, limit=50)
        habits = self.db.get_memories(user_id, memory_type=MemoryType.HABIT, limit=30)
        facts = self.db.get_memories(user_id, memory_type=MemoryType.FACT, limit=30)
        preferences = self.db.get_memories(
            user_id, memory_type=MemoryType.PREFERENCE, limit=30
        )

        profile.skills = list({m.content for m in skills})
        profile.habits = list({m.content for m in habits})
        profile.facts = list({m.content for m in facts})
        profile.preferences = {
            str(i): m.content for i, m in enumerate(preferences)
        }

        # Try to extract user name from facts
        for fact in facts:
            import re
            match = re.search(r"my name is (\w[\w ]+)", fact.content, re.IGNORECASE)
            if match:
                profile.name = match.group(1).strip()
                break

        profile.updated_at = datetime.utcnow()
        self.db.save_user_profile(profile)
