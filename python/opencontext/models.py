"""Data models for openContext."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class MemoryType(str, Enum):
    """Types of memory that can be stored."""
    FACT = "fact"
    SKILL = "skill"
    PREFERENCE = "preference"
    CONTACT = "contact"
    TASK = "task"
    SUMMARY = "summary"
    HABIT = "habit"


class MessageRole(str, Enum):
    """Role of a message in a conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass
class Memory:
    """A single memory item stored in the context database."""
    content: str
    memory_type: MemoryType
    importance: float = 0.5  # 0.0 to 1.0
    tags: List[str] = field(default_factory=list)
    source_session_id: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "importance": self.importance,
            "tags": self.tags,
            "source_session_id": self.source_session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Memory":
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            importance=data.get("importance", 0.5),
            tags=data.get("tags", []),
            source_session_id=data.get("source_session_id"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data.get("access_count", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Message:
    """A single message in a conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Session:
    """A conversation session."""
    user_id: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        return cls(
            id=data["id"],
            user_id=data["user_id"],
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class UserProfile:
    """Aggregated user profile built from memories."""
    user_id: str
    name: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    habits: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "skills": self.skills,
            "preferences": self.preferences,
            "habits": self.habits,
            "facts": self.facts,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        return cls(
            user_id=data["user_id"],
            name=data.get("name"),
            skills=data.get("skills", []),
            preferences=data.get("preferences", {}),
            habits=data.get("habits", []),
            facts=data.get("facts", []),
            updated_at=datetime.fromisoformat(
                data.get("updated_at", datetime.utcnow().isoformat())
            ),
        )


@dataclass
class ContextResult:
    """Result of a context retrieval operation."""
    memories: List[Memory]
    user_profile: Optional[UserProfile]
    injected_system_prompt: str
    relevance_scores: Dict[str, float] = field(default_factory=dict)
