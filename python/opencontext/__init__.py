"""openContext Python SDK."""

from .context_manager import ContextManager
from .database import Database
from .extractor import extract_from_message, extract_from_messages, generate_summary
from .filter import build_context_prompt, filter_memories, score_memory
from .models import (
    ContextResult,
    Memory,
    MemoryType,
    Message,
    MessageRole,
    Session,
    UserProfile,
)

__version__ = "0.1.0"

__all__ = [
    "ContextManager",
    "Database",
    "Memory",
    "MemoryType",
    "Message",
    "MessageRole",
    "Session",
    "UserProfile",
    "ContextResult",
    "extract_from_message",
    "extract_from_messages",
    "generate_summary",
    "build_context_prompt",
    "filter_memories",
    "score_memory",
]
