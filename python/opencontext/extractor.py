"""Information extractor: derives structured memories from conversations."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import Memory, MemoryType, Message, MessageRole


# ---------------------------------------------------------------------------
# Rule-based extraction patterns
# ---------------------------------------------------------------------------

# Each entry: (regex_pattern, memory_type, importance)
_EXTRACTION_RULES: List[Tuple[str, MemoryType, float]] = [
    # Skills
    (r"(?:I\s+(?:know|can|am able to|have learned|am skilled in|specialize in)\s+(.+?))[.\n]",
     MemoryType.SKILL, 0.7),
    (r"(?:I\s+(?:am|work as|am a)\s+(?:an?\s+)?(\w+(?:\s+\w+){0,3}))[.\n]",
     MemoryType.SKILL, 0.7),
    (r"(?:I\s+(?:have|got)\s+(?:a\s+)?(?:degree|certificate|certification)\s+in\s+(.+?))[.\n]",
     MemoryType.SKILL, 0.8),
    # Preferences
    (r"(?:I\s+(?:prefer|like|love|enjoy|hate|dislike|don't like)\s+(.+?))[.\n,]",
     MemoryType.PREFERENCE, 0.6),
    (r"(?:my\s+(?:favorite|preferred|go-to)\s+\w+\s+is\s+(.+?))[.\n,]",
     MemoryType.PREFERENCE, 0.65),
    # Personal facts
    (r"(?:my\s+name\s+is\s+(\w+(?:\s+\w+)?))[.\n,]",
     MemoryType.FACT, 0.9),
    (r"(?:I\s+(?:am|'m)\s+(\d+)\s+years?\s+old)[.\n,]",
     MemoryType.FACT, 0.8),
    (r"(?:I\s+(?:live|am based|am located)\s+in\s+(.+?))[.\n,]",
     MemoryType.FACT, 0.8),
    (r"(?:I\s+(?:work|am employed)\s+at\s+(.+?))[.\n,]",
     MemoryType.FACT, 0.75),
    # Habits
    (r"(?:I\s+(?:usually|always|often|regularly|typically|never)\s+(.+?))[.\n,]",
     MemoryType.HABIT, 0.6),
    (r"(?:every\s+(?:day|morning|evening|night|week)\s+I\s+(.+?))[.\n,]",
     MemoryType.HABIT, 0.65),
    # Tasks / goals
    (r"(?:I\s+(?:need|want|plan)\s+to\s+(.+?))[.\n,]",
     MemoryType.TASK, 0.5),
    (r"(?:remind\s+me\s+to\s+(.+?))[.\n,]",
     MemoryType.TASK, 0.7),
]

# Programming languages / tech keywords for skill detection
_TECH_KEYWORDS = {
    "python", "javascript", "typescript", "java", "c++", "c#", "golang", "go",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "matlab", "sql",
    "html", "css", "react", "vue", "angular", "django", "flask", "fastapi",
    "node.js", "nodejs", "docker", "kubernetes", "aws", "azure", "gcp",
    "machine learning", "deep learning", "nlp", "data science", "devops",
    "linux", "git", "postgresql", "mysql", "mongodb", "redis",
}


def _clean(text: str) -> str:
    """Normalise whitespace and strip trailing punctuation."""
    return re.sub(r"\s+", " ", text).strip(" .,;:!?")


def extract_from_message(
    message: Message,
    session_id: Optional[str] = None,
) -> List[Memory]:
    """Extract structured memories from a single message using rules."""
    if message.role != MessageRole.USER:
        return []

    text = message.content
    found: List[Memory] = []
    seen_contents: set = set()

    for pattern, mem_type, importance in _EXTRACTION_RULES:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            content = _clean(match.group(0))
            if not content or content.lower() in seen_contents:
                continue
            seen_contents.add(content.lower())
            found.append(
                Memory(
                    content=content,
                    memory_type=mem_type,
                    importance=importance,
                    tags=_extract_tags(content),
                    source_session_id=session_id,
                )
            )

    # Detect technology keywords and save as skills
    lower_text = text.lower()
    for tech in _TECH_KEYWORDS:
        if re.search(r"\b" + re.escape(tech) + r"\b", lower_text):
            skill_content = f"Uses / knows {tech}"
            if skill_content.lower() not in seen_contents:
                seen_contents.add(skill_content.lower())
                found.append(
                    Memory(
                        content=skill_content,
                        memory_type=MemoryType.SKILL,
                        importance=0.55,
                        tags=[tech, "technology"],
                        source_session_id=session_id,
                    )
                )

    return found


def extract_from_messages(
    messages: List[Message],
    session_id: Optional[str] = None,
) -> List[Memory]:
    """Extract memories from a list of messages."""
    memories: List[Memory] = []
    for msg in messages:
        memories.extend(extract_from_message(msg, session_id=session_id))
    return memories


def generate_summary(
    messages: List[Message],
    session_id: Optional[str] = None,
    max_length: int = 500,
) -> Optional[Memory]:
    """Generate a brief conversation summary memory."""
    user_messages = [m for m in messages if m.role == MessageRole.USER]
    if not user_messages:
        return None

    topics: List[str] = []
    for msg in user_messages:
        # Grab first sentence of each user turn as a topic hint
        sentences = re.split(r"[.!?]", msg.content)
        if sentences:
            first = _clean(sentences[0])
            if first and len(first) > 10:
                topics.append(first[:120])

    if not topics:
        return None

    summary = "Conversation topics: " + "; ".join(topics[:5])
    if len(summary) > max_length:
        summary = summary[:max_length] + "..."

    return Memory(
        content=summary,
        memory_type=MemoryType.SUMMARY,
        importance=0.4,
        tags=["summary"],
        source_session_id=session_id,
    )


def _extract_tags(text: str) -> List[str]:
    """Derive simple tags from extracted text."""
    tags: List[str] = []
    lower = text.lower()
    for tech in _TECH_KEYWORDS:
        if tech in lower:
            tags.append(tech)
    return tags[:5]
