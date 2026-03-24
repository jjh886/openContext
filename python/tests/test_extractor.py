"""Tests for opencontext.extractor module."""

import pytest

from opencontext.extractor import extract_from_message, extract_from_messages, generate_summary
from opencontext.models import Memory, MemoryType, Message, MessageRole


def user_msg(content: str) -> Message:
    return Message(role=MessageRole.USER, content=content)


def assistant_msg(content: str) -> Message:
    return Message(role=MessageRole.ASSISTANT, content=content)


# ------------------------------------------------------------------ #
# Basic extraction
# ------------------------------------------------------------------ #

def test_extract_skill_from_message():
    msg = user_msg("I know Python and JavaScript very well.")
    memories = extract_from_message(msg)
    types = {m.memory_type for m in memories}
    assert MemoryType.SKILL in types


def test_extract_preference_from_message():
    msg = user_msg("I prefer dark mode in all my editors.")
    memories = extract_from_message(msg)
    types = {m.memory_type for m in memories}
    assert MemoryType.PREFERENCE in types


def test_extract_fact_name():
    msg = user_msg("My name is Alice and I live in Paris.")
    memories = extract_from_message(msg)
    contents = " ".join(m.content for m in memories)
    assert any("Alice" in m.content or "Paris" in m.content for m in memories)


def test_no_extraction_from_assistant():
    """Assistant messages should not generate memories."""
    msg = assistant_msg("I know Python too!")
    memories = extract_from_message(msg)
    assert memories == []


def test_tech_keyword_detection():
    msg = user_msg("I am building an application using React and TypeScript.")
    memories = extract_from_message(msg)
    skill_contents = " ".join(m.content.lower() for m in memories if m.memory_type == MemoryType.SKILL)
    assert "react" in skill_contents or "typescript" in skill_contents


def test_extract_from_multiple_messages():
    messages = [
        user_msg("I prefer Python."),
        assistant_msg("Great choice!"),
        user_msg("I usually exercise every morning."),
    ]
    memories = extract_from_messages(messages)
    types = {m.memory_type for m in memories}
    assert MemoryType.PREFERENCE in types or MemoryType.SKILL in types


def test_extract_with_session_id():
    msg = user_msg("I work at Acme Corp.")
    memories = extract_from_message(msg, session_id="sess-abc")
    for mem in memories:
        assert mem.source_session_id == "sess-abc"


# ------------------------------------------------------------------ #
# Summary generation
# ------------------------------------------------------------------ #

def test_generate_summary():
    messages = [
        user_msg("I need help with Python decorators."),
        assistant_msg("Sure, decorators are..."),
        user_msg("Can you show me an example?"),
        assistant_msg("Here is an example..."),
    ]
    summary = generate_summary(messages, session_id="s1")
    assert summary is not None
    assert summary.memory_type == MemoryType.SUMMARY
    assert "Python" in summary.content or "decorators" in summary.content.lower()


def test_generate_summary_empty():
    summary = generate_summary([])
    assert summary is None


def test_generate_summary_only_assistant():
    messages = [assistant_msg("Hello!"), assistant_msg("How can I help?")]
    summary = generate_summary(messages)
    assert summary is None
