"""Tests for opencontext.context_manager module."""

import os
import tempfile
import pytest

from opencontext.context_manager import ContextManager
from opencontext.models import MemoryType


@pytest.fixture
def cm():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    manager = ContextManager(db_path=db_path, auto_extract=True, auto_summarize=False)
    yield manager
    manager.db.close()
    os.unlink(db_path)


# ------------------------------------------------------------------ #
# augment_messages
# ------------------------------------------------------------------ #

def test_augment_messages_empty(cm):
    result = cm.augment_messages("user1", [])
    assert result == []


def test_augment_messages_no_memories(cm):
    messages = [{"role": "user", "content": "Hello, what is 2+2?"}]
    result = cm.augment_messages("user1", messages)
    # No memories yet – original list returned unchanged
    assert result == messages


def test_augment_messages_injects_context(cm):
    cm.add_memory("user1", "My name is Alice", MemoryType.FACT, importance=0.9)
    cm.add_memory("user1", "I know Python", MemoryType.SKILL, importance=0.8)

    messages = [{"role": "user", "content": "What programming language should I use?"}]
    result = cm.augment_messages("user1", messages)

    # A system message should have been prepended
    assert result[0]["role"] == "system"
    assert "Alice" in result[0]["content"] or "Python" in result[0]["content"]


def test_augment_merges_existing_system_message(cm):
    cm.add_memory("user1", "I work at Acme", MemoryType.FACT, importance=0.9)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about my company."},
    ]
    result = cm.augment_messages("user1", messages)
    # System message should be first and contain original content plus injected context
    assert result[0]["role"] == "system"
    assert "helpful assistant" in result[0]["content"]
    assert "Acme" in result[0]["content"]


# ------------------------------------------------------------------ #
# record_conversation
# ------------------------------------------------------------------ #

def test_record_conversation_creates_session(cm):
    messages = [
        {"role": "user", "content": "My name is Bob and I prefer Golang."},
        {"role": "assistant", "content": "Nice to meet you, Bob!"},
    ]
    session = cm.record_conversation("user1", "sess-1", messages)
    assert session.id == "sess-1"
    assert len(session.messages) == 2


def test_record_conversation_extracts_memories(cm):
    messages = [
        {"role": "user", "content": "I love Python and I am a backend developer."},
        {"role": "assistant", "content": "Great!"},
    ]
    cm.record_conversation("user1", "sess-1", messages)
    memories = cm.get_memories("user1")
    skill_contents = " ".join(m.content.lower() for m in memories if m.memory_type == MemoryType.SKILL)
    assert "python" in skill_contents or "backend" in skill_contents


def test_record_conversation_updates_profile(cm):
    messages = [
        {"role": "user", "content": "My name is Carol."},
        {"role": "assistant", "content": "Hello, Carol!"},
    ]
    cm.record_conversation("user1", "sess-1", messages)
    profile = cm.get_user_profile("user1")
    assert profile is not None


# ------------------------------------------------------------------ #
# Memory CRUD
# ------------------------------------------------------------------ #

def test_add_and_get_memory(cm):
    mem = cm.add_memory("user1", "I enjoy reading sci-fi novels", MemoryType.HABIT, importance=0.6)
    assert mem.id is not None
    fetched = cm.get_memories("user1", memory_type=MemoryType.HABIT)
    assert any(m.content == "I enjoy reading sci-fi novels" for m in fetched)


def test_delete_memory(cm):
    mem = cm.add_memory("user1", "Temporary memory", MemoryType.FACT)
    assert cm.delete_memory(mem.id) is True
    memories = cm.get_memories("user1")
    assert all(m.id != mem.id for m in memories)


def test_search_memories(cm):
    cm.add_memory("user1", "I use PostgreSQL for databases", MemoryType.SKILL)
    cm.add_memory("user1", "I enjoy coffee every morning", MemoryType.HABIT)
    results = cm.search_memories("user1", "PostgreSQL")
    assert len(results) >= 1
    assert any("PostgreSQL" in m.content for m in results)


# ------------------------------------------------------------------ #
# clear_user_data
# ------------------------------------------------------------------ #

def test_clear_user_data(cm):
    cm.add_memory("user1", "Secret memory", MemoryType.FACT)
    cm.record_conversation(
        "user1", "sess-clear",
        [{"role": "user", "content": "Test"}, {"role": "assistant", "content": "Ok"}],
    )
    cm.clear_user_data("user1")
    assert cm.get_memories("user1") == []
    assert cm.get_sessions("user1") == []


# ------------------------------------------------------------------ #
# get_context
# ------------------------------------------------------------------ #

def test_get_context_returns_relevant_memories(cm):
    cm.add_memory("user1", "I am a Python expert", MemoryType.SKILL, importance=0.9)
    cm.add_memory("user1", "I live in Berlin", MemoryType.FACT, importance=0.7)

    result = cm.get_context("user1", "Python libraries recommendation")
    skill_contents = " ".join(m.content for m in result.memories)
    assert "Python" in skill_contents
    assert result.injected_system_prompt != ""
