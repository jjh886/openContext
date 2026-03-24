"""Tests for opencontext.database module."""

import os
import tempfile
import pytest

from opencontext.database import Database
from opencontext.models import Memory, MemoryType, Message, MessageRole, Session, UserProfile


@pytest.fixture
def db():
    """Create a temporary in-memory / file database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    database = Database(db_path)
    yield database
    database.close()
    os.unlink(db_path)


# ------------------------------------------------------------------ #
# Memory tests
# ------------------------------------------------------------------ #

def test_save_and_get_memory(db):
    mem = Memory(content="I know Python", memory_type=MemoryType.SKILL, importance=0.8)
    db.save_memory("user1", mem)
    fetched = db.get_memory(mem.id)
    assert fetched is not None
    assert fetched.content == "I know Python"
    assert fetched.memory_type == MemoryType.SKILL
    assert fetched.importance == pytest.approx(0.8)


def test_save_memory_upsert(db):
    mem = Memory(content="Original", memory_type=MemoryType.FACT, importance=0.5)
    db.save_memory("user1", mem)
    mem.content = "Updated"
    mem.importance = 0.9
    db.save_memory("user1", mem)
    fetched = db.get_memory(mem.id)
    assert fetched.content == "Updated"
    assert fetched.importance == pytest.approx(0.9)


def test_get_memories_by_user(db):
    db.save_memory("user1", Memory(content="A", memory_type=MemoryType.FACT))
    db.save_memory("user1", Memory(content="B", memory_type=MemoryType.SKILL))
    db.save_memory("user2", Memory(content="C", memory_type=MemoryType.FACT))
    memories = db.get_memories("user1")
    assert len(memories) == 2
    contents = {m.content for m in memories}
    assert contents == {"A", "B"}


def test_get_memories_by_type(db):
    db.save_memory("user1", Memory(content="Fact1", memory_type=MemoryType.FACT))
    db.save_memory("user1", Memory(content="Skill1", memory_type=MemoryType.SKILL))
    facts = db.get_memories("user1", memory_type=MemoryType.FACT)
    assert len(facts) == 1
    assert facts[0].content == "Fact1"


def test_delete_memory(db):
    mem = Memory(content="To delete", memory_type=MemoryType.HABIT)
    db.save_memory("user1", mem)
    assert db.get_memory(mem.id) is not None
    deleted = db.delete_memory(mem.id)
    assert deleted is True
    assert db.get_memory(mem.id) is None


def test_delete_nonexistent_memory(db):
    assert db.delete_memory("nonexistent-id") is False


def test_update_memory_access(db):
    mem = Memory(content="Access test", memory_type=MemoryType.FACT)
    db.save_memory("user1", mem)
    db.update_memory_access(mem.id)
    fetched = db.get_memory(mem.id)
    assert fetched.access_count == 1


def test_search_memories(db):
    db.save_memory("user1", Memory(content="I love Python programming", memory_type=MemoryType.SKILL))
    db.save_memory("user1", Memory(content="I live in New York", memory_type=MemoryType.FACT))
    results = db.search_memories("user1", "Python")
    assert len(results) == 1
    assert "Python" in results[0].content


def test_memory_tags_serialization(db):
    mem = Memory(
        content="Tagged memory",
        memory_type=MemoryType.SKILL,
        tags=["python", "backend"],
    )
    db.save_memory("user1", mem)
    fetched = db.get_memory(mem.id)
    assert fetched.tags == ["python", "backend"]


# ------------------------------------------------------------------ #
# Session tests
# ------------------------------------------------------------------ #

def test_save_and_get_session(db):
    session = Session(
        id="sess-1",
        user_id="user1",
        messages=[
            Message(role=MessageRole.USER, content="Hello"),
            Message(role=MessageRole.ASSISTANT, content="Hi there"),
        ],
    )
    db.save_session(session)
    fetched = db.get_session("sess-1")
    assert fetched is not None
    assert fetched.user_id == "user1"
    assert len(fetched.messages) == 2
    assert fetched.messages[0].content == "Hello"


def test_get_sessions_by_user(db):
    for i in range(3):
        db.save_session(Session(id=f"sess-{i}", user_id="user1"))
    db.save_session(Session(id="sess-other", user_id="user2"))
    sessions = db.get_sessions("user1")
    assert len(sessions) == 3


def test_delete_session(db):
    session = Session(id="to-delete", user_id="user1")
    db.save_session(session)
    deleted = db.delete_session("to-delete")
    assert deleted is True
    assert db.get_session("to-delete") is None


# ------------------------------------------------------------------ #
# User profile tests
# ------------------------------------------------------------------ #

def test_save_and_get_profile(db):
    profile = UserProfile(
        user_id="user1",
        name="Alice",
        skills=["Python", "SQL"],
        habits=["morning walk"],
    )
    db.save_user_profile(profile)
    fetched = db.get_user_profile("user1")
    assert fetched is not None
    assert fetched.name == "Alice"
    assert "Python" in fetched.skills
    assert "morning walk" in fetched.habits


def test_profile_upsert(db):
    profile = UserProfile(user_id="user1", name="Alice")
    db.save_user_profile(profile)
    profile.name = "Alice Updated"
    db.save_user_profile(profile)
    fetched = db.get_user_profile("user1")
    assert fetched.name == "Alice Updated"


def test_get_nonexistent_profile(db):
    assert db.get_user_profile("ghost-user") is None
