"""Tests for opencontext.filter module."""

import pytest

from opencontext.filter import (
    build_context_prompt,
    filter_memories,
    score_memory,
)
from opencontext.models import Memory, MemoryType


def make_memory(content: str, mem_type: MemoryType = MemoryType.FACT, importance: float = 0.5) -> Memory:
    return Memory(content=content, memory_type=mem_type, importance=importance)


# ------------------------------------------------------------------ #
# Score tests
# ------------------------------------------------------------------ #

def test_score_relevant_memory():
    mem = make_memory("I know Python programming", MemoryType.SKILL)
    score = score_memory("Python developer needed", mem)
    assert score > 0.0


def test_score_irrelevant_memory():
    mem = make_memory("I enjoy gardening and hiking", MemoryType.HABIT)
    score = score_memory("Python programming question", mem)
    # Relevance should be very low (no shared keywords)
    assert score < 0.4


def test_score_importance_boost():
    low_imp = make_memory("I know Go", MemoryType.SKILL, importance=0.1)
    high_imp = make_memory("I know Go", MemoryType.SKILL, importance=0.9)
    low_score = score_memory("Go programming", low_imp)
    high_score = score_memory("Go programming", high_imp)
    assert high_score > low_score


def test_score_empty_query():
    mem = make_memory("Some memory", MemoryType.FACT)
    score = score_memory("", mem)
    # With empty query, score should only reflect importance
    assert 0.0 <= score <= 1.0


# ------------------------------------------------------------------ #
# Filter tests
# ------------------------------------------------------------------ #

def test_filter_returns_relevant():
    memories = [
        make_memory("I love Python programming", MemoryType.SKILL),
        make_memory("I live in Tokyo", MemoryType.FACT),
        make_memory("I enjoy swimming", MemoryType.HABIT),
    ]
    results = filter_memories("Python question", memories, max_memories=5)
    contents = [m.content for m, _ in results]
    assert any("Python" in c for c in contents)


def test_filter_respects_max():
    memories = [make_memory(f"Fact {i}", MemoryType.FACT, importance=0.8) for i in range(20)]
    results = filter_memories("fact question", memories, max_memories=5)
    assert len(results) <= 5


def test_filter_required_types():
    memories = [
        make_memory("My name is Bob", MemoryType.FACT, importance=0.9),
        make_memory("I swim daily", MemoryType.HABIT, importance=0.2),
    ]
    # Even with an irrelevant query, FACT should be included because it's required
    results = filter_memories(
        "unrelated topic xyz",
        memories,
        max_memories=10,
        min_score=0.0,
        required_types=[MemoryType.FACT],
    )
    mem_types = {m.memory_type for m, _ in results}
    assert MemoryType.FACT in mem_types


def test_filter_sorted_by_score():
    memories = [
        make_memory("Python backend development", MemoryType.SKILL, importance=0.6),
        make_memory("I enjoy running", MemoryType.HABIT, importance=0.5),
    ]
    results = filter_memories("Python backend", memories, max_memories=5, min_score=0.0)
    if len(results) >= 2:
        assert results[0][1] >= results[1][1]


# ------------------------------------------------------------------ #
# Prompt building
# ------------------------------------------------------------------ #

def test_build_context_prompt_empty():
    prompt = build_context_prompt([])
    assert prompt == ""


def test_build_context_prompt_with_memories():
    memories = [
        (make_memory("I am a Python developer", MemoryType.SKILL), 0.9),
        (make_memory("I prefer dark mode", MemoryType.PREFERENCE), 0.7),
    ]
    prompt = build_context_prompt(memories)
    assert "Python developer" in prompt
    assert "dark mode" in prompt
    assert "Skill" in prompt or "skill" in prompt.lower()
    assert "Preference" in prompt or "preference" in prompt.lower()
