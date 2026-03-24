"""Relevance filtering: score and select memories for a given query."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from .models import Memory, MemoryType


# ---------------------------------------------------------------------------
# TF-IDF style relevance scoring (no heavy ML dependencies)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought", "used",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "up", "about",
    "into", "through", "during", "including", "until", "against", "among",
    "throughout", "despite", "towards", "upon", "concerning", "and", "but",
    "or", "nor", "so", "yet", "both", "either", "neither", "not", "only",
    "own", "same", "than", "too", "very", "just", "i", "me", "my", "myself",
    "we", "our", "you", "your", "he", "she", "it", "they", "what", "which",
    "who", "that", "this", "these", "those", "how",
}

# Boost multipliers per memory type
_TYPE_BOOST: Dict[MemoryType, float] = {
    MemoryType.FACT: 1.2,
    MemoryType.SKILL: 1.3,
    MemoryType.PREFERENCE: 1.1,
    MemoryType.HABIT: 1.0,
    MemoryType.TASK: 1.4,
    MemoryType.CONTACT: 1.1,
    MemoryType.SUMMARY: 0.8,
}


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _tfidf_score(query_tokens: List[str], doc_tokens: List[str]) -> float:
    """Compute a simple TF-IDF cosine similarity."""
    if not query_tokens or not doc_tokens:
        return 0.0

    doc_freq = Counter(doc_tokens)
    query_freq = Counter(query_tokens)
    all_terms = set(query_freq) | set(doc_freq)

    # IDF: log((N+1)/(df+1)) — treat as single document (df=1)
    def tf(freq_map: Counter, term: str) -> float:
        return freq_map.get(term, 0) / max(len(freq_map), 1)

    def idf(term: str) -> float:
        return math.log(2.0 / (1.0 + (1 if term in doc_freq else 0)))

    dot, q_norm, d_norm = 0.0, 0.0, 0.0
    for term in all_terms:
        q_tfidf = tf(query_freq, term) * idf(term)
        d_tfidf = tf(doc_freq, term) * idf(term)
        dot += q_tfidf * d_tfidf
        q_norm += q_tfidf ** 2
        d_norm += d_tfidf ** 2

    denom = math.sqrt(q_norm) * math.sqrt(d_norm)
    if denom == 0.0:
        return 0.0
    return dot / denom


def score_memory(query: str, memory: Memory) -> float:
    """Compute relevance score of a memory for a given query string."""
    query_tokens = _tokenize(query)
    doc_tokens = _tokenize(memory.content + " " + " ".join(memory.tags))

    text_score = _tfidf_score(query_tokens, doc_tokens)
    importance_boost = 0.3 * memory.importance
    type_boost = _TYPE_BOOST.get(memory.memory_type, 1.0) - 1.0  # delta
    final_score = (text_score + importance_boost) * (1.0 + type_boost)
    return min(final_score, 1.0)


def filter_memories(
    query: str,
    memories: List[Memory],
    max_memories: int = 10,
    min_score: float = 0.05,
    required_types: Optional[List[MemoryType]] = None,
) -> List[Tuple[Memory, float]]:
    """
    Return the most relevant memories for *query*, ranked by relevance score.

    Args:
        query: The current user query / topic.
        memories: All available memories to filter from.
        max_memories: Maximum number of memories to return.
        min_score: Minimum relevance threshold (0–1).
        required_types: If provided, always include memories of these types
                        regardless of score (up to their quota).

    Returns:
        List of (memory, score) tuples sorted by descending score.
    """
    scored: List[Tuple[Memory, float]] = []

    for memory in memories:
        score = score_memory(query, memory)
        if score >= min_score:
            scored.append((memory, score))

    # Always include required types (e.g., user facts)
    if required_types:
        required_ids = {m.id for m, _ in scored}
        for memory in memories:
            if memory.memory_type in required_types and memory.id not in required_ids:
                score = score_memory(query, memory)
                scored.append((memory, max(score, min_score)))
                required_ids.add(memory.id)

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max_memories]


def build_context_prompt(
    memories: List[Tuple[Memory, float]],
    user_query: str = "",
    include_scores: bool = False,
) -> str:
    """
    Build a system-prompt section that injects relevant memories as context.
    """
    if not memories:
        return ""

    lines: List[str] = [
        "## Relevant context about the user",
        "(This information was retrieved from the user's memory store.)",
        "",
    ]

    type_groups: Dict[str, List[str]] = {}
    for memory, score in memories:
        label = memory.memory_type.value.capitalize()
        entry = memory.content
        if include_scores:
            entry += f" [score={score:.2f}]"
        type_groups.setdefault(label, []).append(entry)

    for label in ["Fact", "Skill", "Preference", "Habit", "Task", "Contact", "Summary"]:
        items = type_groups.get(label)
        if items:
            lines.append(f"**{label}s:**")
            for item in items:
                lines.append(f"- {item}")
            lines.append("")

    return "\n".join(lines).rstrip()
