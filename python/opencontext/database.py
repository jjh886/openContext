"""SQLite database layer for openContext."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

from .models import Memory, MemoryType, Message, MessageRole, Session, UserProfile


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    content TEXT NOT NULL,
    memory_type TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.5,
    tags TEXT NOT NULL DEFAULT '[]',
    source_session_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_accessed TEXT NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_memory_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance DESC);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    messages TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);

CREATE TABLE IF NOT EXISTS user_profiles (
    user_id TEXT PRIMARY KEY,
    name TEXT,
    skills TEXT NOT NULL DEFAULT '[]',
    preferences TEXT NOT NULL DEFAULT '{}',
    habits TEXT NOT NULL DEFAULT '[]',
    facts TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL
);
"""


class Database:
    """Thread-safe SQLite database wrapper for openContext."""

    def __init__(self, db_path: str = "opencontext.db") -> None:
        self.db_path = db_path
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return self._local.conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript(_SCHEMA_SQL)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ------------------------------------------------------------------ #
    # Memory operations
    # ------------------------------------------------------------------ #

    def save_memory(self, user_id: str, memory: Memory) -> Memory:
        """Insert or update a memory record."""
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO memories
                    (id, user_id, content, memory_type, importance, tags,
                     source_session_id, created_at, updated_at, last_accessed,
                     access_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    importance = excluded.importance,
                    tags = excluded.tags,
                    updated_at = excluded.updated_at,
                    last_accessed = excluded.last_accessed,
                    access_count = excluded.access_count,
                    metadata = excluded.metadata
                """,
                (
                    memory.id,
                    user_id,
                    memory.content,
                    memory.memory_type.value,
                    memory.importance,
                    json.dumps(memory.tags),
                    memory.source_session_id,
                    memory.created_at.isoformat(),
                    memory.updated_at.isoformat(),
                    memory.last_accessed.isoformat(),
                    memory.access_count,
                    json.dumps(memory.metadata),
                ),
            )
        return memory

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        row = self._get_conn().execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def get_memories(
        self,
        user_id: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 100,
        min_importance: float = 0.0,
    ) -> List[Memory]:
        sql = "SELECT * FROM memories WHERE user_id = ? AND importance >= ?"
        params: List[Any] = [user_id, min_importance]
        if memory_type is not None:
            sql += " AND memory_type = ?"
            params.append(memory_type.value)
        sql += " ORDER BY importance DESC, last_accessed DESC LIMIT ?"
        params.append(limit)
        rows = self._get_conn().execute(sql, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def delete_memory(self, memory_id: str) -> bool:
        with self._transaction() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    def update_memory_access(self, memory_id: str) -> None:
        now = datetime.utcnow().isoformat()
        with self._transaction() as conn:
            conn.execute(
                """
                UPDATE memories
                SET last_accessed = ?, access_count = access_count + 1
                WHERE id = ?
                """,
                (now, memory_id),
            )

    def search_memories(self, user_id: str, query: str, limit: int = 20) -> List[Memory]:
        """Simple full-text search on memory content."""
        like_query = f"%{query}%"
        rows = self._get_conn().execute(
            """
            SELECT * FROM memories
            WHERE user_id = ? AND content LIKE ?
            ORDER BY importance DESC, last_accessed DESC
            LIMIT ?
            """,
            (user_id, like_query, limit),
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            content=row["content"],
            memory_type=MemoryType(row["memory_type"]),
            importance=row["importance"],
            tags=json.loads(row["tags"]),
            source_session_id=row["source_session_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            access_count=row["access_count"],
            metadata=json.loads(row["metadata"]),
        )

    # ------------------------------------------------------------------ #
    # Session operations
    # ------------------------------------------------------------------ #

    def save_session(self, session: Session) -> Session:
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, user_id, messages, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    messages = excluded.messages,
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata
                """,
                (
                    session.id,
                    session.user_id,
                    json.dumps([m.to_dict() for m in session.messages]),
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                    json.dumps(session.metadata),
                ),
            )
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        row = self._get_conn().execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return self._row_to_session(row) if row else None

    def get_sessions(self, user_id: str, limit: int = 50) -> List[Session]:
        rows = self._get_conn().execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [self._row_to_session(r) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        with self._transaction() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_session(row: sqlite3.Row) -> Session:
        messages_data = json.loads(row["messages"])
        messages = [
            Message(
                role=MessageRole(m["role"]),
                content=m["content"],
                timestamp=datetime.fromisoformat(
                    m.get("timestamp", datetime.utcnow().isoformat())
                ),
                metadata=m.get("metadata", {}),
            )
            for m in messages_data
        ]
        return Session(
            id=row["id"],
            user_id=row["user_id"],
            messages=messages,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            metadata=json.loads(row["metadata"]),
        )

    # ------------------------------------------------------------------ #
    # User profile operations
    # ------------------------------------------------------------------ #

    def save_user_profile(self, profile: UserProfile) -> UserProfile:
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO user_profiles
                    (user_id, name, skills, preferences, habits, facts, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    name = excluded.name,
                    skills = excluded.skills,
                    preferences = excluded.preferences,
                    habits = excluded.habits,
                    facts = excluded.facts,
                    updated_at = excluded.updated_at
                """,
                (
                    profile.user_id,
                    profile.name,
                    json.dumps(profile.skills),
                    json.dumps(profile.preferences),
                    json.dumps(profile.habits),
                    json.dumps(profile.facts),
                    profile.updated_at.isoformat(),
                ),
            )
        return profile

    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        row = self._get_conn().execute(
            "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
        if not row:
            return None
        return UserProfile(
            user_id=row["user_id"],
            name=row["name"],
            skills=json.loads(row["skills"]),
            preferences=json.loads(row["preferences"]),
            habits=json.loads(row["habits"]),
            facts=json.loads(row["facts"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
