"""
openContext REST API server.

Start the server::

    python -m opencontext.server.app
    # or
    uvicorn opencontext.server.app:app --host 0.0.0.0 --port 8765

API prefix: /api/v1
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel as _BaseModel
except ImportError as exc:
    raise ImportError(
        "fastapi and pydantic are required for the server: "
        "pip install fastapi uvicorn"
    ) from exc

from ..context_manager import ContextManager
from ..models import Memory, MemoryType

# ---------------------------------------------------------------------------
# Singleton context manager (one DB, many users)
# ---------------------------------------------------------------------------

_DB_PATH = os.environ.get("OPENCONTEXT_DB_PATH", "opencontext.db")
_cm = ContextManager(db_path=_DB_PATH)

app = FastAPI(
    title="openContext API",
    description="AI context management database – REST interface",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class MessageIn(_BaseModel):
    role: str
    content: str
    metadata: Dict[str, Any] = {}


class AugmentRequest(_BaseModel):
    user_id: str
    messages: List[MessageIn]
    session_id: Optional[str] = None


class RecordRequest(_BaseModel):
    user_id: str
    session_id: str
    messages: List[MessageIn]


class AddMemoryRequest(_BaseModel):
    user_id: str
    content: str
    memory_type: str = "fact"
    importance: float = 0.7
    tags: List[str] = []


class DeleteMemoryRequest(_BaseModel):
    memory_id: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/api/v1/augment")
def augment(req: AugmentRequest) -> Dict[str, Any]:
    """Return messages augmented with relevant context."""
    raw = [{"role": m.role, "content": m.content, **m.metadata} for m in req.messages]
    augmented = _cm.augment_messages(
        user_id=req.user_id,
        messages=raw,
        session_id=req.session_id,
    )
    return {"messages": augmented}


@app.post("/api/v1/record")
def record(req: RecordRequest) -> Dict[str, Any]:
    """Record a completed conversation turn and extract memories."""
    raw = [{"role": m.role, "content": m.content, **m.metadata} for m in req.messages]
    session = _cm.record_conversation(
        user_id=req.user_id,
        session_id=req.session_id,
        messages=raw,
    )
    return {"session_id": session.id, "message_count": len(session.messages)}


@app.post("/api/v1/memories")
def add_memory(req: AddMemoryRequest) -> Dict[str, Any]:
    """Manually add a memory."""
    try:
        mem_type = MemoryType(req.memory_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown memory_type: {req.memory_type}")
    memory = _cm.add_memory(
        user_id=req.user_id,
        content=req.content,
        memory_type=mem_type,
        importance=req.importance,
        tags=req.tags,
    )
    return memory.to_dict()


@app.get("/api/v1/memories")
def get_memories(
    user_id: str = Query(...),
    memory_type: Optional[str] = Query(None),
    limit: int = Query(100, le=1000),
) -> Dict[str, Any]:
    """List memories for a user."""
    mem_type: Optional[MemoryType] = None
    if memory_type:
        try:
            mem_type = MemoryType(memory_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown memory_type: {memory_type}")
    memories = _cm.get_memories(user_id, memory_type=mem_type, limit=limit)
    return {"memories": [m.to_dict() for m in memories], "count": len(memories)}


@app.delete("/api/v1/memories/{memory_id}")
def delete_memory(memory_id: str) -> Dict[str, Any]:
    """Delete a specific memory."""
    deleted = _cm.delete_memory(memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"deleted": True, "memory_id": memory_id}


@app.get("/api/v1/memories/search")
def search_memories(
    user_id: str = Query(...),
    query: str = Query(...),
    limit: int = Query(20, le=100),
) -> Dict[str, Any]:
    """Full-text search over user memories."""
    memories = _cm.search_memories(user_id, query, limit=limit)
    return {"memories": [m.to_dict() for m in memories], "count": len(memories)}


@app.get("/api/v1/profile")
def get_profile(user_id: str = Query(...)) -> Dict[str, Any]:
    """Return the aggregated user profile."""
    profile = _cm.get_user_profile(user_id)
    if not profile:
        return {"user_id": user_id, "exists": False}
    return {**profile.to_dict(), "exists": True}


@app.get("/api/v1/sessions")
def get_sessions(
    user_id: str = Query(...),
    limit: int = Query(50, le=200),
) -> Dict[str, Any]:
    """List recent sessions for a user."""
    sessions = _cm.get_sessions(user_id, limit=limit)
    return {
        "sessions": [
            {
                "id": s.id,
                "user_id": s.user_id,
                "message_count": len(s.messages),
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ],
        "count": len(sessions),
    }


@app.delete("/api/v1/users/{user_id}")
def clear_user_data(user_id: str) -> Dict[str, Any]:
    """Delete all data for a user (GDPR / privacy)."""
    _cm.clear_user_data(user_id)
    return {"cleared": True, "user_id": user_id}


@app.get("/api/v1/context")
def get_context(
    user_id: str = Query(...),
    query: str = Query(...),
    session_id: Optional[str] = Query(None),
) -> Dict[str, Any]:
    """Get relevant context for a query without augmenting messages."""
    result = _cm.get_context(user_id=user_id, query=query, session_id=session_id)
    return {
        "injected_system_prompt": result.injected_system_prompt,
        "memories": [m.to_dict() for m in result.memories],
        "relevance_scores": result.relevance_scores,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "opencontext.server.app:app",
        host=os.environ.get("OPENCONTEXT_HOST", "0.0.0.0"),
        port=int(os.environ.get("OPENCONTEXT_PORT", "8765")),
        reload=False,
    )
