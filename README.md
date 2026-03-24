# openContext

> **AI Memory Management Database & Multi-Language SDK**

openContext is an open-source library that gives AI assistants **true, persistent memory** across conversations.  
It automatically extracts information from multi-turn conversations, scores relevance, filters irrelevant context, and injects only the necessary memories into each LLM call — all without breaking existing setups.

---

## Features

| Feature | Description |
|---------|-------------|
| 🧠 **Persistent memory** | SQLite-backed storage for facts, skills, preferences, habits, tasks, contacts and conversation summaries |
| 🔍 **Auto-extraction** | Rule-based extraction of structured memories from user messages (no LLM call required for extraction) |
| ⚡ **Relevance filtering** | TF-IDF cosine similarity + importance weighting keeps only the most relevant memories per turn |
| 🔌 **Drop-in wrappers** | One-line integration with OpenAI and Anthropic – existing code keeps working |
| 🌐 **REST API server** | Language-agnostic FastAPI server used by the TypeScript and Go SDKs |
| 🛠️ **Multi-language SDK** | Python · TypeScript/JavaScript · Go (more coming) |
| 🔒 **Privacy** | Per-user data isolation; `clear_user_data()` / `DELETE /api/v1/users/{id}` for GDPR compliance |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Your Application                         │
│  ┌───────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │  Python SDK   │  │ TypeScript SDK  │  │     Go SDK       │  │
│  └──────┬────────┘  └────────┬────────┘  └────────┬─────────┘  │
│         │                    │ HTTP                │             │
│         │           ┌────────▼────────┐            │             │
│         │           │  FastAPI Server │◄───────────┘             │
│         │           └────────┬────────┘                          │
│         │ (direct)           │ (direct)                          │
│         └────────────────────┘                                   │
│                      │                                           │
│             ┌─────────▼──────────┐                               │
│             │  ContextManager    │                               │
│             │  ┌──────────────┐  │                               │
│             │  │  Extractor   │  │  ← rule-based memory mining   │
│             │  │  Filter      │  │  ← TF-IDF relevance scoring   │
│             │  │  Database    │  │  ← SQLite (zero dependencies) │
│             │  └──────────────┘  │                               │
│             └────────────────────┘                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Python

```bash
pip install -e python/
# with server support:
pip install -e "python/[server]"
# with OpenAI wrapper:
pip install -e "python/[openai]"
```

#### Direct integration (OpenAI)

```python
from opencontext.wrappers.openai_wrapper import OpenContextOpenAI

client = OpenContextOpenAI(
    user_id="alice",
    openai_api_key="sk-...",      # or set OPENAI_API_KEY
    db_path="memory.db",
    session_id="session-001",
)

# Memories are injected automatically
response = client.chat(messages=[
    {"role": "user", "content": "Which Python libraries are best for me?"}
])
print(response.choices[0].message.content)

# View what was remembered
for mem in client.get_memories():
    print(mem.memory_type.value, "—", mem.content)
```

#### Low-level API

```python
from opencontext import ContextManager, MemoryType

cm = ContextManager(db_path="memory.db")

# Manually add a memory
cm.add_memory("alice", "I am a backend developer", MemoryType.SKILL, importance=0.9)

# Augment messages before sending to any LLM
messages = cm.augment_messages("alice", [
    {"role": "user", "content": "What database should I use?"}
])

# After receiving the response, record the full turn
cm.record_conversation("alice", "sess-1", messages + [
    {"role": "assistant", "content": "PostgreSQL is a great choice for you."}
])
```

### TypeScript / JavaScript

```bash
cd typescript && npm install
```

```typescript
import { OpenContextClient } from "@opencontext/sdk";

const client = new OpenContextClient({ userId: "alice" });

// Augment messages (calls the running Python server)
const { messages } = await client.augment({
  messages: [{ role: "user", content: "What should I use for my backend?" }],
  sessionId: "sess-1",
});

// ... send `messages` to OpenAI ...

// Record the conversation
await client.record({
  sessionId: "sess-1",
  messages: [...messages, { role: "assistant", content: "..." }],
});

// List memories
const { memories } = await client.getMemories();
```

#### With the OpenAI TypeScript SDK

```typescript
import OpenAI from "openai";
import { OpenContextOpenAI } from "@opencontext/sdk";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const client = new OpenContextOpenAI({ userId: "alice", openai });

const response = await client.chat(
  [{ role: "user", content: "Which language should I learn?" }],
  { sessionId: "sess-1" }
);
```

### Go

```go
import opencontext "github.com/jjh886/openContext/golang/opencontext"

client := opencontext.NewClient(opencontext.Options{UserID: "alice"})

// Augment messages
result, _ := client.Augment(ctx, opencontext.AugmentRequest{
    Messages: []opencontext.Message{
        {Role: "user", Content: "What Go libraries should I use?"},
    },
    SessionID: "sess-1",
})

// Record the conversation
client.Record(ctx, opencontext.RecordRequest{
    SessionID: "sess-1",
    Messages:  append(result.Messages, opencontext.Message{
        Role: "assistant", Content: "...",
    }),
})
```

---

## REST API Server

Start the server (Python):

```bash
# Development
uvicorn opencontext.server.app:app --host 0.0.0.0 --port 8765 --reload

# Or run directly
python -m opencontext.server.app
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/augment` | Inject context into messages |
| `POST` | `/api/v1/record` | Record a conversation turn |
| `GET` | `/api/v1/context` | Get relevant context for a query |
| `POST` | `/api/v1/memories` | Add a memory |
| `GET` | `/api/v1/memories` | List memories |
| `DELETE` | `/api/v1/memories/{id}` | Delete a memory |
| `GET` | `/api/v1/memories/search` | Full-text search |
| `GET` | `/api/v1/profile` | Get user profile |
| `GET` | `/api/v1/sessions` | List sessions |
| `DELETE` | `/api/v1/users/{id}` | Delete all user data |

Interactive docs available at `http://localhost:8765/docs`.

---

## Memory Types

| Type | Description | Example |
|------|-------------|---------|
| `fact` | Personal facts | "My name is Alice", "I live in Berlin" |
| `skill` | Technical skills | "I know Python", "Uses PostgreSQL" |
| `preference` | User preferences | "I prefer dark mode", "I like concise answers" |
| `habit` | Regular behaviours | "I usually exercise in the morning" |
| `task` | Goals / reminders | "I need to deploy the API by Friday" |
| `contact` | People / organisations | "I work at Acme Corp" |
| `summary` | Session summaries | Auto-generated after long conversations |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENCONTEXT_DB_PATH` | `opencontext.db` | SQLite database path |
| `OPENCONTEXT_HOST` | `0.0.0.0` | Server bind address |
| `OPENCONTEXT_PORT` | `8765` | Server port |
| `OPENAI_API_KEY` | — | OpenAI API key (for the OpenAI wrapper) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (for the Anthropic wrapper) |

---

## Repository Structure

```
openContext/
├── python/
│   ├── opencontext/
│   │   ├── models.py           # Data models (Memory, Session, UserProfile, …)
│   │   ├── database.py         # SQLite database layer
│   │   ├── extractor.py        # Rule-based memory extraction
│   │   ├── filter.py           # TF-IDF relevance scoring & filtering
│   │   ├── context_manager.py  # Main ContextManager class
│   │   ├── wrappers/
│   │   │   ├── openai_wrapper.py    # OpenAI drop-in wrapper
│   │   │   └── anthropic_wrapper.py # Anthropic drop-in wrapper
│   │   └── server/
│   │       └── app.py          # FastAPI REST server
│   ├── tests/                  # Python tests (pytest)
│   ├── setup.py
│   └── requirements.txt
├── typescript/
│   ├── src/
│   │   ├── client.ts           # REST API client
│   │   ├── types.ts            # TypeScript type definitions
│   │   └── wrappers/
│   │       └── openai.ts       # OpenAI TypeScript wrapper
│   └── tests/                  # Jest tests
├── golang/
│   └── opencontext/
│       ├── client.go           # Go REST API client
│       └── client_test.go      # Go tests
└── docs/
    └── architecture.md
```

---

## Testing

```bash
# Python
cd python && python -m pytest tests/ -v

# TypeScript
cd typescript && npm test

# Go
cd golang && go test ./opencontext/... -v
```

---

## License

MIT — see [LICENSE](LICENSE).
