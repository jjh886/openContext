# Architecture Overview

## openContext System Design

### Component Diagram

```
Your Application
      │
      ├── Python SDK (direct SQLite access)
      │     └── ContextManager
      │           ├── Database (SQLite)
      │           ├── Extractor (rule-based)
      │           └── Filter (TF-IDF)
      │
      └── TypeScript / Go SDK (HTTP)
            └── REST API (FastAPI)
                  └── ContextManager (shared)
```

### Data Flow

#### Augment Flow (before LLM call)
1. Application provides messages + user_id
2. `ContextManager.augment_messages()` extracts the latest user query
3. `Database.get_memories()` fetches all stored memories for the user
4. `filter_memories()` scores each memory against the query using TF-IDF cosine similarity
5. Top-K relevant memories are formatted into a system prompt block
6. The system prompt is prepended to (or merged into) the messages list
7. Augmented messages are returned to the application for sending to the LLM

#### Record Flow (after LLM response)
1. Application provides the full conversation turn (user + assistant messages)
2. `extract_from_messages()` parses user messages with regex rules
3. New `Memory` objects are saved to the SQLite database
4. `_update_user_profile()` rebuilds the aggregated `UserProfile` from all memories
5. If ≥ 4 messages, a `SUMMARY` memory is generated and stored

### Database Schema

**memories** — One row per memory item
- `id` TEXT PRIMARY KEY (UUID)
- `user_id` TEXT NOT NULL (partition key)
- `content` TEXT NOT NULL
- `memory_type` TEXT (fact/skill/preference/habit/task/contact/summary)
- `importance` REAL (0.0–1.0)
- `tags` TEXT (JSON array)
- `source_session_id` TEXT
- `created_at`, `updated_at`, `last_accessed` TEXT (ISO-8601)
- `access_count` INTEGER
- `metadata` TEXT (JSON object)

**sessions** — One row per conversation session
- `id` TEXT PRIMARY KEY
- `user_id` TEXT NOT NULL
- `messages` TEXT (JSON array)
- `created_at`, `updated_at` TEXT
- `metadata` TEXT (JSON)

**user_profiles** — One row per user (derived / cached)
- `user_id` TEXT PRIMARY KEY
- `name` TEXT
- `skills`, `habits`, `facts` TEXT (JSON arrays)
- `preferences` TEXT (JSON object)
- `updated_at` TEXT

### Relevance Scoring

Each memory is scored against the current query using:

```
score = (tfidf_cosine(query, memory) + 0.3 × importance) × type_boost
```

- `tfidf_cosine`: TF-IDF weighted cosine similarity between tokenised query and memory content
- `importance`: User-assigned or extracted importance (0–1)
- `type_boost`: Per-type multiplier (TASK=1.4, SKILL=1.3, FACT=1.2, …)

Memories are then sorted descending by score and the top-K are selected.

### Extraction Rules

The `extractor.py` module uses regular expressions to find:

| Pattern | Memory Type | Example match |
|---------|-------------|---------------|
| `I know / can / am skilled in …` | SKILL | "I know Python" |
| `I am a/an …` | SKILL | "I am a backend developer" |
| `I prefer / like / enjoy …` | PREFERENCE | "I prefer dark mode" |
| `My name is …` | FACT | "My name is Alice" |
| `I live / am based in …` | FACT | "I live in Berlin" |
| `I work at …` | FACT | "I work at Acme" |
| `I usually / always …` | HABIT | "I usually run in the morning" |
| `I need / want / plan to …` | TASK | "I need to deploy by Friday" |
| Technology keywords | SKILL | "react", "postgresql", "docker", … |

### REST API Contract

All endpoints live under `/api/v1/`.  
The server is a stateless FastAPI application backed by a single `ContextManager` instance (configured via env vars).

See `README.md` for the full endpoint table.
