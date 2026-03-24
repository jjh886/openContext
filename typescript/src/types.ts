/**
 * openContext TypeScript SDK – type definitions.
 */

export type MemoryType =
  | "fact"
  | "skill"
  | "preference"
  | "contact"
  | "task"
  | "summary"
  | "habit";

export type MessageRole = "user" | "assistant" | "system";

export interface Memory {
  id: string;
  content: string;
  memory_type: MemoryType;
  importance: number;
  tags: string[];
  source_session_id: string | null;
  created_at: string;
  updated_at: string;
  last_accessed: string;
  access_count: number;
  metadata: Record<string, unknown>;
}

export interface ChatMessage {
  role: MessageRole;
  content: string;
  [key: string]: unknown;
}

export interface UserProfile {
  user_id: string;
  name: string | null;
  skills: string[];
  preferences: Record<string, unknown>;
  habits: string[];
  facts: string[];
  updated_at: string;
  exists?: boolean;
}

export interface Session {
  id: string;
  user_id: string;
  message_count: number;
  created_at: string;
  updated_at: string;
}

export interface ContextResult {
  injected_system_prompt: string;
  memories: Memory[];
  relevance_scores: Record<string, number>;
}

export interface AugmentResult {
  messages: ChatMessage[];
}

export interface RecordResult {
  session_id: string;
  message_count: number;
}

export interface MemoryListResult {
  memories: Memory[];
  count: number;
}

export interface SessionListResult {
  sessions: Session[];
  count: number;
}

export interface OpenContextClientOptions {
  /** Base URL of the openContext REST server. Default: http://localhost:8765 */
  serverUrl?: string;
  /** Default user_id used when not provided explicitly on each call. */
  userId?: string;
  /** Fetch timeout in milliseconds. Default: 10000 */
  timeoutMs?: number;
}
