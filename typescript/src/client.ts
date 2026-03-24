/**
 * openContext REST API client for TypeScript/JavaScript.
 *
 * @example
 * ```typescript
 * import { OpenContextClient } from "@opencontext/sdk";
 *
 * const client = new OpenContextClient({ userId: "alice" });
 *
 * // Augment messages before sending to LLM
 * const { messages } = await client.augment({
 *   messages: [{ role: "user", content: "What Python libs should I use?" }],
 * });
 *
 * // Record the conversation after receiving a response
 * await client.record({
 *   sessionId: "session-1",
 *   messages: [...messages, { role: "assistant", content: "..." }],
 * });
 * ```
 */

import {
  AugmentResult,
  ChatMessage,
  ContextResult,
  Memory,
  MemoryListResult,
  MemoryType,
  OpenContextClientOptions,
  RecordResult,
  Session,
  SessionListResult,
  UserProfile,
} from "./types";

export class OpenContextClient {
  private serverUrl: string;
  private defaultUserId: string | undefined;
  private timeoutMs: number;

  constructor(options: OpenContextClientOptions = {}) {
    this.serverUrl = (options.serverUrl ?? "http://localhost:8765").replace(/\/$/, "");
    this.defaultUserId = options.userId;
    this.timeoutMs = options.timeoutMs ?? 10_000;
  }

  // ------------------------------------------------------------------ //
  // Core context operations
  // ------------------------------------------------------------------ //

  /**
   * Augment a list of messages with relevant user memories.
   * Call this before sending messages to the LLM.
   */
  async augment(params: {
    messages: ChatMessage[];
    userId?: string;
    sessionId?: string;
  }): Promise<AugmentResult> {
    const userId = this._userId(params.userId);
    return this._post<AugmentResult>("/api/v1/augment", {
      user_id: userId,
      messages: params.messages,
      session_id: params.sessionId ?? null,
    });
  }

  /**
   * Record a completed conversation and extract new memories.
   * Call this after receiving the LLM response.
   */
  async record(params: {
    sessionId: string;
    messages: ChatMessage[];
    userId?: string;
  }): Promise<RecordResult> {
    const userId = this._userId(params.userId);
    return this._post<RecordResult>("/api/v1/record", {
      user_id: userId,
      session_id: params.sessionId,
      messages: params.messages,
    });
  }

  /**
   * Get relevant context for a query without augmenting messages.
   */
  async getContext(params: {
    query: string;
    userId?: string;
    sessionId?: string;
  }): Promise<ContextResult> {
    const userId = this._userId(params.userId);
    const search = new URLSearchParams({ user_id: userId, query: params.query });
    if (params.sessionId) search.set("session_id", params.sessionId);
    return this._get<ContextResult>(`/api/v1/context?${search}`);
  }

  // ------------------------------------------------------------------ //
  // Memory management
  // ------------------------------------------------------------------ //

  /**
   * Add a memory manually.
   */
  async addMemory(params: {
    content: string;
    memoryType?: MemoryType;
    importance?: number;
    tags?: string[];
    userId?: string;
  }): Promise<Memory> {
    const userId = this._userId(params.userId);
    return this._post<Memory>("/api/v1/memories", {
      user_id: userId,
      content: params.content,
      memory_type: params.memoryType ?? "fact",
      importance: params.importance ?? 0.7,
      tags: params.tags ?? [],
    });
  }

  /**
   * List all memories for a user.
   */
  async getMemories(params: {
    userId?: string;
    memoryType?: MemoryType;
    limit?: number;
  } = {}): Promise<MemoryListResult> {
    const userId = this._userId(params.userId);
    const search = new URLSearchParams({ user_id: userId });
    if (params.memoryType) search.set("memory_type", params.memoryType);
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    return this._get<MemoryListResult>(`/api/v1/memories?${search}`);
  }

  /**
   * Delete a specific memory by ID.
   */
  async deleteMemory(memoryId: string): Promise<{ deleted: boolean; memory_id: string }> {
    return this._delete(`/api/v1/memories/${encodeURIComponent(memoryId)}`);
  }

  /**
   * Full-text search over a user's memories.
   */
  async searchMemories(params: {
    query: string;
    userId?: string;
    limit?: number;
  }): Promise<MemoryListResult> {
    const userId = this._userId(params.userId);
    const search = new URLSearchParams({ user_id: userId, query: params.query });
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    return this._get<MemoryListResult>(`/api/v1/memories/search?${search}`);
  }

  // ------------------------------------------------------------------ //
  // User profile & sessions
  // ------------------------------------------------------------------ //

  /**
   * Get the aggregated user profile.
   */
  async getUserProfile(userId?: string): Promise<UserProfile> {
    const uid = this._userId(userId);
    return this._get<UserProfile>(`/api/v1/profile?user_id=${encodeURIComponent(uid)}`);
  }

  /**
   * List recent sessions for a user.
   */
  async getSessions(params: { userId?: string; limit?: number } = {}): Promise<SessionListResult> {
    const userId = this._userId(params.userId);
    const search = new URLSearchParams({ user_id: userId });
    if (params.limit !== undefined) search.set("limit", String(params.limit));
    return this._get<SessionListResult>(`/api/v1/sessions?${search}`);
  }

  /**
   * Clear all data for a user (GDPR / privacy).
   */
  async clearUserData(userId?: string): Promise<{ cleared: boolean; user_id: string }> {
    const uid = this._userId(userId);
    return this._delete(`/api/v1/users/${encodeURIComponent(uid)}`);
  }

  // ------------------------------------------------------------------ //
  // HTTP helpers
  // ------------------------------------------------------------------ //

  private _userId(provided?: string): string {
    const uid = provided ?? this.defaultUserId;
    if (!uid) throw new Error("userId is required (pass it or set defaultUserId in constructor)");
    return uid;
  }

  private async _post<T>(path: string, body: unknown): Promise<T> {
    const resp = await this._fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return resp.json() as Promise<T>;
  }

  private async _get<T>(path: string): Promise<T> {
    const resp = await this._fetch(path, { method: "GET" });
    return resp.json() as Promise<T>;
  }

  private async _delete<T>(path: string): Promise<T> {
    const resp = await this._fetch(path, { method: "DELETE" });
    return resp.json() as Promise<T>;
  }

  private async _fetch(path: string, init: RequestInit): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const resp = await fetch(`${this.serverUrl}${path}`, {
        ...init,
        signal: controller.signal,
      });
      if (!resp.ok) {
        const text = await resp.text().catch(() => "");
        throw new Error(`openContext API error ${resp.status}: ${text}`);
      }
      return resp;
    } finally {
      clearTimeout(timer);
    }
  }
}
