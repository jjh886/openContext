/**
 * Tests for OpenContextClient – mocked HTTP layer so no server needed.
 */

import { OpenContextClient } from "../src/client";

// Mock global fetch
const mockFetch = jest.fn();
global.fetch = mockFetch as typeof fetch;

function mockOk(body: unknown): void {
  mockFetch.mockResolvedValueOnce({
    ok: true,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  } as unknown as Response);
}

function mockError(status: number, text: string): void {
  mockFetch.mockResolvedValueOnce({
    ok: false,
    status,
    json: () => Promise.resolve({ detail: text }),
    text: () => Promise.resolve(text),
  } as unknown as Response);
}

beforeEach(() => {
  mockFetch.mockClear();
});

describe("OpenContextClient", () => {
  const client = new OpenContextClient({ userId: "test-user" });

  // ---------------------------------------------------------------- //
  // augment
  // ---------------------------------------------------------------- //

  test("augment sends correct payload", async () => {
    const augmented = [
      { role: "system" as const, content: "## Context\n- Fact: I use Python" },
      { role: "user" as const, content: "Hello" },
    ];
    mockOk({ messages: augmented });

    const result = await client.augment({
      messages: [{ role: "user", content: "Hello" }],
      sessionId: "sess-1",
    });

    expect(result.messages).toHaveLength(2);
    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("http://localhost:8765/api/v1/augment");
    const body = JSON.parse(opts.body as string);
    expect(body.user_id).toBe("test-user");
    expect(body.session_id).toBe("sess-1");
  });

  // ---------------------------------------------------------------- //
  // record
  // ---------------------------------------------------------------- //

  test("record sends correct payload", async () => {
    mockOk({ session_id: "sess-1", message_count: 2 });

    const result = await client.record({
      sessionId: "sess-1",
      messages: [
        { role: "user", content: "I love TypeScript" },
        { role: "assistant", content: "Great!" },
      ],
    });

    expect(result.session_id).toBe("sess-1");
    expect(result.message_count).toBe(2);
    const [, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(opts.body as string);
    expect(body.session_id).toBe("sess-1");
    expect(body.messages).toHaveLength(2);
  });

  // ---------------------------------------------------------------- //
  // getContext
  // ---------------------------------------------------------------- //

  test("getContext builds correct query string", async () => {
    mockOk({ injected_system_prompt: "## Context", memories: [], relevance_scores: {} });

    await client.getContext({ query: "Python tips" });

    const [url] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/api/v1/context");
    expect(url).toContain("query=Python+tips");
    expect(url).toContain("user_id=test-user");
  });

  // ---------------------------------------------------------------- //
  // addMemory / getMemories / deleteMemory
  // ---------------------------------------------------------------- //

  test("addMemory sends correct payload", async () => {
    const mem = {
      id: "mem-1",
      content: "I know Rust",
      memory_type: "skill",
      importance: 0.8,
      tags: ["rust"],
      source_session_id: null,
      created_at: "2024-01-01T00:00:00",
      updated_at: "2024-01-01T00:00:00",
      last_accessed: "2024-01-01T00:00:00",
      access_count: 0,
      metadata: {},
    };
    mockOk(mem);

    const result = await client.addMemory({ content: "I know Rust", memoryType: "skill" });

    expect(result.id).toBe("mem-1");
    const [, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    const body = JSON.parse(opts.body as string);
    expect(body.content).toBe("I know Rust");
    expect(body.memory_type).toBe("skill");
  });

  test("getMemories includes memory_type filter when provided", async () => {
    mockOk({ memories: [], count: 0 });

    await client.getMemories({ memoryType: "skill" });

    const [url] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("memory_type=skill");
  });

  test("deleteMemory calls DELETE endpoint", async () => {
    mockOk({ deleted: true, memory_id: "mem-abc" });

    const result = await client.deleteMemory("mem-abc");

    expect(result.deleted).toBe(true);
    const [url, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/api/v1/memories/mem-abc");
    expect(opts.method).toBe("DELETE");
  });

  // ---------------------------------------------------------------- //
  // getUserProfile / getSessions / clearUserData
  // ---------------------------------------------------------------- //

  test("getUserProfile calls correct endpoint", async () => {
    mockOk({ user_id: "test-user", name: "Alice", skills: [], preferences: {}, habits: [], facts: [], updated_at: "2024-01-01T00:00:00", exists: true });

    const profile = await client.getUserProfile();
    expect(profile.user_id).toBe("test-user");
    const [url] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/api/v1/profile");
  });

  test("clearUserData calls DELETE on user endpoint", async () => {
    mockOk({ cleared: true, user_id: "test-user" });

    const result = await client.clearUserData();
    expect(result.cleared).toBe(true);
    const [url, opts] = mockFetch.mock.calls[0] as [string, RequestInit];
    expect(url).toContain("/api/v1/users/test-user");
    expect(opts.method).toBe("DELETE");
  });

  // ---------------------------------------------------------------- //
  // Error handling
  // ---------------------------------------------------------------- //

  test("throws on non-ok response", async () => {
    mockError(404, "Memory not found");

    await expect(client.deleteMemory("ghost-id")).rejects.toThrow(
      "openContext API error 404"
    );
  });

  test("throws when userId is missing", () => {
    const noUserClient = new OpenContextClient();
    expect(() => noUserClient["_userId"](undefined)).toThrow("userId is required");
  });
});
