/**
 * openContext OpenAI wrapper for TypeScript.
 *
 * @example
 * ```typescript
 * import OpenAI from "openai";
 * import { OpenContextOpenAI } from "@opencontext/sdk/wrappers/openai";
 *
 * const client = new OpenContextOpenAI({
 *   userId: "alice",
 *   openai: new OpenAI({ apiKey: process.env.OPENAI_API_KEY }),
 * });
 *
 * const response = await client.chat([
 *   { role: "user", content: "What libraries would you recommend for me?" }
 * ], { sessionId: "session-1" });
 * ```
 */

import { OpenContextClient } from "../client";
import { ChatMessage, MemoryType, OpenContextClientOptions } from "../types";

export interface OpenContextOpenAIOptions extends OpenContextClientOptions {
  /** User ID for this client instance. */
  userId: string;
  /**
   * An instantiated OpenAI client.
   * Pass your own configured `new OpenAI({ apiKey: "..." })` instance.
   */
  openai: {
    chat: {
      completions: {
        create(params: {
          model: string;
          messages: ChatMessage[];
          [key: string]: unknown;
        }): Promise<{
          choices: Array<{ message: { content: string | null; role: string } }>;
          [key: string]: unknown;
        }>;
      };
    };
  };
  /** Default model to use. Default: gpt-3.5-turbo */
  model?: string;
}

export class OpenContextOpenAI {
  private client: OpenContextClient;
  private openai: OpenContextOpenAIOptions["openai"];
  private userId: string;
  private model: string;

  constructor(options: OpenContextOpenAIOptions) {
    this.userId = options.userId;
    this.model = options.model ?? "gpt-3.5-turbo";
    this.openai = options.openai;
    this.client = new OpenContextClient({
      serverUrl: options.serverUrl,
      userId: options.userId,
      timeoutMs: options.timeoutMs,
    });
  }

  /**
   * Send a chat request with automatic context injection and memory recording.
   */
  async chat(
    messages: ChatMessage[],
    options: {
      sessionId?: string;
      model?: string;
      injectContext?: boolean;
      record?: boolean;
      [key: string]: unknown;
    } = {}
  ): Promise<{ choices: Array<{ message: { content: string | null; role: string } }>; [key: string]: unknown }> {
    const { sessionId, model, injectContext = true, record = true, ...rest } = options;

    let outMessages = messages;
    if (injectContext) {
      const result = await this.client.augment({
        messages,
        userId: this.userId,
        sessionId,
      });
      outMessages = result.messages;
    }

    const response = await this.openai.chat.completions.create({
      model: model ?? this.model,
      messages: outMessages,
      ...rest,
    });

    if (record && sessionId) {
      const assistantContent = response.choices[0]?.message?.content ?? "";
      await this.client.record({
        sessionId,
        userId: this.userId,
        messages: [...outMessages, { role: "assistant", content: assistantContent }],
      });
    }

    return response;
  }

  /** Manually add a memory for this user. */
  async addMemory(
    content: string,
    memoryType: MemoryType = "fact",
    importance = 0.8
  ): Promise<void> {
    await this.client.addMemory({ content, memoryType, importance, userId: this.userId });
  }
}
