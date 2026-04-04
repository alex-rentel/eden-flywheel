import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { Storage } from "../src/storage.js";
import {
  generateTrainingData,
  validateBatch,
  parseGeneratedConversation,
  buildSystemPrompt,
} from "../src/generate.js";
import fs from "fs";
import path from "path";
import os from "os";

let storage: Storage;
let dbPath: string;

beforeEach(() => {
  dbPath = path.join(os.tmpdir(), `flywheel-gen-${Date.now()}-${Math.random().toString(36).slice(2)}.db`);
  storage = new Storage(dbPath);
});

afterEach(() => {
  storage.close();
  for (const ext of ["", "-wal", "-shm"]) {
    try { fs.unlinkSync(dbPath + ext); } catch {}
  }
});

describe("parseGeneratedConversation", () => {
  it("parses valid ChatML JSON", () => {
    const input = JSON.stringify({
      messages: [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Fix the bug" },
        { role: "assistant", content: "Done" },
      ],
    });
    const result = parseGeneratedConversation(input);
    expect(result).not.toBeNull();
    expect(result!.messages).toHaveLength(3);
  });

  it("strips markdown code blocks", () => {
    const json = JSON.stringify({
      messages: [
        { role: "user", content: "test" },
        { role: "assistant", content: "ok" },
      ],
    });
    const input = "```json\n" + json + "\n```";
    const result = parseGeneratedConversation(input);
    expect(result).not.toBeNull();
    expect(result!.messages).toHaveLength(2);
  });

  it("returns null for invalid JSON", () => {
    expect(parseGeneratedConversation("not json")).toBeNull();
  });

  it("returns null when missing user or assistant", () => {
    const input = JSON.stringify({
      messages: [{ role: "system", content: "test" }],
    });
    expect(parseGeneratedConversation(input)).toBeNull();
  });

  it("returns null when messages is not an array", () => {
    expect(parseGeneratedConversation('{"messages": "nope"}')).toBeNull();
  });

  it("returns null when message lacks role or content", () => {
    const input = JSON.stringify({
      messages: [
        { role: "user" }, // missing content
        { role: "assistant", content: "ok" },
      ],
    });
    expect(parseGeneratedConversation(input)).toBeNull();
  });
});

describe("buildSystemPrompt", () => {
  it("includes tool schemas when provided", () => {
    const prompt = buildSystemPrompt(["Read: Read files", "Bash: Run commands"], "easy");
    expect(prompt).toContain("Read: Read files");
    expect(prompt).toContain("Bash: Run commands");
    expect(prompt).toContain("simple, single-step");
  });

  it("uses default tools when none provided", () => {
    const prompt = buildSystemPrompt([], "hard");
    expect(prompt).toContain("Read: Read a file");
    expect(prompt).toContain("Bash: Run a shell command");
    expect(prompt).toContain("complex tool call");
  });

  it("falls back to medium difficulty for unknown values", () => {
    const prompt = buildSystemPrompt([], "medium");
    expect(prompt).toContain("multi-step");
  });
});

describe("generateTrainingData", () => {
  it("returns error when no API key", async () => {
    const originalKey = process.env.OPENROUTER_API_KEY;
    delete process.env.OPENROUTER_API_KEY;

    const result = await generateTrainingData(storage, { count: 1 });
    expect(result.stored).toBe(0);
    expect(result.errors[0]).toContain("OPENROUTER_API_KEY");

    if (originalKey) process.env.OPENROUTER_API_KEY = originalKey;
  });

  it("generates and stores conversations with mocked API", async () => {
    const mockResponse = JSON.stringify({
      messages: [
        { role: "system", content: "You are helpful." },
        { role: "user", content: "Read the file" },
        { role: "assistant", content: 'Let me read it. <tool_call>{"name": "Read", "arguments": "/src/index.ts"}</tool_call>' },
        { role: "tool", content: "file contents" },
        { role: "assistant", content: "The file contains the main entry point." },
      ],
    });

    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      new Response(JSON.stringify({
        choices: [{ message: { content: mockResponse } }],
      }), { status: 200, headers: { "Content-Type": "application/json" } })
    );

    const result = await generateTrainingData(storage, { count: 2 }, "test-key");
    expect(result.stored).toBe(2);
    expect(result.batchId).toMatch(/^gen-/);
    expect(result.errors).toHaveLength(0);

    // Verify sessions were stored
    const sessions = storage.listSessions();
    expect(sessions.length).toBe(2);

    fetchSpy.mockRestore();
  });

  it("handles API error responses", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("Rate limited", { status: 429 })
    );

    const result = await generateTrainingData(storage, { count: 1 }, "test-key");
    expect(result.stored).toBe(0);
    expect(result.errors[0]).toContain("429");

    fetchSpy.mockRestore();
  });

  it("handles unparseable responses", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({
        choices: [{ message: { content: "This is not JSON" } }],
      }), { status: 200, headers: { "Content-Type": "application/json" } })
    );

    const result = await generateTrainingData(storage, { count: 1 }, "test-key");
    expect(result.stored).toBe(0);
    expect(result.errors[0]).toContain("parse");

    fetchSpy.mockRestore();
  });
});

describe("validateBatch", () => {
  it("returns empty result when no API key", async () => {
    const originalKey = process.env.ANTHROPIC_API_KEY;
    delete process.env.ANTHROPIC_API_KEY;

    const result = await validateBatch(storage, "gen-fake", 5);
    expect(result.sampled).toBe(0);

    if (originalKey) process.env.ANTHROPIC_API_KEY = originalKey;
  });

  it("returns empty result for unknown batch", async () => {
    const result = await validateBatch(storage, "gen-nonexistent", 5, "test-key");
    expect(result.sampled).toBe(0);
  });

  it("validates sessions with mocked Claude API", async () => {
    // First create some synthetic sessions
    const batchId = "gen-test-validate";
    const sid = storage.createSession({ source: "synthetic", batchId });
    storage.addMessage(sid, "user", "Help me debug this");
    storage.addMessage(sid, "assistant", "Let me look at the code");
    storage.stopSession(sid);

    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({
        content: [{ type: "text", text: '{"score": 4, "feedback": "Good conversation structure"}' }],
      }), { status: 200, headers: { "Content-Type": "application/json" } })
    );

    const result = await validateBatch(storage, batchId, 5, "test-key");
    expect(result.sampled).toBe(1);
    expect(result.avgScore).toBe(4);
    expect(result.flagged).toBe(0);

    fetchSpy.mockRestore();
  });
});
