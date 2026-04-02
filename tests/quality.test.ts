import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { Storage } from "../src/storage.js";
import { SessionCapture } from "../src/capture.js";
import { Exporter } from "../src/export.js";
import { scoreSession, sessionFingerprint, deduplicateSessions, computeDataStats } from "../src/quality.js";
import { estimateTokens, estimateTokensForMessages } from "../src/tokens.js";
import fs from "fs";
import path from "path";
import os from "os";

let storage: Storage;
let capture: SessionCapture;
let exporter: Exporter;
let dbPath: string;

beforeEach(() => {
  dbPath = path.join(os.tmpdir(), `flywheel-quality-${Date.now()}-${Math.random().toString(36).slice(2)}.db`);
  storage = new Storage(dbPath);
  capture = new SessionCapture(storage);
  exporter = new Exporter(storage);
});

afterEach(() => {
  storage.close();
  for (const ext of ["", "-wal", "-shm"]) {
    try { fs.unlinkSync(dbPath + ext); } catch {}
  }
});

// ── Token estimation ────────────────────────────────────────────

describe("Token estimation", () => {
  it("estimates English text at ~4 chars/token", () => {
    const text = "Hello world, this is a test of the token estimation system.";
    const tokens = estimateTokens(text);
    expect(tokens).toBeGreaterThan(10);
    expect(tokens).toBeLessThan(30);
  });

  it("estimates code at ~3.5 chars/token", () => {
    const code = 'function foo() { return bar[0] + baz.map(x => x * 2); }';
    const tokens = estimateTokens(code);
    expect(tokens).toBeGreaterThan(10);
    expect(tokens).toBeLessThan(30);
  });

  it("estimates CJK text at ~1.5 chars/token", () => {
    const cjk = "こんにちは世界、これはテストです。日本語のテキストを処理しています。";
    const tokens = estimateTokens(cjk);
    expect(tokens).toBeGreaterThan(15);
  });

  it("returns 0 for empty string", () => {
    expect(estimateTokens("")).toBe(0);
  });

  it("estimates messages with overhead", () => {
    const msgs = [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there, how can I help?" },
    ];
    const total = estimateTokensForMessages(msgs);
    expect(total).toBeGreaterThan(8); // at least the overhead
  });
});

// ── Quality scoring ─────────────────────────────────────────────

describe("Quality scoring", () => {
  it("scores a good session with tool calls highly", () => {
    const id = capture.start();
    capture.addMessage(id, { role: "user", content: "Fix the bug in auth.ts" });
    capture.addMessage(id, { role: "assistant", content: "Reading file", toolName: "read_file" });
    capture.addMessage(id, { role: "tool", content: "file contents", toolCallId: "tc_1" });
    capture.addMessage(id, { role: "assistant", content: "I found and fixed the issue." });
    capture.stop(id);

    const session = storage.getSession(id)!;
    const messages = storage.getMessages(id);
    const quality = scoreSession(session, messages);

    expect(quality.score).toBeGreaterThan(0.6);
    expect(quality.hasToolCalls).toBe(true);
    expect(quality.hasErrors).toBe(false);
    expect(quality.reasons).toContain("contains_tool_calls");
    expect(quality.reasons).toContain("multi_turn");
  });

  it("scores a session with errors lower", () => {
    const id = capture.start();
    capture.addMessage(id, { role: "user", content: "test" });
    capture.addMessage(id, { role: "tool", content: "error: something went wrong" });
    capture.stop(id);

    const session = storage.getSession(id)!;
    const messages = storage.getMessages(id);
    const quality = scoreSession(session, messages);

    expect(quality.score).toBeLessThan(0.5);
    expect(quality.hasErrors).toBe(true);
    expect(quality.reasons).toContain("contains_errors");
  });

  it("penalizes single-message sessions", () => {
    const id = capture.start();
    capture.addMessage(id, { role: "user", content: "hi" });
    capture.stop(id);

    const session = storage.getSession(id)!;
    const messages = storage.getMessages(id);
    const quality = scoreSession(session, messages);

    expect(quality.score).toBeLessThan(0.4);
    expect(quality.reasons).toContain("too_short");
  });

  it("rewards successful tool call patterns", () => {
    const id = capture.start();
    capture.addMessage(id, { role: "user", content: "Read and fix the file" });
    capture.addMessage(id, { role: "assistant", content: "reading", toolName: "read_file" });
    capture.addMessage(id, { role: "tool", content: "contents", toolCallId: "tc_1" });
    capture.addMessage(id, { role: "assistant", content: "Fixed it" });
    capture.addMessage(id, { role: "assistant", content: "writing", toolName: "write_file" });
    capture.addMessage(id, { role: "tool", content: "ok", toolCallId: "tc_2" });
    capture.addMessage(id, { role: "assistant", content: "Done writing" });
    capture.stop(id);

    const session = storage.getSession(id)!;
    const messages = storage.getMessages(id);
    const quality = scoreSession(session, messages);

    expect(quality.reasons.some((r) => r.startsWith("successful_tool_patterns"))).toBe(true);
  });

  it("quality score is stored on session stop", () => {
    const id = capture.start();
    capture.addMessage(id, { role: "user", content: "Hello" });
    capture.addMessage(id, { role: "assistant", content: "Hi there" });
    capture.addMessage(id, { role: "user", content: "Thanks" });
    capture.addMessage(id, { role: "assistant", content: "You're welcome" });
    capture.stop(id);

    const session = storage.getSession(id)!;
    expect(session.quality_score).toBeGreaterThan(0);
    expect(session.fingerprint).toBeTruthy();
  });
});

// ── Deduplication ───────────────────────────────────────────────

describe("Deduplication", () => {
  it("detects duplicate sessions with same first/last messages", () => {
    const sessions = [
      {
        id: "a",
        messages: [
          { id: "1", session_id: "a", role: "user", content: "Fix the auth bug", tool_call_id: null, tool_name: null, timestamp: "", token_estimate: 0 },
          { id: "2", session_id: "a", role: "assistant", content: "Done, I fixed it", tool_call_id: null, tool_name: null, timestamp: "", token_estimate: 0 },
        ],
      },
      {
        id: "b",
        messages: [
          { id: "3", session_id: "b", role: "user", content: "Fix the auth bug", tool_call_id: null, tool_name: null, timestamp: "", token_estimate: 0 },
          { id: "4", session_id: "b", role: "assistant", content: "Done, I fixed it", tool_call_id: null, tool_name: null, timestamp: "", token_estimate: 0 },
        ],
      },
    ];

    const keep = deduplicateSessions(sessions);
    expect(keep.size).toBe(1);
    expect(keep.has("a")).toBe(true);
  });

  it("keeps distinct sessions", () => {
    const sessions = [
      {
        id: "a",
        messages: [
          { id: "1", session_id: "a", role: "user", content: "Fix the auth bug", tool_call_id: null, tool_name: null, timestamp: "", token_estimate: 0 },
          { id: "2", session_id: "a", role: "assistant", content: "Done fixing auth", tool_call_id: null, tool_name: null, timestamp: "", token_estimate: 0 },
        ],
      },
      {
        id: "b",
        messages: [
          { id: "3", session_id: "b", role: "user", content: "Add unit tests for payment", tool_call_id: null, tool_name: null, timestamp: "", token_estimate: 0 },
          { id: "4", session_id: "b", role: "assistant", content: "Tests added and passing", tool_call_id: null, tool_name: null, timestamp: "", token_estimate: 0 },
        ],
      },
    ];

    const keep = deduplicateSessions(sessions);
    expect(keep.size).toBe(2);
  });

  it("dedup works in export pipeline", () => {
    // Create two duplicate sessions
    for (let i = 0; i < 2; i++) {
      const id = capture.start();
      capture.addMessage(id, { role: "user", content: "Fix the same bug" });
      capture.addMessage(id, { role: "assistant", content: "Fixed the same way" });
      capture.stop(id);
    }

    // Create one unique session
    const id3 = capture.start();
    capture.addMessage(id3, { role: "user", content: "Different task entirely" });
    capture.addMessage(id3, { role: "assistant", content: "Different response" });
    capture.stop(id3);

    const withDedup = exporter.exportWithOptions({ deduplicate: true });
    const withoutDedup = exporter.exportWithOptions({ deduplicate: false });

    const dedupLines = withDedup.split("\n").filter(Boolean).length;
    const noDedupLines = withoutDedup.split("\n").filter(Boolean).length;

    expect(noDedupLines).toBe(3);
    expect(dedupLines).toBe(2);
  });
});

// ── Multi-format export ─────────────────────────────────────────

describe("Multi-format export", () => {
  function createTestSession() {
    const id = capture.start();
    capture.addMessage(id, { role: "user", content: "Write a function to sort an array" });
    capture.addMessage(id, { role: "assistant", content: "Here is the sort function:\n\nfunction sort(arr) { return arr.sort(); }" });
    capture.stop(id);
    return id;
  }

  it("exports ChatML format", () => {
    createTestSession();
    const jsonl = exporter.exportWithOptions({ format: "chatml" });
    const parsed = JSON.parse(jsonl.split("\n")[0]);

    expect(parsed.messages).toBeDefined();
    expect(parsed.messages[0].role).toBe("system");
    expect(parsed.messages[1].role).toBe("user");
    expect(parsed.messages[2].role).toBe("assistant");
  });

  it("exports Alpaca format", () => {
    createTestSession();
    const jsonl = exporter.exportWithOptions({ format: "alpaca" });
    const parsed = JSON.parse(jsonl.split("\n")[0]);

    expect(parsed.instruction).toBeDefined();
    expect(parsed.input).toBeDefined();
    expect(parsed.output).toBeDefined();
    expect(parsed.instruction).toContain("sort");
    expect(parsed.output).toContain("sort");
  });

  it("exports ShareGPT format", () => {
    createTestSession();
    const jsonl = exporter.exportWithOptions({ format: "sharegpt" });
    const parsed = JSON.parse(jsonl.split("\n")[0]);

    expect(parsed.conversations).toBeDefined();
    expect(parsed.conversations[0].from).toBe("human");
    expect(parsed.conversations[1].from).toBe("gpt");
  });

  it("exports raw format (no system prompt)", () => {
    createTestSession();
    const jsonl = exporter.exportWithOptions({ format: "raw" });
    const parsed = JSON.parse(jsonl.split("\n")[0]);

    expect(parsed.messages).toBeDefined();
    expect(parsed.messages[0].role).toBe("user"); // no system prompt
  });

  it("strips system prompt when requested", () => {
    createTestSession();
    const jsonl = exporter.exportWithOptions({ format: "chatml", stripSystemPrompt: true });
    const parsed = JSON.parse(jsonl.split("\n")[0]);

    expect(parsed.messages[0].role).toBe("user"); // system prompt stripped
  });
});

// ── Export validation ───────────────────────────────────────────

describe("Export validation", () => {
  it("validates correct ChatML JSONL", () => {
    const id = capture.start();
    capture.addMessage(id, { role: "user", content: "test" });
    capture.addMessage(id, { role: "assistant", content: "reply" });
    capture.stop(id);

    const jsonl = exporter.exportWithOptions({ format: "chatml" });
    const errors = exporter.validateExport(jsonl);
    expect(errors).toHaveLength(0);
  });

  it("validates correct Alpaca JSONL", () => {
    const id = capture.start();
    capture.addMessage(id, { role: "user", content: "test" });
    capture.addMessage(id, { role: "assistant", content: "reply" });
    capture.stop(id);

    const jsonl = exporter.exportWithOptions({ format: "alpaca" });
    const errors = exporter.validateExport(jsonl);
    expect(errors).toHaveLength(0);
  });

  it("validates correct ShareGPT JSONL", () => {
    const id = capture.start();
    capture.addMessage(id, { role: "user", content: "test" });
    capture.addMessage(id, { role: "assistant", content: "reply" });
    capture.stop(id);

    const jsonl = exporter.exportWithOptions({ format: "sharegpt" });
    const errors = exporter.validateExport(jsonl);
    expect(errors).toHaveLength(0);
  });

  it("detects invalid JSON", () => {
    const errors = exporter.validateExport("not json\nalso not json");
    expect(errors).toHaveLength(2);
    expect(errors[0]).toContain("invalid JSON");
  });

  it("detects empty export", () => {
    const errors = exporter.validateExport("");
    expect(errors).toHaveLength(1);
    expect(errors[0]).toContain("Empty export");
  });
});

// ── Data statistics ─────────────────────────────────────────────

describe("Data statistics", () => {
  it("computes stats across sessions", () => {
    // Create varied sessions
    for (let i = 0; i < 5; i++) {
      const id = capture.start();
      capture.addMessage(id, { role: "user", content: `Task ${i}` });
      capture.addMessage(id, { role: "assistant", content: `Reading`, toolName: "read_file" });
      capture.addMessage(id, { role: "tool", content: "result", toolCallId: `tc_${i}` });
      capture.addMessage(id, { role: "assistant", content: `Done ${i}` });
      capture.stop(id);
    }

    for (let i = 0; i < 3; i++) {
      const id = capture.start();
      capture.addMessage(id, { role: "user", content: `Simple ${i}` });
      capture.addMessage(id, { role: "assistant", content: `Reply ${i}` });
      capture.stop(id);
    }

    const stats = exporter.getDataStats();

    expect(stats.totalSessions).toBe(8);
    expect(stats.totalTokens).toBeGreaterThan(0);
    expect(stats.avgTokensPerSession).toBeGreaterThan(0);
    expect(stats.avgTurnsPerSession).toBeGreaterThan(0);
    expect(stats.avgQualityScore).toBeGreaterThan(0);
    expect(stats.toolCallDistribution["read_file"]).toBe(5);
    expect(stats.qualityDistribution.high + stats.qualityDistribution.medium + stats.qualityDistribution.low).toBe(8);
  });

  it("handles empty database", () => {
    const stats = exporter.getDataStats();
    expect(stats.totalSessions).toBe(0);
    expect(stats.avgQualityScore).toBe(0);
  });
});

// ── Quality-based filtering ─────────────────────────────────────

describe("Quality-based filtering", () => {
  it("filters by minimum quality score", () => {
    // High quality: tool calls, multi-turn
    const good = capture.start();
    capture.addMessage(good, { role: "user", content: "Fix the bug" });
    capture.addMessage(good, { role: "assistant", content: "Reading", toolName: "read" });
    capture.addMessage(good, { role: "tool", content: "data", toolCallId: "tc_1" });
    capture.addMessage(good, { role: "assistant", content: "Fixed it" });
    capture.stop(good);

    // Low quality: minimal two-turn session
    const bad = capture.start();
    capture.addMessage(bad, { role: "user", content: "x" });
    capture.addMessage(bad, { role: "assistant", content: "y" });
    capture.stop(bad);

    const highQuality = exporter.exportWithOptions({ minQuality: 0.6 });
    const allQuality = exporter.exportWithOptions();

    const highLines = highQuality.split("\n").filter(Boolean).length;
    const allLines = allQuality.split("\n").filter(Boolean).length;

    expect(allLines).toBe(2);
    expect(highLines).toBe(1);
  });
});
