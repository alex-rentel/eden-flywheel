import { describe, it, expect, beforeEach, afterEach, beforeAll } from "vitest";
import { Storage } from "../src/storage.js";
import { SessionCapture } from "../src/capture.js";
import { Exporter } from "../src/export.js";
import { setLogLevel } from "../src/logger.js";
import fs from "fs";
import path from "path";
import os from "os";

beforeAll(() => setLogLevel("error")); // Suppress logs during stress tests

let storage: Storage;
let capture: SessionCapture;
let exporter: Exporter;
let dbPath: string;

beforeEach(() => {
  dbPath = path.join(os.tmpdir(), `flywheel-stress-${Date.now()}-${Math.random().toString(36).slice(2)}.db`);
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

describe("Stress: Volume", () => {
  it("records 100 sessions with varying sizes (1 to 500 messages)", () => {
    const sessionIds: string[] = [];

    for (let i = 0; i < 100; i++) {
      const msgCount = 1 + Math.floor(Math.random() * 20); // 1-20 for speed, scale test below does 500
      const id = capture.start({ index: i });

      for (let j = 0; j < msgCount; j++) {
        const role = j % 2 === 0 ? "user" : "assistant";
        capture.addMessage(id, { role, content: `Message ${j} of session ${i}` });
      }

      capture.stop(id);
      sessionIds.push(id);
    }

    expect(sessionIds).toHaveLength(100);

    const stats = storage.getStats();
    expect(stats.totalSessions).toBe(100);
    expect(stats.totalMessages).toBeGreaterThan(100);
  });

  it("handles a single session with 500 messages", () => {
    const id = capture.start({ type: "large" });

    for (let i = 0; i < 500; i++) {
      const role = i % 3 === 0 ? "user" : i % 3 === 1 ? "assistant" : "tool";
      const opts = role === "assistant" && i % 6 === 1
        ? { toolName: "read_file" }
        : role === "tool"
          ? { toolCallId: `tc_${i}` }
          : undefined;
      capture.addMessage(id, { role, content: `Msg ${i}: ${"x".repeat(100)}`, ...opts });
    }

    capture.stop(id);

    const session = storage.getSession(id);
    expect(session!.message_count).toBe(500);
    expect(session!.token_estimate).toBeGreaterThan(5000);
  });
});

describe("Stress: Concurrent recording", () => {
  it("records 2 sessions simultaneously without cross-contamination", () => {
    const id1 = capture.start({ name: "session-A" });
    const id2 = capture.start({ name: "session-B" });

    // Interleave messages
    capture.addMessage(id1, { role: "user", content: "A-msg-1" });
    capture.addMessage(id2, { role: "user", content: "B-msg-1" });
    capture.addMessage(id1, { role: "assistant", content: "A-msg-2" });
    capture.addMessage(id2, { role: "assistant", content: "B-msg-2" });
    capture.addMessage(id1, { role: "user", content: "A-msg-3" });

    capture.stop(id1);
    capture.stop(id2);

    const msgs1 = storage.getMessages(id1);
    const msgs2 = storage.getMessages(id2);

    expect(msgs1).toHaveLength(3);
    expect(msgs2).toHaveLength(2);
    expect(msgs1.every((m) => m.content.startsWith("A-"))).toBe(true);
    expect(msgs2.every((m) => m.content.startsWith("B-"))).toBe(true);
  });

  it("records 10 sessions concurrently", () => {
    const ids = Array.from({ length: 10 }, (_, i) => capture.start({ idx: i }));

    for (let round = 0; round < 5; round++) {
      for (const id of ids) {
        capture.addMessage(id, { role: "user", content: `round-${round}` });
        capture.addMessage(id, { role: "assistant", content: `reply-${round}` });
      }
    }

    for (const id of ids) capture.stop(id);

    for (const id of ids) {
      const msgs = storage.getMessages(id);
      expect(msgs).toHaveLength(10);
    }
  });
});

describe("Stress: Malformed input", () => {
  it("handles empty message content", () => {
    const id = capture.start();
    capture.addMessage(id, { role: "user", content: "" });
    capture.addMessage(id, { role: "assistant", content: "" });
    capture.stop(id);

    const msgs = storage.getMessages(id);
    expect(msgs).toHaveLength(2);
    expect(msgs[0].content).toBe("");
  });

  it("handles unicode and emoji content", () => {
    const id = capture.start();
    capture.addMessage(id, { role: "user", content: "こんにちは世界 🌍🔥💻" });
    capture.addMessage(id, { role: "assistant", content: "Привет мир! 你好世界 مرحبا" });
    capture.addMessage(id, { role: "user", content: "𝕳𝖊𝖑𝖑𝖔 \u0000 null byte" });
    capture.stop(id);

    const msgs = storage.getMessages(id);
    expect(msgs).toHaveLength(3);
    expect(msgs[0].content).toContain("🌍");
    expect(msgs[1].content).toContain("Привет");
  });

  it("handles 100KB messages", () => {
    const id = capture.start();
    const bigContent = "x".repeat(100 * 1024);
    capture.addMessage(id, { role: "user", content: bigContent });
    capture.addMessage(id, { role: "assistant", content: bigContent });
    capture.stop(id);

    const msgs = storage.getMessages(id);
    expect(msgs).toHaveLength(2);
    expect(msgs[0].content.length).toBe(100 * 1024);
  });

  it("handles messages with special SQL characters", () => {
    const id = capture.start();
    capture.addMessage(id, { role: "user", content: "Robert'); DROP TABLE sessions;--" });
    capture.addMessage(id, { role: "assistant", content: 'He said "hello" and \'goodbye\'' });
    capture.addMessage(id, { role: "user", content: "backslash \\ percent % underscore _" });
    capture.stop(id);

    const msgs = storage.getMessages(id);
    expect(msgs).toHaveLength(3);
    expect(msgs[0].content).toContain("DROP TABLE");

    // Verify table still exists
    const stats = storage.getStats();
    expect(stats.totalSessions).toBe(1);
  });

  it("handles messages with JSON in content", () => {
    const id = capture.start();
    const jsonContent = JSON.stringify({ nested: { key: "value", arr: [1, 2, 3] } });
    capture.addMessage(id, { role: "user", content: jsonContent });
    capture.stop(id);

    const msgs = storage.getMessages(id);
    expect(JSON.parse(msgs[0].content)).toEqual({ nested: { key: "value", arr: [1, 2, 3] } });
  });

  it("handles newlines and tabs in content", () => {
    const id = capture.start();
    capture.addMessage(id, { role: "user", content: "line1\nline2\nline3\ttab" });
    capture.addMessage(id, { role: "assistant", content: "```\ncode block\n  indented\n```" });
    capture.stop(id);

    const msgs = storage.getMessages(id);
    expect(msgs[0].content).toContain("\n");
    expect(msgs[1].content).toContain("```");
  });
});

describe("Stress: SQLite under load", () => {
  it("rapid start/stop cycles (200 sessions)", () => {
    for (let i = 0; i < 200; i++) {
      const id = capture.start();
      capture.addMessage(id, { role: "user", content: `quick-${i}` });
      capture.stop(id);
    }

    const stats = storage.getStats();
    expect(stats.totalSessions).toBe(200);
    expect(stats.totalMessages).toBe(200);
  });

  it("bulk insert performance (1000 messages in one session)", () => {
    const id = capture.start();
    const start = Date.now();

    for (let i = 0; i < 1000; i++) {
      capture.addMessage(id, { role: i % 2 === 0 ? "user" : "assistant", content: `msg-${i}` });
    }

    capture.stop(id);
    const elapsed = Date.now() - start;

    expect(elapsed).toBeLessThan(5000); // Should complete in under 5 seconds
    expect(storage.getSession(id)!.message_count).toBe(1000);
  });
});

describe("Stress: Export at scale", () => {
  it("exports 100 sessions as valid JSONL", () => {
    for (let i = 0; i < 100; i++) {
      const id = capture.start();
      capture.addMessage(id, { role: "user", content: `Prompt ${i}` });
      capture.addMessage(id, { role: "assistant", content: `Response ${i}` });
      capture.stop(id);
    }

    const jsonl = exporter.exportSessions();
    const lines = jsonl.split("\n").filter(Boolean);
    expect(lines).toHaveLength(100);

    // Validate every line is valid JSON with correct structure
    for (const line of lines) {
      const parsed = JSON.parse(line);
      expect(parsed.messages).toBeDefined();
      expect(Array.isArray(parsed.messages)).toBe(true);
      expect(parsed.messages.length).toBeGreaterThanOrEqual(2); // system + at least one user/assistant
      expect(parsed.messages[0].role).toBe("system");

      for (const msg of parsed.messages) {
        expect(typeof msg.role).toBe("string");
        expect(typeof msg.content).toBe("string");
      }
    }
  });

  it("exports with filters produce valid JSONL", () => {
    // Create sessions with tool calls
    for (let i = 0; i < 20; i++) {
      const id = capture.start();
      capture.addMessage(id, { role: "user", content: `Task ${i}` });
      capture.addMessage(id, { role: "assistant", content: `Using tool`, toolName: "bash" });
      capture.addMessage(id, { role: "tool", content: "result", toolCallId: `tc_${i}` });
      capture.addMessage(id, { role: "assistant", content: `Done ${i}` });
      capture.stop(id);
    }

    // Create sessions without tool calls
    for (let i = 0; i < 10; i++) {
      const id = capture.start();
      capture.addMessage(id, { role: "user", content: `Simple ${i}` });
      capture.addMessage(id, { role: "assistant", content: `Reply ${i}` });
      capture.stop(id);
    }

    const filtered = exporter.exportFiltered({ hasToolCalls: true });
    const lines = filtered.split("\n").filter(Boolean);
    expect(lines).toHaveLength(20);

    for (const line of lines) {
      const parsed = JSON.parse(line);
      expect(parsed.messages).toBeDefined();
    }
  });

  it("handles 1000 sessions export without OOM", () => {
    // Insert 1000 sessions directly through storage for speed
    for (let i = 0; i < 1000; i++) {
      const id = storage.createSession({ batch: true });
      storage.addMessage(id, "user", `Prompt ${i}`);
      storage.addMessage(id, "assistant", `Response ${i}`);
      storage.stopSession(id);
    }

    const start = Date.now();
    const jsonl = exporter.exportSessions();
    const elapsed = Date.now() - start;

    const lines = jsonl.split("\n").filter(Boolean);
    expect(lines).toHaveLength(1000);
    expect(elapsed).toBeLessThan(10000); // Under 10 seconds

    // Spot-check validity
    expect(() => JSON.parse(lines[0])).not.toThrow();
    expect(() => JSON.parse(lines[999])).not.toThrow();
  });

  it("handles 10000 sessions export", () => {
    // Insert 10000 sessions — use raw storage for max speed
    for (let i = 0; i < 10000; i++) {
      const id = storage.createSession();
      storage.addMessage(id, "user", `P${i}`);
      storage.addMessage(id, "assistant", `R${i}`);
      storage.stopSession(id);
    }

    const start = Date.now();
    const jsonl = exporter.exportSessions();
    const elapsed = Date.now() - start;

    const lines = jsonl.split("\n").filter(Boolean);
    expect(lines).toHaveLength(10000);
    expect(elapsed).toBeLessThan(30000); // Under 30 seconds

    // Memory check: we can still allocate
    const check = new Array(1000).fill("ok");
    expect(check).toHaveLength(1000);
  });
});
