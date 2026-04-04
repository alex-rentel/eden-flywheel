import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { Storage } from "../src/storage.js";
import { Exporter } from "../src/export.js";
import fs from "fs";
import path from "path";
import os from "os";

let storage: Storage;
let exporter: Exporter;
let dbPath: string;

beforeEach(() => {
  dbPath = path.join(os.tmpdir(), `flywheel-export-test-${Date.now()}.db`);
  storage = new Storage(dbPath);
  exporter = new Exporter(storage);
});

afterEach(() => {
  storage.close();
  try { fs.unlinkSync(dbPath); } catch {}
  try { fs.unlinkSync(dbPath + "-wal"); } catch {}
  try { fs.unlinkSync(dbPath + "-shm"); } catch {}
});

function createSampleSession(): string {
  const id = storage.createSession({ project: "test" });
  storage.addMessage(id, "user", "Fix the bug in auth.ts");
  storage.addMessage(id, "assistant", "I'll read the file first", { toolName: "read_file" });
  storage.addMessage(id, "tool", "file contents here", { toolCallId: "tc_1" });
  storage.addMessage(id, "assistant", "I found the issue and fixed it.");
  storage.stopSession(id);
  return id;
}

describe("Exporter", () => {
  it("exports sessions as JSONL", () => {
    const id = createSampleSession();
    const jsonl = exporter.exportSessions([id]);
    const lines = jsonl.split("\n").filter(Boolean);
    expect(lines).toHaveLength(1);

    const example = JSON.parse(lines[0]);
    expect(example.messages).toBeDefined();
    expect(example.messages[0].role).toBe("system");
    expect(example.messages.length).toBeGreaterThanOrEqual(3);
  });

  it("exports all completed sessions when no IDs given", () => {
    createSampleSession();
    createSampleSession();

    // Create an active (not stopped) session — should not be exported
    const activeId = storage.createSession();
    storage.addMessage(activeId, "user", "test");

    const jsonl = exporter.exportSessions();
    const lines = jsonl.split("\n").filter(Boolean);
    expect(lines).toHaveLength(2);
  });

  it("formats tool calls correctly", () => {
    const id = createSampleSession();
    const jsonl = exporter.exportSessions([id]);
    const example = JSON.parse(jsonl.split("\n")[0]);

    // Find the assistant message with tool call
    const toolCallMsg = example.messages.find(
      (m: { role: string; content: string }) =>
        m.role === "assistant" && m.content.includes("<tool_call>")
    );
    expect(toolCallMsg).toBeDefined();

    // Find the tool result
    const toolResult = example.messages.find(
      (m: { role: string }) => m.role === "tool"
    );
    expect(toolResult).toBeDefined();
  });

  it("filters by quality", () => {
    // Good session with tool calls, no errors
    const goodId = storage.createSession();
    storage.addMessage(goodId, "user", "test");
    storage.addMessage(goodId, "assistant", "result", { toolName: "bash" });
    storage.stopSession(goodId);

    // Bad session with errors
    const badId = storage.createSession();
    storage.addMessage(badId, "user", "test");
    storage.addMessage(badId, "assistant", "error: something broke");
    storage.stopSession(badId);

    const filtered = exporter.exportFiltered({ hasToolCalls: true, noErrors: true });
    const lines = filtered.split("\n").filter(Boolean);
    expect(lines).toHaveLength(1);
  });

  it("returns empty string when no sessions match", () => {
    const jsonl = exporter.exportSessions();
    expect(jsonl).toBe("");
  });

  it("exports with eval split", () => {
    // Create enough sessions for a meaningful split
    for (let i = 0; i < 10; i++) {
      createSampleSession();
    }

    const { train, eval: evalData } = exporter.exportWithEvalSplit({ evalSplitPercent: 10 });
    const trainLines = train.split("\n").filter(Boolean);
    const evalLines = evalData.split("\n").filter(Boolean);

    expect(evalLines.length).toBeGreaterThanOrEqual(1);
    expect(trainLines.length).toBeGreaterThanOrEqual(1);
    expect(trainLines.length + evalLines.length).toBe(10);
  });

  it("eval split with 1 session still produces eval set", () => {
    createSampleSession();
    const { train, eval: evalData } = exporter.exportWithEvalSplit({ evalSplitPercent: 10 });

    // With 1 session, eval gets 1 and train gets 0
    const evalLines = evalData.split("\n").filter(Boolean);
    expect(evalLines.length).toBe(1);
  });
});
