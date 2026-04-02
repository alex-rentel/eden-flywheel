import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { Storage } from "../src/storage.js";
import fs from "fs";
import path from "path";
import os from "os";

let storage: Storage;
let dbPath: string;

beforeEach(() => {
  dbPath = path.join(os.tmpdir(), `flywheel-test-${Date.now()}.db`);
  storage = new Storage(dbPath);
});

afterEach(() => {
  storage.close();
  try { fs.unlinkSync(dbPath); } catch {}
  try { fs.unlinkSync(dbPath + "-wal"); } catch {}
  try { fs.unlinkSync(dbPath + "-shm"); } catch {}
});

describe("Storage", () => {
  it("creates and retrieves a session", () => {
    const id = storage.createSession({ project: "test" });
    const session = storage.getSession(id);
    expect(session).toBeDefined();
    expect(session!.id).toBe(id);
    expect(session!.message_count).toBe(0);
    expect(session!.stopped_at).toBeNull();
  });

  it("stops a session", () => {
    const id = storage.createSession();
    storage.stopSession(id);
    const session = storage.getSession(id);
    expect(session!.stopped_at).not.toBeNull();
  });

  it("adds messages and updates counters", () => {
    const id = storage.createSession();
    storage.addMessage(id, "user", "Hello world");
    storage.addMessage(id, "assistant", "Hi there", { toolName: "read_file" });

    const session = storage.getSession(id);
    expect(session!.message_count).toBe(2);
    expect(session!.token_estimate).toBeGreaterThan(0);
    expect(session!.has_tool_calls).toBe(1);
  });

  it("retrieves messages in order", () => {
    const id = storage.createSession();
    storage.addMessage(id, "user", "First");
    storage.addMessage(id, "assistant", "Second");
    storage.addMessage(id, "user", "Third");

    const messages = storage.getMessages(id);
    expect(messages).toHaveLength(3);
    expect(messages[0].role).toBe("user");
    expect(messages[0].content).toBe("First");
    expect(messages[2].content).toBe("Third");
  });

  it("lists sessions in descending order", () => {
    storage.createSession({ order: 1 });
    storage.createSession({ order: 2 });
    const sessions = storage.listSessions();
    expect(sessions).toHaveLength(2);
  });

  it("returns correct stats", () => {
    const id1 = storage.createSession();
    storage.addMessage(id1, "user", "Hello");
    storage.addMessage(id1, "tool", "error: file not found");

    const id2 = storage.createSession();
    storage.addMessage(id2, "user", "Test");
    storage.addMessage(id2, "assistant", "Result", { toolName: "bash" });

    const stats = storage.getStats();
    expect(stats.totalSessions).toBe(2);
    expect(stats.totalMessages).toBe(4);
    expect(stats.sessionsWithToolCalls).toBe(1);
    expect(stats.sessionsWithErrors).toBe(1);
  });

  it("filters quality sessions", () => {
    const id1 = storage.createSession();
    storage.addMessage(id1, "user", "test");
    storage.addMessage(id1, "assistant", "ok", { toolName: "read" });
    storage.stopSession(id1);

    const id2 = storage.createSession();
    storage.addMessage(id2, "user", "test");
    storage.addMessage(id2, "tool", "command failed");
    storage.stopSession(id2);

    const withTools = storage.getQualitySessions({ hasToolCalls: true });
    expect(withTools).toHaveLength(1);
    expect(withTools[0].id).toBe(id1);

    const noErrors = storage.getQualitySessions({ noErrors: true });
    expect(noErrors).toHaveLength(1);
    expect(noErrors[0].id).toBe(id1);
  });

  it("does not flag 'error handling' in assistant messages as errors", () => {
    const id = storage.createSession();
    storage.addMessage(id, "assistant", "error handling is important");
    storage.addMessage(id, "assistant", "no error found in the code");
    storage.addMessage(id, "user", "what about error boundaries?");
    storage.stopSession(id);

    const session = storage.getSession(id)!;
    expect(session.has_errors).toBe(0);
  });

  it("only flags tool results with actual error patterns", () => {
    const id = storage.createSession();
    storage.addMessage(id, "tool", "error: ENOENT no such file");
    const session = storage.getSession(id)!;
    expect(session.has_errors).toBe(1);
  });

  it("flags tool results with 'failed' as errors", () => {
    const id = storage.createSession();
    storage.addMessage(id, "tool", "command failed with exit code 1");
    const session = storage.getSession(id)!;
    expect(session.has_errors).toBe(1);
  });

  it("flags tool results with 'exception:' as errors", () => {
    const id = storage.createSession();
    storage.addMessage(id, "tool", "exception: null pointer");
    const session = storage.getSession(id)!;
    expect(session.has_errors).toBe(1);
  });

  it("does not flag tool message with bare 'error' (no colon)", () => {
    const id = storage.createSession();
    storage.addMessage(id, "tool", "there was an error in the file");
    const session = storage.getSession(id)!;
    expect(session.has_errors).toBe(0);
  });
});
