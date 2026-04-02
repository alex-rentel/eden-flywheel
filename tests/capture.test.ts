import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { Storage } from "../src/storage.js";
import { SessionCapture } from "../src/capture.js";
import fs from "fs";
import path from "path";
import os from "os";

let storage: Storage;
let capture: SessionCapture;
let dbPath: string;

beforeEach(() => {
  dbPath = path.join(os.tmpdir(), `flywheel-capture-test-${Date.now()}.db`);
  storage = new Storage(dbPath);
  capture = new SessionCapture(storage);
});

afterEach(() => {
  storage.close();
  try { fs.unlinkSync(dbPath); } catch {}
  try { fs.unlinkSync(dbPath + "-wal"); } catch {}
  try { fs.unlinkSync(dbPath + "-shm"); } catch {}
});

describe("SessionCapture", () => {
  it("starts and stops a session", () => {
    const id = capture.start({ project: "test" });
    expect(capture.isActive(id)).toBe(true);

    capture.addMessage(id, { role: "user", content: "Hello" });
    capture.addMessage(id, { role: "assistant", content: "Hi" });

    const result = capture.stop(id);
    expect(result.messageCount).toBe(2);
    expect(result.tokenEstimate).toBeGreaterThan(0);
    expect(capture.isActive(id)).toBe(false);
  });

  it("tracks active sessions", () => {
    const id1 = capture.start();
    const id2 = capture.start();
    expect(capture.getActiveSessions()).toHaveLength(2);

    capture.stop(id1);
    expect(capture.getActiveSessions()).toHaveLength(1);
    expect(capture.getActiveSessions()[0]).toBe(id2);
  });

  it("throws when stopping an inactive session", () => {
    expect(() => capture.stop("nonexistent")).toThrow();
  });

  it("throws when adding message to inactive session", () => {
    expect(() =>
      capture.addMessage("nonexistent", { role: "user", content: "test" })
    ).toThrow();
  });
});
