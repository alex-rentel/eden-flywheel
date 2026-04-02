/**
 * Additional tests to boost coverage on uncovered branches.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { Storage } from "../src/storage.js";
import { SessionCapture } from "../src/capture.js";
import { Exporter } from "../src/export.js";
import { scoreSession, computeDataStats, sessionFingerprint } from "../src/quality.js";
import { estimateTokens } from "../src/tokens.js";
import { saveConfig, loadConfig, resetConfigCache, getConfigPath } from "../src/config.js";
import { setLogLevel } from "../src/logger.js";
import { getTrainingHistory, getActiveAdapter } from "../src/training.js";
import fs from "fs";
import path from "path";
import os from "os";

let storage: Storage;
let capture: SessionCapture;
let exporter: Exporter;
let dbPath: string;

beforeEach(() => {
  setLogLevel("error");
  dbPath = path.join(os.tmpdir(), `flywheel-cov-${Date.now()}-${Math.random().toString(36).slice(2)}.db`);
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

describe("Export coverage: edge cases", () => {
  it("exports with sessionIds that don't exist", () => {
    const jsonl = exporter.exportWithOptions({ sessionIds: ["nonexistent-id"] });
    expect(jsonl).toBe("");
  });

  it("exports alpaca with tool-only session (no plain assistant)", () => {
    const id = storage.createSession();
    storage.addMessage(id, "user", "test");
    // Only tool-call assistant messages, no plain one
    storage.addMessage(id, "assistant", '{"path":"test.ts"}', { toolName: "read_file" });
    storage.stopSession(id);

    const jsonl = exporter.exportWithOptions({ sessionIds: [id], format: "alpaca" });
    // Alpaca needs a plain assistant message for output
    expect(jsonl).toBe("");
  });

  it("exports sharegpt with minimal session", () => {
    const id = storage.createSession();
    storage.addMessage(id, "user", "hello");
    storage.addMessage(id, "assistant", "hi");
    storage.stopSession(id);

    const jsonl = exporter.exportWithOptions({ sessionIds: [id], format: "sharegpt" });
    const parsed = JSON.parse(jsonl);
    expect(parsed.conversations[0].from).toBe("human");
    expect(parsed.conversations[1].from).toBe("gpt");
  });

  it("exports raw format with tool messages", () => {
    const id = storage.createSession();
    storage.addMessage(id, "user", "test");
    storage.addMessage(id, "assistant", "args", { toolName: "bash" });
    storage.addMessage(id, "tool", "result", { toolCallId: "tc_1" });
    storage.addMessage(id, "assistant", "done");
    storage.stopSession(id);

    const jsonl = exporter.exportWithOptions({ sessionIds: [id], format: "raw" });
    const parsed = JSON.parse(jsonl);
    expect(parsed.messages.some((m: { content: string }) => m.content.includes("<tool_call>"))).toBe(true);
  });

  it("validates unrecognized format", () => {
    const errors = exporter.validateExport('{"unknown": true}');
    expect(errors.some((e) => e.includes("unrecognized"))).toBe(true);
  });

  it("validates messages with non-string role", () => {
    const errors = exporter.validateExport('{"messages": [{"role": 123, "content": "test"}]}');
    expect(errors.some((e) => e.includes("invalid role"))).toBe(true);
  });

  it("validates messages with non-string content", () => {
    const errors = exporter.validateExport('{"messages": [{"role": "user", "content": 123}]}');
    expect(errors.some((e) => e.includes("invalid content"))).toBe(true);
  });

  it("validates messages not array", () => {
    const errors = exporter.validateExport('{"messages": "not an array"}');
    expect(errors.some((e) => e.includes("not an array"))).toBe(true);
  });

  it("validates alpaca with non-string instruction", () => {
    const errors = exporter.validateExport('{"instruction": 123, "input": "", "output": "test"}');
    expect(errors.some((e) => e.includes("invalid instruction"))).toBe(true);
  });

  it("validates alpaca with non-string output", () => {
    const errors = exporter.validateExport('{"instruction": "test", "input": "", "output": 123}');
    expect(errors.some((e) => e.includes("invalid output"))).toBe(true);
  });

  it("validates conversations not array", () => {
    const errors = exporter.validateExport('{"conversations": "not array"}');
    expect(errors.some((e) => e.includes("not an array"))).toBe(true);
  });

  it("getSessionQuality returns null for nonexistent", () => {
    const q = exporter.getSessionQuality("nonexistent");
    expect(q).toBeNull();
  });

  it("getSessionQuality returns score for existing", () => {
    const id = storage.createSession();
    storage.addMessage(id, "user", "test");
    storage.addMessage(id, "assistant", "reply");
    storage.stopSession(id);

    const q = exporter.getSessionQuality(id);
    expect(q).not.toBeNull();
    expect(q!.score).toBeGreaterThan(0);
  });

  it("export session with empty messages returns nothing", () => {
    const id = storage.createSession();
    storage.stopSession(id);

    const jsonl = exporter.exportWithOptions({ sessionIds: [id] });
    expect(jsonl).toBe("");
  });
});

describe("Quality coverage: edge cases", () => {
  it("scores very long session negatively", () => {
    const id = storage.createSession();
    // Create a session with lots of tokens
    for (let i = 0; i < 10; i++) {
      storage.addMessage(id, "user", "x".repeat(4000));
      storage.addMessage(id, "assistant", "y".repeat(4000));
    }
    storage.stopSession(id);

    const session = storage.getSession(id)!;
    const messages = storage.getMessages(id);
    const quality = scoreSession(session, messages);
    expect(quality.reasons).toContain("very_long");
  });

  it("fingerprint handles empty messages", () => {
    const fp = sessionFingerprint([]);
    expect(fp).toBeTruthy();
    expect(fp.length).toBe(16);
  });

  it("computeDataStats handles sessions in all turn buckets", () => {
    // Create sessions of varying sizes
    const sizes = [1, 5, 15, 30, 60];
    const items = sizes.map((n) => {
      const id = storage.createSession();
      for (let i = 0; i < n; i++) {
        storage.addMessage(id, i % 2 === 0 ? "user" : "assistant", `msg-${i}`);
      }
      storage.stopSession(id);
      const session = storage.getSession(id)!;
      const messages = storage.getMessages(id);
      return { session, messages, quality: scoreSession(session, messages) };
    });

    const stats = computeDataStats(items);
    expect(stats.turnHistogram["1-3"]).toBeGreaterThan(0);
    expect(stats.turnHistogram["4-10"]).toBeGreaterThan(0);
    expect(stats.turnHistogram["11-25"]).toBeGreaterThan(0);
    expect(stats.turnHistogram["26-50"]).toBeGreaterThan(0);
    expect(stats.turnHistogram["50+"]).toBeGreaterThan(0);
  });
});

describe("Token estimation: edge branches", () => {
  it("handles very short text", () => {
    expect(estimateTokens("a")).toBe(1);
  });

  it("handles whitespace-heavy text", () => {
    const tokens = estimateTokens("   \n\n\t\t   hello   \n\n   world   ");
    expect(tokens).toBeGreaterThan(0);
  });
});

describe("Config: getConfigPath", () => {
  it("returns a path containing config.json", () => {
    const p = getConfigPath();
    expect(p).toContain("config.json");
  });

  it("resetConfigCache allows re-loading", () => {
    resetConfigCache();
    // loadConfig with no file returns empty config
    const config = loadConfig();
    expect(config).toBeDefined();
  });
});

describe("Storage: default path and edge cases", () => {
  it("handles getQualitySessions with minMessages", () => {
    const id1 = storage.createSession();
    storage.addMessage(id1, "user", "a");
    storage.stopSession(id1);

    const id2 = storage.createSession();
    for (let i = 0; i < 5; i++) storage.addMessage(id2, "user", `msg-${i}`);
    storage.stopSession(id2);

    const filtered = storage.getQualitySessions({ minMessages: 3 });
    expect(filtered).toHaveLength(1);
    expect(filtered[0].id).toBe(id2);
  });

  it("updateSessionQuality updates correctly", () => {
    const id = storage.createSession();
    storage.addMessage(id, "user", "test");
    storage.stopSession(id);

    storage.updateSessionQuality(id, 0.85, "abc123");
    const session = storage.getSession(id)!;
    expect(session.quality_score).toBe(0.85);
    expect(session.fingerprint).toBe("abc123");
  });
});

describe("Training: history and active adapter", () => {
  it("getTrainingHistory returns array", () => {
    const history = getTrainingHistory();
    expect(Array.isArray(history)).toBe(true);
  });

  it("getActiveAdapter returns null or a valid path", () => {
    const active = getActiveAdapter();
    // In test environment without a promoted adapter, this should be null
    // (unless a previous test promoted one to ~/.eden-models/active/)
    if (active !== null) {
      expect(typeof active).toBe("string");
      expect(fs.existsSync(active)).toBe(true);
    } else {
      expect(active).toBeNull();
    }
  });

  it("promoteAdapter copies files correctly", async () => {
    const { promoteAdapter } = await import("../src/training.js");
    const tmpAdapter = path.join(os.tmpdir(), `flywheel-promote-${Date.now()}`);
    fs.mkdirSync(tmpAdapter, { recursive: true });
    fs.writeFileSync(path.join(tmpAdapter, "adapters.safetensors"), "weights");
    fs.writeFileSync(path.join(tmpAdapter, "config.json"), '{}');

    const promoted = promoteAdapter(tmpAdapter, `test-${Date.now()}`);
    expect(fs.existsSync(path.join(promoted, "adapters.safetensors"))).toBe(true);
    expect(fs.existsSync(path.join(promoted, "config.json"))).toBe(true);

    fs.rmSync(tmpAdapter, { recursive: true, force: true });
    fs.rmSync(promoted, { recursive: true, force: true });
  });

  it("trainAdapter with existing but empty data file", async () => {
    const { trainAdapter } = await import("../src/training.js");
    const tmpData = path.join(os.tmpdir(), `flywheel-traindata-${Date.now()}.jsonl`);
    fs.writeFileSync(tmpData, ""); // empty file

    const result = await trainAdapter({
      baseModel: "nonexistent-model",
      trainData: tmpData,
    });

    // Will fail because python3/mlx-lm isn't available, but it should reach the execFile call
    expect(result.error).toBeTruthy();

    fs.unlinkSync(tmpData);
  });
});
