import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { FlywheelError, SessionNotFoundError, SessionNotActiveError, ExportError, TrainingError, StorageError, ConfigError } from "../src/errors.js";
import { logger, setLogLevel, getLogLevel } from "../src/logger.js";
import { parseCliArgs, resolveConfig, saveConfig, loadConfig, getConfigPath, resetConfigCache, type FlywheelConfig } from "../src/config.js";
import fs from "fs";
import path from "path";
import os from "os";

// ── Typed Errors ────────────────────────────────────────────────

describe("Typed errors", () => {
  it("FlywheelError has code and message", () => {
    const err = new FlywheelError("test error", "TEST_CODE");
    expect(err.message).toBe("test error");
    expect(err.code).toBe("TEST_CODE");
    expect(err.name).toBe("FlywheelError");
    expect(err instanceof Error).toBe(true);
  });

  it("SessionNotFoundError", () => {
    const err = new SessionNotFoundError("abc-123");
    expect(err.message).toContain("abc-123");
    expect(err.code).toBe("SESSION_NOT_FOUND");
    expect(err instanceof FlywheelError).toBe(true);
  });

  it("SessionNotActiveError", () => {
    const err = new SessionNotActiveError("xyz-789");
    expect(err.message).toContain("xyz-789");
    expect(err.code).toBe("SESSION_NOT_ACTIVE");
  });

  it("ExportError", () => {
    const err = new ExportError("bad format");
    expect(err.code).toBe("EXPORT_ERROR");
  });

  it("TrainingError", () => {
    const err = new TrainingError("mlx failed");
    expect(err.code).toBe("TRAINING_ERROR");
  });

  it("StorageError", () => {
    const err = new StorageError("db locked");
    expect(err.code).toBe("STORAGE_ERROR");
  });

  it("ConfigError", () => {
    const err = new ConfigError("invalid format");
    expect(err.code).toBe("CONFIG_ERROR");
  });
});

// ── Logger ──────────────────────────────────────────────────────

describe("Logger", () => {
  let stderrOutput: string[];
  const originalWrite = process.stderr.write;

  beforeEach(() => {
    stderrOutput = [];
    process.stderr.write = ((chunk: string | Uint8Array) => {
      stderrOutput.push(typeof chunk === "string" ? chunk : chunk.toString());
      return true;
    }) as typeof process.stderr.write;
    setLogLevel("debug");
  });

  afterEach(() => {
    process.stderr.write = originalWrite;
    setLogLevel("info");
  });

  it("logs structured JSON to stderr", () => {
    logger.info("test message", { key: "value" });
    expect(stderrOutput).toHaveLength(1);
    const parsed = JSON.parse(stderrOutput[0]);
    expect(parsed.level).toBe("info");
    expect(parsed.message).toBe("test message");
    expect(parsed.key).toBe("value");
    expect(parsed.timestamp).toBeTruthy();
  });

  it("respects log level filtering", () => {
    setLogLevel("warn");
    logger.debug("should not appear");
    logger.info("should not appear");
    logger.warn("should appear");
    logger.error("should appear");
    expect(stderrOutput).toHaveLength(2);
  });

  it("logs at all levels", () => {
    setLogLevel("debug");
    logger.debug("d");
    logger.info("i");
    logger.warn("w");
    logger.error("e");
    expect(stderrOutput).toHaveLength(4);
  });

  it("getLogLevel returns current level", () => {
    setLogLevel("error");
    expect(getLogLevel()).toBe("error");
  });
});

// ── Config ──────────────────────────────────────────────────────

describe("Config", () => {
  let tmpDir: string;
  const envHome = process.env.HOME;

  beforeEach(() => {
    tmpDir = path.join(os.tmpdir(), `flywheel-config-${Date.now()}-${Math.random().toString(36).slice(2)}`);
    fs.mkdirSync(tmpDir, { recursive: true });
    resetConfigCache();
  });

  afterEach(() => {
    fs.rmSync(tmpDir, { recursive: true, force: true });
    resetConfigCache();
  });

  it("parseCliArgs handles --verbose", () => {
    const result = parseCliArgs(["--verbose"]);
    expect(result.logLevel).toBe("debug");
  });

  it("parseCliArgs handles --quiet", () => {
    const result = parseCliArgs(["--quiet"]);
    expect(result.logLevel).toBe("error");
  });

  it("parseCliArgs handles --log-level", () => {
    const result = parseCliArgs(["--log-level", "warn"]);
    expect(result.logLevel).toBe("warn");
  });

  it("parseCliArgs handles --db-path", () => {
    const result = parseCliArgs(["--db-path", "/tmp/test.db"]);
    expect(result.dbPath).toBe("/tmp/test.db");
  });

  it("parseCliArgs handles no args", () => {
    const result = parseCliArgs([]);
    expect(Object.keys(result)).toHaveLength(0);
  });

  it("resolveConfig merges file config with CLI overrides", () => {
    const config = resolveConfig({ logLevel: "debug" });
    expect(config.logLevel).toBe("debug");
  });

  it("getConfigPath returns expected path", () => {
    const p = getConfigPath();
    expect(p).toContain("config.json");
    expect(p).toContain(".eden-flywheel");
  });
});

// ── Integration: typed errors in capture ────────────────────────

describe("Typed errors in capture", () => {
  it("stop throws SessionNotActiveError", async () => {
    const { Storage } = await import("../src/storage.js");
    const { SessionCapture } = await import("../src/capture.js");
    const { SessionNotActiveError } = await import("../src/errors.js");

    const dbPath = path.join(os.tmpdir(), `flywheel-err-${Date.now()}.db`);
    const storage = new Storage(dbPath);
    const capture = new SessionCapture(storage);

    try {
      capture.stop("nonexistent");
      expect.unreachable("Should have thrown");
    } catch (err) {
      expect(err instanceof SessionNotActiveError).toBe(true);
      expect((err as SessionNotActiveError).code).toBe("SESSION_NOT_ACTIVE");
    }

    storage.close();
    for (const ext of ["", "-wal", "-shm"]) {
      try { fs.unlinkSync(dbPath + ext); } catch {}
    }
  });

  it("addMessage throws SessionNotActiveError", async () => {
    const { Storage } = await import("../src/storage.js");
    const { SessionCapture } = await import("../src/capture.js");
    const { SessionNotActiveError } = await import("../src/errors.js");

    const dbPath = path.join(os.tmpdir(), `flywheel-err2-${Date.now()}.db`);
    const storage = new Storage(dbPath);
    const capture = new SessionCapture(storage);

    try {
      capture.addMessage("nonexistent", { role: "user", content: "test" });
      expect.unreachable("Should have thrown");
    } catch (err) {
      expect(err instanceof SessionNotActiveError).toBe(true);
    }

    storage.close();
    for (const ext of ["", "-wal", "-shm"]) {
      try { fs.unlinkSync(dbPath + ext); } catch {}
    }
  });
});
