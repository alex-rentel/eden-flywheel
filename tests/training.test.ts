import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { promoteAdapter, getTrainingHistory, getActiveAdapter, setTrainingStorage, isOllamaRunning } from "../src/training.js";
import { Storage } from "../src/storage.js";
import fs from "fs";
import path from "path";
import os from "os";

let tmpDir: string;

beforeEach(() => {
  tmpDir = path.join(os.tmpdir(), `flywheel-train-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  fs.mkdirSync(tmpDir, { recursive: true });
});

afterEach(() => {
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

describe("Training: promote adapter", () => {
  it("copies adapter files to target directory", async () => {
    const adapterDir = path.join(tmpDir, "adapter");
    fs.mkdirSync(adapterDir);
    fs.writeFileSync(path.join(adapterDir, "adapters.safetensors"), "fake-weights");
    fs.writeFileSync(path.join(adapterDir, "config.json"), '{"rank": 8}');

    const result = await promoteAdapter(adapterDir, `promote-test-${Date.now()}`);
    expect(fs.existsSync(path.join(result.promotedPath, "adapters.safetensors"))).toBe(true);
    expect(fs.existsSync(path.join(result.promotedPath, "config.json"))).toBe(true);
    expect(fs.readFileSync(path.join(result.promotedPath, "adapters.safetensors"), "utf-8")).toBe("fake-weights");
    expect(result.adapterFormat).toBe("mlx");

    fs.rmSync(result.promotedPath, { recursive: true, force: true });
  });

  it("throws when adapter path does not exist", async () => {
    await expect(promoteAdapter("/nonexistent/path")).rejects.toThrow();
  });

  it("copies subdirectories recursively", async () => {
    const adapterDir = path.join(tmpDir, "nested-adapter");
    fs.mkdirSync(adapterDir);
    fs.writeFileSync(path.join(adapterDir, "adapters.safetensors"), "weights");

    const subDir = path.join(adapterDir, "config");
    fs.mkdirSync(subDir);
    fs.writeFileSync(path.join(subDir, "tokenizer.json"), '{"type":"bpe"}');

    const result = await promoteAdapter(adapterDir, `nested-test-${Date.now()}`);
    expect(fs.existsSync(path.join(result.promotedPath, "adapters.safetensors"))).toBe(true);
    expect(fs.existsSync(path.join(result.promotedPath, "config", "tokenizer.json"))).toBe(true);
    expect(result.adapterFormat).toBe("mlx");

    fs.rmSync(result.promotedPath, { recursive: true, force: true });
  });

  it("detects GGUF adapter format", async () => {
    const adapterDir = path.join(tmpDir, "gguf-adapter");
    fs.mkdirSync(adapterDir);
    fs.writeFileSync(path.join(adapterDir, "adapter.gguf"), "fake-gguf");

    const result = await promoteAdapter(adapterDir, `gguf-test-${Date.now()}`, { deployOllama: false });
    expect(result.adapterFormat).toBe("gguf");
    expect(result.ollamaDeployed).toBe(false);

    fs.rmSync(result.promotedPath, { recursive: true, force: true });
  });

  it("adds note for MLX adapters about Ollama limitation", async () => {
    const adapterDir = path.join(tmpDir, "mlx-adapter");
    fs.mkdirSync(adapterDir);
    fs.writeFileSync(path.join(adapterDir, "adapters.safetensors"), "weights");

    const result = await promoteAdapter(adapterDir, `mlx-test-${Date.now()}`);
    expect(result.note).toContain("mlx-lm");

    fs.rmSync(result.promotedPath, { recursive: true, force: true });
  });
});

describe("Training: train result format", () => {
  it("trainAdapter returns correct structure for missing data", async () => {
    // Import dynamically to test with missing file
    const { trainAdapter } = await import("../src/training.js");
    const result = await trainAdapter({
      baseModel: "test-model",
      trainData: "/nonexistent/train.jsonl",
    });

    expect(result.error).toContain("not found");
    expect(result.baseModel).toBe("test-model");
    expect(result.adapterPath).toBeTruthy();
  });
});

describe("Training: evaluate adapter", () => {
  it("returns honest 'cannot evaluate' when no test data provided", async () => {
    const { evaluateAdapter } = await import("../src/training.js");

    // Create a fake adapter with some size
    const adapterDir = path.join(tmpDir, "eval-adapter");
    fs.mkdirSync(adapterDir);
    fs.writeFileSync(path.join(adapterDir, "adapters.safetensors"), "x".repeat(1024 * 1024)); // 1MB

    const result = await evaluateAdapter("test-model", adapterDir);
    expect(result.baseScore).toBe(0.5);
    expect(result.adaptedScore).toBe(0.5);
    expect(result.improved).toBe(false);
    expect(result.details).toContain("No test data available");
  });

  it("evaluates when explicit test data is provided", async () => {
    const { evaluateAdapter } = await import("../src/training.js");

    const adapterDir = path.join(tmpDir, "eval-adapter-with-test");
    fs.mkdirSync(adapterDir);
    fs.writeFileSync(path.join(adapterDir, "adapters.safetensors"), "weights");

    const testFile = path.join(tmpDir, "eval.jsonl");
    fs.writeFileSync(testFile, '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}\n');

    // Will fail because python3/mlx-lm isn't available, but should attempt eval with testData
    const result = await evaluateAdapter("test-model", adapterDir, testFile);
    // It reached the mlx_lm evaluate path (which errors), not the "no test data" path
    expect(result.details).not.toContain("No test data available");
    expect(result.testCases).toBe(1);
  });

  it("returns no improvement when adapter files missing", async () => {
    const { evaluateAdapter } = await import("../src/training.js");

    const emptyDir = path.join(tmpDir, "empty-adapter");
    fs.mkdirSync(emptyDir);

    const result = await evaluateAdapter("test-model", emptyDir);
    expect(result.improved).toBe(false);
    expect(result.details).toContain("No adapter files");
  });
});

describe("Training: MCP tools integration", () => {
  it("server exposes train, eval, promote tools", async () => {
    const { McpServer } = await import("@modelcontextprotocol/sdk/server/mcp.js");
    const { InMemoryTransport } = await import("@modelcontextprotocol/sdk/inMemory.js");
    const { Client } = await import("@modelcontextprotocol/sdk/client/index.js");
    const { z } = await import("zod");
    const { Storage } = await import("../src/storage.js");
    const { trainAdapter, evaluateAdapter, promoteAdapter } = await import("../src/training.js");

    const dbPath = path.join(tmpDir, "test.db");
    const storage = new Storage(dbPath);

    const server = new McpServer({ name: "test", version: "1.0.0" });

    // Register the training tools
    server.tool("flywheel_train", "Train", {
      baseModel: z.string(),
      trainData: z.string(),
    }, async ({ baseModel, trainData }) => {
      const result = await trainAdapter({ baseModel, trainData });
      return { content: [{ type: "text" as const, text: JSON.stringify(result) }] };
    });

    server.tool("flywheel_eval", "Eval", {
      baseModel: z.string(),
      adapterPath: z.string(),
    }, async ({ baseModel, adapterPath }) => {
      const result = await evaluateAdapter(baseModel, adapterPath);
      return { content: [{ type: "text" as const, text: JSON.stringify(result) }] };
    });

    server.tool("flywheel_promote", "Promote", {
      adapterPath: z.string(),
    }, async ({ adapterPath }) => {
      const promoted = promoteAdapter(adapterPath);
      return { content: [{ type: "text" as const, text: JSON.stringify({ promoted }) }] };
    });

    const client = new Client({ name: "test-client", version: "1.0.0" });
    const [ct, st] = InMemoryTransport.createLinkedPair();
    await Promise.all([server.connect(st), client.connect(ct)]);

    // List tools
    const tools = await client.listTools();
    const names = tools.tools.map((t) => t.name).sort();
    expect(names).toContain("flywheel_train");
    expect(names).toContain("flywheel_eval");
    expect(names).toContain("flywheel_promote");

    // Test train with missing data file
    const trainResult = await client.callTool({
      name: "flywheel_train",
      arguments: { baseModel: "test-model", trainData: "/nonexistent/data.jsonl" },
    });
    const trainData = JSON.parse((trainResult.content as Array<{ type: string; text: string }>)[0].text);
    expect(trainData.error).toContain("not found");

    // Test eval with fake adapter
    const adapterDir = path.join(tmpDir, "mcp-adapter");
    fs.mkdirSync(adapterDir);
    fs.writeFileSync(path.join(adapterDir, "adapters.safetensors"), "x".repeat(500 * 1024));

    const evalResult = await client.callTool({
      name: "flywheel_eval",
      arguments: { baseModel: "test-model", adapterPath: adapterDir },
    });
    const evalData = JSON.parse((evalResult.content as Array<{ type: string; text: string }>)[0].text);
    expect(evalData.improved).toBe(false);
    expect(evalData.details).toContain("No test data available");

    // Test promote
    const promoteResult = await client.callTool({
      name: "flywheel_promote",
      arguments: { adapterPath: adapterDir },
    });
    const promoteData = JSON.parse((promoteResult.content as Array<{ type: string; text: string }>)[0].text);
    expect(promoteData.promoted).toBeTruthy();

    await client.close();
    await server.close();
    storage.close();
  });
});

describe("Training: SQLite history", () => {
  let dbStorage: Storage;
  let dbPath: string;

  beforeEach(() => {
    dbPath = path.join(os.tmpdir(), `flywheel-train-db-${Date.now()}-${Math.random().toString(36).slice(2)}.db`);
    dbStorage = new Storage(dbPath);
    setTrainingStorage(dbStorage);
  });

  afterEach(() => {
    setTrainingStorage(null);
    dbStorage.close();
    for (const ext of ["", "-wal", "-shm"]) {
      try { fs.unlinkSync(dbPath + ext); } catch {}
    }
  });

  it("records and retrieves training runs via SQLite", () => {
    dbStorage.recordTrainingRun({
      adapterPath: "/tmp/adapter-1",
      baseModel: "test-model",
      iterations: 100,
      durationSeconds: 60.5,
      trainLoss: 0.25,
      evalLoss: 0.30,
      error: null,
    });

    dbStorage.recordTrainingRun({
      adapterPath: "/tmp/adapter-2",
      baseModel: "test-model-2",
      iterations: 200,
      durationSeconds: 120,
      trainLoss: null,
      evalLoss: null,
      error: "Training failed: OOM",
    });

    const runs = dbStorage.getTrainingRuns();
    expect(runs).toHaveLength(2);
    // Both may share the same created_at second, so just check both exist
    const models = runs.map((r) => r.baseModel).sort();
    expect(models).toEqual(["test-model", "test-model-2"]);
    const failed = runs.find((r) => r.error !== null)!;
    expect(failed.error).toBe("Training failed: OOM");
    const success = runs.find((r) => r.error === null)!;
    expect(success.trainLoss).toBe(0.25);
  });

  it("getTrainingHistory uses SQLite when storage is set", () => {
    dbStorage.recordTrainingRun({
      adapterPath: "/tmp/adapter-x",
      baseModel: "qwen",
      iterations: 50,
      durationSeconds: 30,
      trainLoss: 0.1,
      evalLoss: 0.2,
      error: null,
    });

    const history = getTrainingHistory();
    expect(history).toHaveLength(1);
    expect(history[0].baseModel).toBe("qwen");
    expect(history[0].trainLoss).toBe(0.1);
  });

  it("recordTrainingRun and getTrainingRuns round-trip correctly", () => {
    const id = dbStorage.recordTrainingRun({
      adapterPath: "/tmp/test-adapter",
      baseModel: "test-model",
      iterations: 50,
      durationSeconds: 10.5,
      trainLoss: 0.15,
      evalLoss: null,
      error: null,
    });

    expect(id).toBeTruthy();
    const runs = dbStorage.getTrainingRuns();
    expect(runs).toHaveLength(1);
    expect(runs[0].adapterPath).toBe("/tmp/test-adapter");
    expect(runs[0].durationSeconds).toBe(10.5);
    expect(runs[0].trainLoss).toBe(0.15);
    expect(runs[0].evalLoss).toBeNull();
    expect(runs[0].createdAt).toBeTruthy();
  });
});

describe("Training: Ollama integration", () => {
  it("isOllamaRunning returns boolean", async () => {
    const running = await isOllamaRunning();
    expect(typeof running).toBe("boolean");
  });

  it("getTrainingHistory returns empty array without storage", () => {
    setTrainingStorage(null);
    const history = getTrainingHistory();
    expect(history).toEqual([]);
  });
});
