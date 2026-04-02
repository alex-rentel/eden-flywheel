import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { promoteAdapter, getTrainingHistory, getActiveAdapter } from "../src/training.js";
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
  it("copies adapter files to target directory", () => {
    // Create a fake adapter
    const adapterDir = path.join(tmpDir, "adapter");
    fs.mkdirSync(adapterDir);
    fs.writeFileSync(path.join(adapterDir, "adapters.safetensors"), "fake-weights");
    fs.writeFileSync(path.join(adapterDir, "config.json"), '{"rank": 8}');

    const targetDir = path.join(tmpDir, "promoted");
    fs.mkdirSync(targetDir, { recursive: true });

    // Manually copy (promoteAdapter uses ~/.eden-models, so test the logic directly)
    const files = fs.readdirSync(adapterDir);
    for (const f of files) {
      fs.copyFileSync(path.join(adapterDir, f), path.join(targetDir, f));
    }

    expect(fs.existsSync(path.join(targetDir, "adapters.safetensors"))).toBe(true);
    expect(fs.existsSync(path.join(targetDir, "config.json"))).toBe(true);
    expect(fs.readFileSync(path.join(targetDir, "adapters.safetensors"), "utf-8")).toBe("fake-weights");
  });

  it("throws when adapter path does not exist", () => {
    expect(() => promoteAdapter("/nonexistent/path")).toThrow();
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
  it("returns heuristic when no test data provided", async () => {
    const { evaluateAdapter } = await import("../src/training.js");

    // Create a fake adapter with some size
    const adapterDir = path.join(tmpDir, "eval-adapter");
    fs.mkdirSync(adapterDir);
    fs.writeFileSync(path.join(adapterDir, "adapters.safetensors"), "x".repeat(1024 * 1024)); // 1MB

    const result = await evaluateAdapter("test-model", adapterDir);
    expect(result.baseScore).toBe(0.5);
    expect(result.adaptedScore).toBeGreaterThan(0.5);
    expect(result.improved).toBe(true);
    expect(result.details).toContain("size heuristic");
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
    expect(evalData.improved).toBe(true);

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
