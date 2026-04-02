import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { Storage } from "../src/storage.js";
import { SessionCapture } from "../src/capture.js";
import { Exporter } from "../src/export.js";
import { AutoCapture } from "../src/autocapture.js";
import { getTrainingHistory, getActiveAdapter } from "../src/training.js";
import fs from "fs";
import path from "path";
import os from "os";

let storage: Storage;
let capture: SessionCapture;
let exporter: Exporter;
let autoCapture: AutoCapture;
let server: McpServer;
let client: Client;
let dbPath: string;

function registerResources() {
  server.resource(
    "status",
    "flywheel://status",
    { description: "Status", mimeType: "application/json" },
    async () => ({
      contents: [{
        uri: "flywheel://status",
        mimeType: "application/json",
        text: JSON.stringify({
          ...storage.getStats(),
          activeRecordings: capture.getActiveSessions().length,
          autoCapture: autoCapture.isEnabled(),
        }, null, 2),
      }],
    })
  );

  server.resource(
    "sessions",
    "flywheel://sessions",
    { description: "Sessions", mimeType: "application/json" },
    async () => ({
      contents: [{
        uri: "flywheel://sessions",
        mimeType: "application/json",
        text: JSON.stringify(storage.listSessions().map((s) => ({
          id: s.id,
          messages: s.message_count,
          status: s.stopped_at ? "completed" : "recording",
        })), null, 2),
      }],
    })
  );

  server.resource(
    "latest-export",
    "flywheel://latest-export",
    { description: "Export", mimeType: "application/jsonl" },
    async () => ({
      contents: [{
        uri: "flywheel://latest-export",
        mimeType: "application/jsonl",
        text: exporter.exportWithOptions({ format: "chatml" }) || "// empty",
      }],
    })
  );

  server.resource(
    "training-history",
    "flywheel://training-history",
    { description: "History", mimeType: "application/json" },
    async () => ({
      contents: [{
        uri: "flywheel://training-history",
        mimeType: "application/json",
        text: JSON.stringify({
          runs: getTrainingHistory(),
          activeAdapter: getActiveAdapter(),
        }, null, 2),
      }],
    })
  );
}

beforeEach(async () => {
  dbPath = path.join(os.tmpdir(), `flywheel-res-${Date.now()}-${Math.random().toString(36).slice(2)}.db`);
  storage = new Storage(dbPath);
  capture = new SessionCapture(storage);
  exporter = new Exporter(storage);
  autoCapture = new AutoCapture(capture);

  server = new McpServer(
    { name: "test-flywheel", version: "1.0.0" },
    { capabilities: { resources: {}, tools: {} } }
  );
  registerResources();

  client = new Client({ name: "test-client", version: "1.0.0" }, { capabilities: {} });
  const [ct, st] = InMemoryTransport.createLinkedPair();
  await Promise.all([server.connect(st), client.connect(ct)]);
});

afterEach(async () => {
  await client.close();
  await server.close();
  storage.close();
  for (const ext of ["", "-wal", "-shm"]) {
    try { fs.unlinkSync(dbPath + ext); } catch {}
  }
});

describe("MCP Resources", () => {
  it("lists all 4 resources", async () => {
    const result = await client.listResources();
    const uris = result.resources.map((r) => r.uri).sort();
    expect(uris).toEqual([
      "flywheel://latest-export",
      "flywheel://sessions",
      "flywheel://status",
      "flywheel://training-history",
    ]);
  });

  it("reads flywheel://status", async () => {
    const result = await client.readResource({ uri: "flywheel://status" });
    const text = (result.contents[0] as { text: string }).text;
    const data = JSON.parse(text);
    expect(data.totalSessions).toBe(0);
    expect(data.autoCapture).toBe(false);
  });

  it("reads flywheel://status with data", async () => {
    const id = storage.createSession();
    storage.addMessage(id, "user", "test");
    storage.stopSession(id);

    const result = await client.readResource({ uri: "flywheel://status" });
    const data = JSON.parse((result.contents[0] as { text: string }).text);
    expect(data.totalSessions).toBe(1);
    expect(data.totalMessages).toBe(1);
  });

  it("reads flywheel://sessions", async () => {
    const id1 = storage.createSession();
    storage.addMessage(id1, "user", "test1");
    storage.stopSession(id1);

    const id2 = storage.createSession();
    storage.addMessage(id2, "user", "test2");

    const result = await client.readResource({ uri: "flywheel://sessions" });
    const sessions = JSON.parse((result.contents[0] as { text: string }).text);
    expect(sessions).toHaveLength(2);
    expect(sessions.find((s: { id: string }) => s.id === id1).status).toBe("completed");
    expect(sessions.find((s: { id: string }) => s.id === id2).status).toBe("recording");
  });

  it("reads flywheel://latest-export", async () => {
    const id = storage.createSession();
    storage.addMessage(id, "user", "Fix the bug");
    storage.addMessage(id, "assistant", "Done!");
    storage.stopSession(id);

    const result = await client.readResource({ uri: "flywheel://latest-export" });
    const text = (result.contents[0] as { text: string }).text;
    const lines = text.split("\n").filter(Boolean);
    expect(lines).toHaveLength(1);

    const parsed = JSON.parse(lines[0]);
    expect(parsed.messages).toBeDefined();
    expect(parsed.messages[0].role).toBe("system");
  });

  it("reads flywheel://latest-export when empty", async () => {
    const result = await client.readResource({ uri: "flywheel://latest-export" });
    const text = (result.contents[0] as { text: string }).text;
    expect(text).toContain("empty");
  });

  it("reads flywheel://training-history", async () => {
    const result = await client.readResource({ uri: "flywheel://training-history" });
    const data = JSON.parse((result.contents[0] as { text: string }).text);
    expect(data.runs).toBeDefined();
    expect(Array.isArray(data.runs)).toBe(true);
  });
});
