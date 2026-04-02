import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { Storage } from "../src/storage.js";
import { SessionCapture } from "../src/capture.js";
import { Exporter } from "../src/export.js";
import { z } from "zod";
import fs from "fs";
import path from "path";
import os from "os";

let storage: Storage;
let capture: SessionCapture;
let exporter: Exporter;
let server: McpServer;
let client: Client;
let dbPath: string;

function registerTools() {
  server.tool(
    "flywheel_record_start",
    "Start recording",
    { metadata: z.string().optional() },
    async ({ metadata }) => {
      const meta = metadata ? JSON.parse(metadata) : undefined;
      const sessionId = capture.start(meta);
      return { content: [{ type: "text" as const, text: JSON.stringify({ sessionId, status: "recording" }) }] };
    }
  );

  server.tool(
    "flywheel_record_stop",
    "Stop recording",
    {
      sessionId: z.string(),
      messages: z.array(z.object({
        role: z.string(),
        content: z.string(),
        toolCallId: z.string().optional(),
        toolName: z.string().optional(),
      })),
    },
    async ({ sessionId, messages }) => {
      for (const msg of messages) capture.addMessage(sessionId, msg);
      const result = capture.stop(sessionId);
      return { content: [{ type: "text" as const, text: JSON.stringify({ sessionId, status: "saved", ...result }) }] };
    }
  );

  server.tool("flywheel_export", "Export JSONL", { sessionIds: z.array(z.string()).optional() }, async ({ sessionIds }) => {
    const jsonl = exporter.exportSessions(sessionIds);
    return { content: [{ type: "text" as const, text: jsonl || "No sessions." }] };
  });

  server.tool("flywheel_status", "Show stats", {}, async () => {
    const stats = storage.getStats();
    return { content: [{ type: "text" as const, text: JSON.stringify(stats) }] };
  });

  server.tool("flywheel_filter", "Filter export", {
    hasToolCalls: z.boolean().optional(),
    noErrors: z.boolean().optional(),
    minMessages: z.number().optional(),
  }, async ({ hasToolCalls, noErrors, minMessages }) => {
    const jsonl = exporter.exportFiltered({ hasToolCalls, noErrors, minMessages });
    return { content: [{ type: "text" as const, text: jsonl || "No matches." }] };
  });

  server.tool("flywheel_list", "List sessions", {}, async () => {
    const sessions = storage.listSessions();
    return { content: [{ type: "text" as const, text: JSON.stringify(sessions) }] };
  });
}

beforeEach(async () => {
  dbPath = path.join(os.tmpdir(), `flywheel-server-test-${Date.now()}.db`);
  storage = new Storage(dbPath);
  capture = new SessionCapture(storage);
  exporter = new Exporter(storage);

  server = new McpServer({ name: "test-flywheel", version: "1.0.0" });
  registerTools();

  client = new Client({ name: "test-client", version: "1.0.0" });
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await Promise.all([server.connect(serverTransport), client.connect(clientTransport)]);
});

afterEach(async () => {
  await client.close();
  await server.close();
  storage.close();
  try { fs.unlinkSync(dbPath); } catch {}
  try { fs.unlinkSync(dbPath + "-wal"); } catch {}
  try { fs.unlinkSync(dbPath + "-shm"); } catch {}
});

describe("MCP Server E2E", () => {
  it("lists all 6 tools", async () => {
    const result = await client.listTools();
    const names = result.tools.map((t) => t.name).sort();
    expect(names).toEqual([
      "flywheel_export",
      "flywheel_filter",
      "flywheel_list",
      "flywheel_record_start",
      "flywheel_record_stop",
      "flywheel_status",
    ]);
  });

  it("full workflow: record -> stop -> list -> export -> filter -> status", async () => {
    // 1. Start recording
    const startResult = await client.callTool({ name: "flywheel_record_start", arguments: {} });
    const startData = JSON.parse((startResult.content as Array<{ type: string; text: string }>)[0].text);
    expect(startData.status).toBe("recording");
    const sessionId = startData.sessionId;

    // 2. Stop recording with messages
    const stopResult = await client.callTool({
      name: "flywheel_record_stop",
      arguments: {
        sessionId,
        messages: [
          { role: "user", content: "Fix the login bug" },
          { role: "assistant", content: "I'll check the auth module", toolName: "read_file" },
          { role: "tool", content: "module.exports = ...", toolCallId: "tc_1" },
          { role: "assistant", content: "Fixed the issue by updating the token validation." },
        ],
      },
    });
    const stopData = JSON.parse((stopResult.content as Array<{ type: string; text: string }>)[0].text);
    expect(stopData.status).toBe("saved");
    expect(stopData.messageCount).toBe(4);

    // 3. List sessions
    const listResult = await client.callTool({ name: "flywheel_list", arguments: {} });
    const sessions = JSON.parse((listResult.content as Array<{ type: string; text: string }>)[0].text);
    expect(sessions).toHaveLength(1);
    expect(sessions[0].id).toBe(sessionId);

    // 4. Export
    const exportResult = await client.callTool({ name: "flywheel_export", arguments: {} });
    const exportText = (exportResult.content as Array<{ type: string; text: string }>)[0].text;
    expect(exportText).not.toBe("No sessions.");
    const example = JSON.parse(exportText.split("\n")[0]);
    expect(example.messages[0].role).toBe("system");

    // 5. Filter (has tool calls, no errors)
    const filterResult = await client.callTool({
      name: "flywheel_filter",
      arguments: { hasToolCalls: true, noErrors: true },
    });
    const filterText = (filterResult.content as Array<{ type: string; text: string }>)[0].text;
    expect(filterText).not.toBe("No matches.");

    // 6. Status
    const statusResult = await client.callTool({ name: "flywheel_status", arguments: {} });
    const stats = JSON.parse((statusResult.content as Array<{ type: string; text: string }>)[0].text);
    expect(stats.totalSessions).toBe(1);
    expect(stats.totalMessages).toBe(4);
    expect(stats.sessionsWithToolCalls).toBe(1);
  });

  it("status shows zero on empty database", async () => {
    const result = await client.callTool({ name: "flywheel_status", arguments: {} });
    const stats = JSON.parse((result.content as Array<{ type: string; text: string }>)[0].text);
    expect(stats.totalSessions).toBe(0);
    expect(stats.totalMessages).toBe(0);
  });

  it("export returns empty when no sessions", async () => {
    const result = await client.callTool({ name: "flywheel_export", arguments: {} });
    const text = (result.content as Array<{ type: string; text: string }>)[0].text;
    expect(text).toContain("No sessions");
  });
});
