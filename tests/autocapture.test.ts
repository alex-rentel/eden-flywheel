import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { Storage } from "../src/storage.js";
import { SessionCapture } from "../src/capture.js";
import {
  AutoCapture,
  estimateCost,
  parseClaudeCodeMessage,
  buildSessionMetadata,
  detectGitMetadata,
} from "../src/autocapture.js";
import fs from "fs";
import path from "path";
import os from "os";

let storage: Storage;
let capture: SessionCapture;
let autoCapture: AutoCapture;
let dbPath: string;

beforeEach(() => {
  dbPath = path.join(os.tmpdir(), `flywheel-auto-${Date.now()}-${Math.random().toString(36).slice(2)}.db`);
  storage = new Storage(dbPath);
  capture = new SessionCapture(storage);
  autoCapture = new AutoCapture(capture);
});

afterEach(() => {
  // Stop any active sessions
  if (autoCapture.isEnabled()) autoCapture.disable();
  storage.close();
  for (const ext of ["", "-wal", "-shm"]) {
    try { fs.unlinkSync(dbPath + ext); } catch {}
  }
});

describe("AutoCapture", () => {
  it("enables and creates a session", () => {
    const sessionId = autoCapture.enable();
    expect(sessionId).toBeTruthy();
    expect(autoCapture.isEnabled()).toBe(true);
    expect(autoCapture.getSessionId()).toBe(sessionId);
  });

  it("disables and stops the session", () => {
    autoCapture.enable();
    autoCapture.ingest("user", "Hello");
    autoCapture.ingest("assistant", "Hi there");

    const { sessionId, result } = autoCapture.disable();
    expect(sessionId).toBeTruthy();
    expect(result!.messageCount).toBe(2);
    expect(autoCapture.isEnabled()).toBe(false);
  });

  it("ingests messages automatically", () => {
    autoCapture.enable();
    autoCapture.ingest("user", "Fix the bug");
    autoCapture.ingest("assistant", "Reading file", { toolName: "read_file" });
    autoCapture.ingest("tool", "file contents", { toolCallId: "tc_1" });
    autoCapture.ingest("assistant", "Done");

    const { result } = autoCapture.disable();
    expect(result!.messageCount).toBe(4);
  });

  it("silently ignores messages when disabled", () => {
    autoCapture.ingest("user", "This should be ignored");
    expect(autoCapture.getSessionId()).toBeNull();
  });

  it("enable returns same session if already active", () => {
    const id1 = autoCapture.enable();
    const id2 = autoCapture.enable();
    expect(id1).toBe(id2);
  });

  it("disable returns null when not enabled", () => {
    const { sessionId } = autoCapture.disable();
    expect(sessionId).toBeNull();
  });
});

describe("Claude Code tool call parsing", () => {
  it("parses text content", () => {
    const msgs = parseClaudeCodeMessage("user", "Hello");
    expect(msgs).toHaveLength(1);
    expect(msgs[0].role).toBe("user");
    expect(msgs[0].content).toBe("Hello");
  });

  it("parses tool_use blocks", () => {
    const blocks = [
      { type: "tool_use", id: "toolu_123", name: "Read", input: { file_path: "/src/index.ts" } },
    ];
    const msgs = parseClaudeCodeMessage("assistant", blocks);
    expect(msgs).toHaveLength(1);
    expect(msgs[0].role).toBe("assistant");
    expect(msgs[0].toolName).toBe("Read");
    expect(msgs[0].toolCallId).toBe("toolu_123");
    expect(JSON.parse(msgs[0].content)).toEqual({ file_path: "/src/index.ts" });
  });

  it("parses tool_result blocks", () => {
    const blocks = [
      { type: "tool_result", tool_use_id: "toolu_123", content: "file contents here" },
    ];
    const msgs = parseClaudeCodeMessage("user", blocks);
    expect(msgs).toHaveLength(1);
    expect(msgs[0].role).toBe("tool");
    expect(msgs[0].content).toBe("file contents here");
    expect(msgs[0].toolCallId).toBe("toolu_123");
  });

  it("parses mixed content blocks", () => {
    const blocks = [
      { type: "text", text: "Let me read the file" },
      { type: "tool_use", id: "toolu_456", name: "Bash", input: { command: "ls -la" } },
    ];
    const msgs = parseClaudeCodeMessage("assistant", blocks);
    expect(msgs).toHaveLength(2);
    expect(msgs[0].role).toBe("assistant");
    expect(msgs[0].content).toBe("Let me read the file");
    expect(msgs[1].toolName).toBe("Bash");
  });

  it("captures text blocks using block.text (Claude API format)", () => {
    const blocks = [
      { type: "text", text: "hello" },
    ];
    const msgs = parseClaudeCodeMessage("assistant", blocks);
    expect(msgs).toHaveLength(1);
    expect(msgs[0].role).toBe("assistant");
    expect(msgs[0].content).toBe("hello");
  });

  it("ignores text blocks with content instead of text field", () => {
    const blocks = [
      { type: "text", content: "old format" },
    ];
    const msgs = parseClaudeCodeMessage("assistant", blocks);
    expect(msgs).toHaveLength(0);
  });

  it("handles array content in tool_result", () => {
    const blocks = [
      {
        type: "tool_result",
        tool_use_id: "toolu_789",
        content: [
          { type: "text", text: "Line 1" },
          { type: "text", text: "Line 2" },
        ],
      },
    ];
    const msgs = parseClaudeCodeMessage("user", blocks);
    expect(msgs[0].content).toBe("Line 1\nLine 2");
  });
});

describe("Cost estimation", () => {
  it("estimates cost for a simple conversation", () => {
    const messages = [
      { role: "user", content: "Fix the authentication bug in login.ts" },
      { role: "assistant", content: "I'll read the file and fix the issue. Here's the updated code with proper token validation." },
    ];
    const cost = estimateCost(messages);
    expect(cost.inputTokens).toBeGreaterThan(0);
    expect(cost.outputTokens).toBeGreaterThan(0);
    expect(cost.estimatedCostUsd).toBeGreaterThan(0);
    expect(cost.estimatedCostUsd).toBeLessThan(1); // Should be tiny
  });

  it("uses custom pricing", () => {
    const messages = [
      { role: "user", content: "x".repeat(4000) }, // ~1000 tokens
      { role: "assistant", content: "x".repeat(4000) },
    ];
    const defaultCost = estimateCost(messages);
    const expensiveCost = estimateCost(messages, {
      inputPerMillion: 15.0,
      outputPerMillion: 75.0,
    });
    expect(expensiveCost.estimatedCostUsd).toBeGreaterThan(defaultCost.estimatedCostUsd);
  });

  it("separates input and output tokens correctly", () => {
    const messages = [
      { role: "system", content: "You are helpful." },
      { role: "user", content: "Hello" },
      { role: "tool", content: "Tool result" },
      { role: "assistant", content: "Response" },
    ];
    const cost = estimateCost(messages);
    expect(cost.inputTokens).toBeGreaterThan(cost.outputTokens);
  });
});

describe("Session metadata", () => {
  it("builds metadata with git info", () => {
    const meta = buildSessionMetadata({ cwd: process.cwd() });
    expect(meta.startedAt).toBeTruthy();
    expect(meta.workingDirectory).toBeTruthy();
    // Git info may or may not be available in test environment
  });

  it("handles missing git gracefully", () => {
    const git = detectGitMetadata("/nonexistent/dir");
    expect(git.repo).toBeUndefined();
    expect(git.branch).toBeUndefined();
  });

  it("includes model and client info when provided", () => {
    const meta = buildSessionMetadata({
      model: "claude-sonnet-4-6",
      clientName: "claude-code",
      clientVersion: "1.0.0",
    });
    expect(meta.model).toBe("claude-sonnet-4-6");
    expect(meta.clientName).toBe("claude-code");
  });
});

describe("MCP auto-capture tools", () => {
  it("autocapture, ingest, and cost tools work via MCP", async () => {
    const { McpServer } = await import("@modelcontextprotocol/sdk/server/mcp.js");
    const { InMemoryTransport } = await import("@modelcontextprotocol/sdk/inMemory.js");
    const { Client } = await import("@modelcontextprotocol/sdk/client/index.js");
    const { z } = await import("zod");

    const server = new McpServer({ name: "test", version: "1.0.0" });

    // Register tools
    server.tool("flywheel_autocapture", "Toggle auto-capture", {
      enabled: z.boolean(),
    }, async ({ enabled }) => {
      if (enabled) {
        const sid = autoCapture.enable();
        return { content: [{ type: "text" as const, text: JSON.stringify({ status: "enabled", sessionId: sid }) }] };
      } else {
        const { sessionId, result } = autoCapture.disable();
        return { content: [{ type: "text" as const, text: JSON.stringify({ status: "disabled", sessionId, ...result }) }] };
      }
    });

    server.tool("flywheel_ingest", "Ingest message", {
      role: z.string(),
      content: z.string(),
      toolName: z.string().optional(),
    }, async ({ role, content, toolName }) => {
      autoCapture.ingest(role, content, { toolName });
      return { content: [{ type: "text" as const, text: "ok" }] };
    });

    server.tool("flywheel_cost", "Get cost", {
      sessionId: z.string(),
    }, async ({ sessionId }) => {
      const messages = storage.getMessages(sessionId);
      const cost = estimateCost(messages);
      return { content: [{ type: "text" as const, text: JSON.stringify(cost) }] };
    });

    const client = new Client({ name: "test-client", version: "1.0.0" });
    const [ct, st] = InMemoryTransport.createLinkedPair();
    await Promise.all([server.connect(st), client.connect(ct)]);

    // Enable auto-capture
    const enableResult = await client.callTool({ name: "flywheel_autocapture", arguments: { enabled: true } });
    const enableData = JSON.parse((enableResult.content as Array<{ type: string; text: string }>)[0].text);
    expect(enableData.status).toBe("enabled");
    const sessionId = enableData.sessionId;

    // Ingest messages
    await client.callTool({ name: "flywheel_ingest", arguments: { role: "user", content: "Hello world" } });
    await client.callTool({ name: "flywheel_ingest", arguments: { role: "assistant", content: "Hi there!", toolName: "greeting" } });

    // Check cost
    const costResult = await client.callTool({ name: "flywheel_cost", arguments: { sessionId } });
    const costData = JSON.parse((costResult.content as Array<{ type: string; text: string }>)[0].text);
    expect(costData.inputTokens).toBeGreaterThan(0);
    expect(costData.outputTokens).toBeGreaterThan(0);
    expect(costData.estimatedCostUsd).toBeGreaterThanOrEqual(0);

    // Disable
    const disableResult = await client.callTool({ name: "flywheel_autocapture", arguments: { enabled: false } });
    const disableData = JSON.parse((disableResult.content as Array<{ type: string; text: string }>)[0].text);
    expect(disableData.status).toBe("disabled");
    expect(disableData.messageCount).toBe(2);

    await client.close();
    await server.close();
  });
});
