#!/usr/bin/env node
/**
 * eden-flywheel MCP Server
 *
 * Captures AI coding sessions as training data for fine-tuning local models.
 * Exposes 6 tools: record_start, record_stop, export, status, filter, list.
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { Storage } from "./storage.js";
import { SessionCapture } from "./capture.js";
import { Exporter } from "./export.js";

const storage = new Storage();
const capture = new SessionCapture(storage);
const exporter = new Exporter(storage);

const server = new McpServer({
  name: "eden-flywheel",
  version: "1.0.0",
});

// ── flywheel_record_start ───────────────────────────────────────

server.tool(
  "flywheel_record_start",
  "Start recording an AI coding session. Returns a session ID to use with record_stop.",
  {
    metadata: z.string().optional().describe("Optional JSON metadata about the session (project, model, etc.)"),
  },
  async ({ metadata }) => {
    const meta = metadata ? JSON.parse(metadata) : undefined;
    const sessionId = capture.start(meta);
    return {
      content: [
        {
          type: "text" as const,
          text: JSON.stringify({ sessionId, status: "recording" }),
        },
      ],
    };
  }
);

// ── flywheel_record_stop ────────────────────────────────────────

server.tool(
  "flywheel_record_stop",
  "Stop recording a session and save it. Provide messages captured during the session.",
  {
    sessionId: z.string().describe("The session ID returned by flywheel_record_start"),
    messages: z.array(z.object({
      role: z.string().describe("Message role: user, assistant, or tool"),
      content: z.string().describe("Message content"),
      toolCallId: z.string().optional().describe("Tool call ID if this is a tool result"),
      toolName: z.string().optional().describe("Tool name if this is a tool call"),
    })).describe("Array of messages captured during the session"),
  },
  async ({ sessionId, messages }) => {
    // Add all messages to the session
    for (const msg of messages) {
      capture.addMessage(sessionId, msg);
    }

    const result = capture.stop(sessionId);
    return {
      content: [
        {
          type: "text" as const,
          text: JSON.stringify({
            sessionId,
            status: "saved",
            messageCount: result.messageCount,
            tokenEstimate: result.tokenEstimate,
          }),
        },
      ],
    };
  }
);

// ── flywheel_export ─────────────────────────────────────────────

server.tool(
  "flywheel_export",
  "Export recorded sessions as SFT training JSONL. Each line is a complete conversation formatted for fine-tuning.",
  {
    sessionIds: z.array(z.string()).optional().describe("Specific session IDs to export. If omitted, exports all completed sessions."),
  },
  async ({ sessionIds }) => {
    const jsonl = exporter.exportSessions(sessionIds);
    const lineCount = jsonl ? jsonl.split("\n").filter(Boolean).length : 0;

    return {
      content: [
        {
          type: "text" as const,
          text: lineCount > 0
            ? `Exported ${lineCount} training examples:\n\n${jsonl}`
            : "No completed sessions to export.",
        },
      ],
    };
  }
);

// ── flywheel_status ─────────────────────────────────────────────

server.tool(
  "flywheel_status",
  "Show captured sessions count, total tokens, data quality stats.",
  {},
  async () => {
    const stats = storage.getStats();
    const activeSessions = capture.getActiveSessions();

    return {
      content: [
        {
          type: "text" as const,
          text: JSON.stringify({
            ...stats,
            activeRecordings: activeSessions.length,
            activeSessionIds: activeSessions,
          }, null, 2),
        },
      ],
    };
  }
);

// ── flywheel_filter ─────────────────────────────────────────────

server.tool(
  "flywheel_filter",
  "Export filtered training data. Filter by quality: sessions with tool calls, no errors, minimum message count.",
  {
    hasToolCalls: z.boolean().optional().describe("Only include sessions that contain tool calls"),
    noErrors: z.boolean().optional().describe("Exclude sessions that contain errors"),
    minMessages: z.number().optional().describe("Minimum number of messages per session"),
  },
  async ({ hasToolCalls, noErrors, minMessages }) => {
    const jsonl = exporter.exportFiltered({ hasToolCalls, noErrors, minMessages });
    const lineCount = jsonl ? jsonl.split("\n").filter(Boolean).length : 0;

    return {
      content: [
        {
          type: "text" as const,
          text: lineCount > 0
            ? `Filtered export: ${lineCount} quality training examples:\n\n${jsonl}`
            : "No sessions match the filter criteria.",
        },
      ],
    };
  }
);

// ── flywheel_list ───────────────────────────────────────────────

server.tool(
  "flywheel_list",
  "List all captured sessions with metadata (ID, start/stop time, message count, tokens, quality indicators).",
  {},
  async () => {
    const sessions = storage.listSessions();

    if (sessions.length === 0) {
      return {
        content: [{ type: "text" as const, text: "No sessions recorded yet." }],
      };
    }

    const summary = sessions.map((s) => ({
      id: s.id,
      startedAt: s.started_at,
      stoppedAt: s.stopped_at,
      messages: s.message_count,
      tokens: s.token_estimate,
      hasToolCalls: s.has_tool_calls > 0,
      hasErrors: s.has_errors > 0,
      status: s.stopped_at ? "completed" : "recording",
    }));

    return {
      content: [
        {
          type: "text" as const,
          text: JSON.stringify(summary, null, 2),
        },
      ],
    };
  }
);

// ── Start server ────────────────────────────────────────────────

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("eden-flywheel MCP server running on stdio");
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
