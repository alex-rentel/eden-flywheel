#!/usr/bin/env node
/**
 * eden-flywheel MCP Server
 *
 * Captures AI coding sessions as training data for fine-tuning local models.
 * Exposes 9 tools: record_start, record_stop, export, status, filter, list, train, eval, promote.
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { Storage } from "./storage.js";
import { SessionCapture } from "./capture.js";
import { Exporter } from "./export.js";
import { trainAdapter, evaluateAdapter, promoteAdapter, getTrainingHistory, getActiveAdapter } from "./training.js";

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
            qualityScore: result.qualityScore,
          }),
        },
      ],
    };
  }
);

// ── flywheel_export ─────────────────────────────────────────────

server.tool(
  "flywheel_export",
  "Export recorded sessions as training JSONL. Supports formats: chatml (default), alpaca, sharegpt, raw. Can deduplicate and filter by quality score.",
  {
    sessionIds: z.array(z.string()).optional().describe("Specific session IDs to export. If omitted, exports all completed sessions."),
    format: z.enum(["chatml", "alpaca", "sharegpt", "raw"]).optional().describe("Export format (default: chatml)"),
    deduplicate: z.boolean().optional().describe("Remove near-duplicate sessions"),
    minQuality: z.number().optional().describe("Minimum quality score (0.0-1.0) to include"),
    stripSystemPrompt: z.boolean().optional().describe("Omit system prompt from ChatML output"),
  },
  async ({ sessionIds, format, deduplicate, minQuality, stripSystemPrompt }) => {
    const jsonl = exporter.exportWithOptions({ sessionIds, format, deduplicate, minQuality, stripSystemPrompt });
    const lineCount = jsonl ? jsonl.split("\n").filter(Boolean).length : 0;

    // Validate the export
    const errors = exporter.validateExport(jsonl);

    return {
      content: [
        {
          type: "text" as const,
          text: lineCount > 0
            ? `Exported ${lineCount} training examples (format: ${format || "chatml"})${errors.length > 0 ? `\nValidation warnings: ${errors.join("; ")}` : ""}:\n\n${jsonl}`
            : "No completed sessions to export.",
        },
      ],
    };
  }
);

// ── flywheel_status ─────────────────────────────────────────────

server.tool(
  "flywheel_status",
  "Show captured sessions count, total tokens, data quality stats, tool call distribution, and token histograms.",
  {},
  async () => {
    const stats = storage.getStats();
    const activeSessions = capture.getActiveSessions();
    const dataStats = exporter.getDataStats();

    return {
      content: [
        {
          type: "text" as const,
          text: JSON.stringify({
            ...stats,
            activeRecordings: activeSessions.length,
            activeSessionIds: activeSessions,
            quality: {
              avgScore: dataStats.avgQualityScore,
              distribution: dataStats.qualityDistribution,
            },
            avgTokensPerSession: dataStats.avgTokensPerSession,
            avgTurnsPerSession: dataStats.avgTurnsPerSession,
            toolCallDistribution: dataStats.toolCallDistribution,
            turnHistogram: dataStats.turnHistogram,
            tokenHistogram: dataStats.tokenHistogram,
          }, null, 2),
        },
      ],
    };
  }
);

// ── flywheel_filter ─────────────────────────────────────────────

server.tool(
  "flywheel_filter",
  "Export filtered training data. Filter by quality: sessions with tool calls, no errors, minimum message count, quality score. Supports dedup and multiple formats.",
  {
    hasToolCalls: z.boolean().optional().describe("Only include sessions that contain tool calls"),
    noErrors: z.boolean().optional().describe("Exclude sessions that contain errors"),
    minMessages: z.number().optional().describe("Minimum number of messages per session"),
    format: z.enum(["chatml", "alpaca", "sharegpt", "raw"]).optional().describe("Export format (default: chatml)"),
    deduplicate: z.boolean().optional().describe("Remove near-duplicate sessions"),
    minQuality: z.number().optional().describe("Minimum quality score (0.0-1.0)"),
  },
  async ({ hasToolCalls, noErrors, minMessages, format, deduplicate, minQuality }) => {
    const jsonl = exporter.exportFiltered({ hasToolCalls, noErrors, minMessages, format, deduplicate, minQuality });
    const lineCount = jsonl ? jsonl.split("\n").filter(Boolean).length : 0;

    return {
      content: [
        {
          type: "text" as const,
          text: lineCount > 0
            ? `Filtered export: ${lineCount} quality training examples (format: ${format || "chatml"}):\n\n${jsonl}`
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
      qualityScore: s.quality_score,
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

// ── flywheel_train ──────────────────────────────────────────────

server.tool(
  "flywheel_train",
  "Trigger a LoRA fine-tune using mlx-lm. Takes a base model and training JSONL, produces an adapter.",
  {
    baseModel: z.string().describe("Base model path or HuggingFace ID (e.g., mlx-community/Qwen2.5-Coder-3B-Instruct-4bit)"),
    trainData: z.string().describe("Path to training JSONL file"),
    outputDir: z.string().optional().describe("Output directory for adapter (default: ~/.eden-models/adapters/lora-<timestamp>)"),
    iterations: z.number().optional().describe("Training iterations (default: 100)"),
    batchSize: z.number().optional().describe("Batch size (default: 2)"),
    learningRate: z.number().optional().describe("Learning rate (default: 1e-5)"),
    loraRank: z.number().optional().describe("LoRA rank (default: 8)"),
    loraLayers: z.number().optional().describe("Number of LoRA layers (default: 16)"),
  },
  async (config) => {
    const result = await trainAdapter(config);

    return {
      content: [
        {
          type: "text" as const,
          text: JSON.stringify({
            status: result.error ? "failed" : "completed",
            adapterPath: result.adapterPath,
            baseModel: result.baseModel,
            iterations: result.iterations,
            durationSeconds: result.durationSeconds,
            trainLoss: result.trainLoss,
            evalLoss: result.evalLoss,
            error: result.error,
          }, null, 2),
        },
      ],
    };
  }
);

// ── flywheel_eval ───────────────────────────────────────────────

server.tool(
  "flywheel_eval",
  "Evaluate base model vs fine-tuned adapter on test cases. Compares performance to determine if the adapter improves the model.",
  {
    baseModel: z.string().describe("Base model path or HuggingFace ID"),
    adapterPath: z.string().describe("Path to the LoRA adapter directory"),
    testData: z.string().optional().describe("Path to test JSONL. If omitted, uses adapter size heuristic."),
  },
  async ({ baseModel, adapterPath, testData }) => {
    const result = await evaluateAdapter(baseModel, adapterPath, testData);

    return {
      content: [
        {
          type: "text" as const,
          text: JSON.stringify({
            baseScore: result.baseScore,
            adaptedScore: result.adaptedScore,
            improved: result.improved,
            testCases: result.testCases,
            details: result.details,
            recommendation: result.improved
              ? "Adapter shows improvement. Run flywheel_promote to deploy."
              : "No improvement detected. Consider more training data or tuning hyperparameters.",
          }, null, 2),
        },
      ],
    };
  }
);

// ── flywheel_promote ────────────────────────────────────────────

server.tool(
  "flywheel_promote",
  "Promote a successful adapter to ~/.eden-models/active/ for use as the default model.",
  {
    adapterPath: z.string().describe("Path to the adapter directory to promote"),
    name: z.string().optional().describe("Name for the promoted adapter (default: flywheel-latest)"),
  },
  async ({ adapterPath, name }) => {
    try {
      const promotedPath = promoteAdapter(adapterPath, name);
      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify({
              status: "promoted",
              from: adapterPath,
              to: promotedPath,
              name: name || "flywheel-latest",
            }, null, 2),
          },
        ],
      };
    } catch (err) {
      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify({
              status: "failed",
              error: err instanceof Error ? err.message : String(err),
            }),
          },
        ],
      };
    }
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
