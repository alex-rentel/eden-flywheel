#!/usr/bin/env node
/**
 * training-flywheel MCP Server
 *
 * Captures AI coding sessions as training data for fine-tuning local models.
 * The closed loop: use AI -> capture -> fine-tune -> deploy better model -> repeat.
 */
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { Storage } from "./storage.js";
import { SessionCapture } from "./capture.js";
import { Exporter } from "./export.js";
import { trainAdapter, evaluateAdapter, promoteAdapter, getTrainingHistory, getActiveAdapter, setTrainingStorage } from "./training.js";
import { AutoCapture, estimateCost, parseClaudeCodeMessage, buildSessionMetadata } from "./autocapture.js";
import { parseCliArgs, resolveConfig } from "./config.js";
import { logger, setLogLevel } from "./logger.js";

// Resolve config from file + CLI args
const cliOverrides = parseCliArgs(process.argv.slice(2));
const config = resolveConfig(cliOverrides);

if (config.logLevel) setLogLevel(config.logLevel);

const storage = new Storage(config.dbPath);
setTrainingStorage(storage);
const capture = new SessionCapture(storage);
const exporter = new Exporter(storage);
const autoCapture = new AutoCapture(capture);

const server = new McpServer({
  name: "training-flywheel",
  version: "1.0.0",
}, {
  capabilities: {
    resources: {},
    tools: {},
  },
});

// ── MCP Resources ───────────────────────────────────────────────

server.resource(
  "status",
  "flywheel://status",
  { description: "Current recording state and stats", mimeType: "application/json" },
  async () => {
    const stats = storage.getStats();
    const activeSessions = capture.getActiveSessions();
    const dataStats = exporter.getDataStats();

    return {
      contents: [
        {
          uri: "flywheel://status",
          mimeType: "application/json",
          text: JSON.stringify({
            ...stats,
            activeRecordings: activeSessions.length,
            activeSessionIds: activeSessions,
            autoCapture: autoCapture.isEnabled(),
            autoSessionId: autoCapture.getSessionId(),
            quality: {
              avgScore: dataStats.avgQualityScore,
              distribution: dataStats.qualityDistribution,
            },
            avgTokensPerSession: dataStats.avgTokensPerSession,
            avgTurnsPerSession: dataStats.avgTurnsPerSession,
            toolCallDistribution: dataStats.toolCallDistribution,
          }, null, 2),
        },
      ],
    };
  }
);

server.resource(
  "sessions",
  "flywheel://sessions",
  { description: "Session index as structured data", mimeType: "application/json" },
  async () => {
    const sessions = storage.listSessions();
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
      contents: [
        {
          uri: "flywheel://sessions",
          mimeType: "application/json",
          text: JSON.stringify(summary, null, 2),
        },
      ],
    };
  }
);

server.resource(
  "latest-export",
  "flywheel://latest-export",
  { description: "Most recent JSONL export", mimeType: "application/jsonl" },
  async () => {
    const jsonl = exporter.exportWithOptions({ format: "chatml" });
    return {
      contents: [
        {
          uri: "flywheel://latest-export",
          mimeType: "application/jsonl",
          text: jsonl || "// No completed sessions to export",
        },
      ],
    };
  }
);

server.resource(
  "training-history",
  "flywheel://training-history",
  { description: "All fine-tune runs and results", mimeType: "application/json" },
  async () => {
    const history = getTrainingHistory();
    const active = getActiveAdapter();

    return {
      contents: [
        {
          uri: "flywheel://training-history",
          mimeType: "application/json",
          text: JSON.stringify({
            runs: history,
            activeAdapter: active,
            totalRuns: history.length,
            successfulRuns: history.filter((r) => !r.error).length,
          }, null, 2),
        },
      ],
    };
  }
);

// ── flywheel_record_start ───────────────────────────────────────

server.tool(
  "flywheel_record_start",
  "Start recording an AI coding session. Returns a session ID to use with record_stop.",
  {
    metadata: z.string().optional().describe("Optional JSON metadata about the session (project, model, etc.)"),
  },
  async ({ metadata }) => {
    let meta: Record<string, unknown> | undefined;
    if (metadata) {
      try {
        meta = JSON.parse(metadata);
      } catch {
        return {
          content: [{ type: "text" as const, text: JSON.stringify({ error: "Invalid JSON in metadata" }) }],
        };
      }
    }
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
  "Export recorded sessions as training JSONL. Supports formats: chatml (default), alpaca, sharegpt, raw. Can deduplicate, filter by quality, and auto-split 10% as held-out eval data.",
  {
    sessionIds: z.array(z.string()).optional().describe("Specific session IDs to export. If omitted, exports all completed sessions."),
    format: z.enum(["chatml", "alpaca", "sharegpt", "raw"]).optional().describe("Export format (default: chatml)"),
    deduplicate: z.boolean().optional().describe("Remove near-duplicate sessions"),
    minQuality: z.number().optional().describe("Minimum quality score (0.0-1.0) to include"),
    stripSystemPrompt: z.boolean().optional().describe("Omit system prompt from ChatML output"),
    evalSplit: z.boolean().optional().describe("Auto-split 10% of sessions into a held-out eval set (saved alongside training JSONL)"),
    outputPath: z.string().optional().describe("Path to write JSONL file. If evalSplit is true, eval set is written to <path>.eval.jsonl"),
  },
  async ({ sessionIds, format, deduplicate, minQuality, stripSystemPrompt, evalSplit, outputPath }) => {
    if (evalSplit) {
      const { train, eval: evalData } = exporter.exportWithEvalSplit({
        sessionIds, format, deduplicate, minQuality, stripSystemPrompt, evalSplitPercent: 10,
      });
      const trainLineCount = train ? train.split("\n").filter(Boolean).length : 0;
      const evalLineCount = evalData ? evalData.split("\n").filter(Boolean).length : 0;

      if (outputPath && trainLineCount > 0) {
        const fs = await import("fs");
        fs.writeFileSync(outputPath, train + "\n");
        fs.writeFileSync(outputPath.replace(/\.jsonl$/, "") + ".eval.jsonl", evalData + "\n");
      }

      const errors = exporter.validateExport(train);
      return {
        content: [
          {
            type: "text" as const,
            text: trainLineCount > 0
              ? `Exported ${trainLineCount} training + ${evalLineCount} eval examples (format: ${format || "chatml"})${errors.length > 0 ? `\nValidation warnings: ${errors.join("; ")}` : ""}${outputPath ? `\nTrain: ${outputPath}\nEval: ${outputPath.replace(/\.jsonl$/, "")}.eval.jsonl` : ""}:\n\n${train}`
              : "No completed sessions to export.",
          },
        ],
      };
    }

    const jsonl = exporter.exportWithOptions({ sessionIds, format, deduplicate, minQuality, stripSystemPrompt });
    const lineCount = jsonl ? jsonl.split("\n").filter(Boolean).length : 0;

    if (outputPath && lineCount > 0) {
      const fs = await import("fs");
      fs.writeFileSync(outputPath, jsonl + "\n");
    }

    const errors = exporter.validateExport(jsonl);
    return {
      content: [
        {
          type: "text" as const,
          text: lineCount > 0
            ? `Exported ${lineCount} training examples (format: ${format || "chatml"})${errors.length > 0 ? `\nValidation warnings: ${errors.join("; ")}` : ""}${outputPath ? `\nWritten to: ${outputPath}` : ""}:\n\n${jsonl}`
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
    outputDir: z.string().optional().describe("Output directory for adapter (default: ~/.config/training-flywheel/models/adapters/lora-<timestamp>)"),
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
  "Evaluate base model vs fine-tuned adapter on held-out test data. Requires an eval set — use flywheel_export with evalSplit: true to generate one.",
  {
    baseModel: z.string().describe("Base model path or HuggingFace ID"),
    adapterPath: z.string().describe("Path to the LoRA adapter directory"),
    testData: z.string().describe("Path to eval JSONL file (generated by flywheel_export with evalSplit: true)"),
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
  "Promote a successful adapter to the active slot for deployment. Copies adapter to ~/.config/training-flywheel/models/active/.",
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

// ── flywheel_autocapture ────────────────────────────────────────

server.tool(
  "flywheel_autocapture",
  "Enable or disable automatic session recording. When enabled, all messages are captured without needing record_start/stop.",
  {
    enabled: z.boolean().describe("true to enable, false to disable auto-capture"),
    model: z.string().optional().describe("Model being used (for cost tracking)"),
  },
  async ({ enabled, model }) => {
    if (enabled) {
      const sessionId = autoCapture.enable({ model });
      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify({
              status: "enabled",
              sessionId,
              metadata: buildSessionMetadata({ model }),
            }, null, 2),
          },
        ],
      };
    } else {
      const { sessionId, result } = autoCapture.disable();
      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify({
              status: "disabled",
              sessionId,
              ...(result || {}),
            }, null, 2),
          },
        ],
      };
    }
  }
);

// ── flywheel_ingest ─────────────────────────────────────────────

server.tool(
  "flywheel_ingest",
  "Ingest a message into the auto-capture session. Use this to feed messages when auto-capture is enabled.",
  {
    role: z.string().describe("Message role: user, assistant, or tool"),
    content: z.string().describe("Message content"),
    toolCallId: z.string().optional().describe("Tool call ID"),
    toolName: z.string().optional().describe("Tool name"),
  },
  async ({ role, content, toolCallId, toolName }) => {
    autoCapture.ingest(role, content, { toolCallId, toolName });
    return {
      content: [
        {
          type: "text" as const,
          text: JSON.stringify({
            status: "ingested",
            sessionId: autoCapture.getSessionId(),
            role,
            contentLength: content.length,
          }),
        },
      ],
    };
  }
);

// ── flywheel_cost ───────────────────────────────────────────────

server.tool(
  "flywheel_cost",
  "Estimate API cost for a session. Shows input/output token counts and estimated USD cost.",
  {
    sessionId: z.string().describe("Session ID to calculate cost for"),
    inputPricePerMillion: z.number().optional().describe("Input token price per million (default: $3.00)"),
    outputPricePerMillion: z.number().optional().describe("Output token price per million (default: $15.00)"),
  },
  async ({ sessionId, inputPricePerMillion, outputPricePerMillion }) => {
    const messages = storage.getMessages(sessionId);
    if (messages.length === 0) {
      return {
        content: [{ type: "text" as const, text: JSON.stringify({ error: "Session not found or empty" }) }],
      };
    }

    const pricing = (inputPricePerMillion || outputPricePerMillion)
      ? {
          inputPerMillion: inputPricePerMillion ?? 3.0,
          outputPerMillion: outputPricePerMillion ?? 15.0,
        }
      : undefined;

    const cost = estimateCost(messages, pricing);

    return {
      content: [
        {
          type: "text" as const,
          text: JSON.stringify({
            sessionId,
            ...cost,
            messageCount: messages.length,
          }, null, 2),
        },
      ],
    };
  }
);

// ── Start server ────────────────────────────────────────────────

async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  logger.info("training-flywheel MCP server running on stdio", {
    logLevel: config.logLevel || "info",
    dbPath: config.dbPath || "default",
  });
}

main().catch((err) => {
  logger.error("Fatal error", { error: err instanceof Error ? err.message : String(err) });
  process.exit(1);
});
