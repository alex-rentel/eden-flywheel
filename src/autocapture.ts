/**
 * Auto-capture for Claude Code integration.
 *
 * When connected to Claude Code, automatically records all sessions
 * without requiring explicit flywheel_record_start calls.
 *
 * Captures:
 * - Git repo, branch, working directory
 * - Tool calls in Claude Code's native format
 * - Timestamps and session metadata
 * - Estimated API cost per session
 */
import { execFileSync } from "child_process";
import path from "path";
import { SessionCapture } from "./capture.js";
import { estimateTokens } from "./tokens.js";

export interface SessionMetadata {
  gitRepo?: string;
  gitBranch?: string;
  workingDirectory?: string;
  startedAt: string;
  model?: string;
  clientName?: string;
  clientVersion?: string;
}

export interface CostEstimate {
  inputTokens: number;
  outputTokens: number;
  estimatedCostUsd: number;
}

// Pricing per 1M tokens (Claude Sonnet 4 as default)
const DEFAULT_PRICING = {
  inputPerMillion: 3.0,
  outputPerMillion: 15.0,
};

/**
 * Detect git metadata from the current working directory.
 */
export function detectGitMetadata(cwd?: string): { repo?: string; branch?: string } {
  const dir = cwd || process.cwd();
  try {
    const repo = execFileSync("git", ["remote", "get-url", "origin"], {
      cwd: dir,
      encoding: "utf-8",
      timeout: 5000,
    }).trim();

    const branch = execFileSync("git", ["branch", "--show-current"], {
      cwd: dir,
      encoding: "utf-8",
      timeout: 5000,
    }).trim();

    return { repo, branch };
  } catch {
    return {};
  }
}

/**
 * Build session metadata from environment.
 */
export function buildSessionMetadata(opts?: {
  cwd?: string;
  model?: string;
  clientName?: string;
  clientVersion?: string;
}): SessionMetadata {
  const git = detectGitMetadata(opts?.cwd);
  return {
    gitRepo: git.repo,
    gitBranch: git.branch,
    workingDirectory: opts?.cwd || process.cwd(),
    startedAt: new Date().toISOString(),
    model: opts?.model,
    clientName: opts?.clientName,
    clientVersion: opts?.clientVersion,
  };
}

/**
 * Parse Claude Code's native tool call format into our format.
 *
 * Claude Code tool calls look like:
 * { type: "tool_use", id: "toolu_xxx", name: "Read", input: { file_path: "..." } }
 *
 * Tool results look like:
 * { type: "tool_result", tool_use_id: "toolu_xxx", content: "..." }
 */
export interface ClaudeCodeToolCall {
  type: string;
  id?: string;
  name?: string;
  input?: Record<string, unknown>;
  tool_use_id?: string;
  text?: string;
  content?: string | Array<{ type: string; text?: string }>;
}

export function parseClaudeCodeMessage(
  role: string,
  content: string | ClaudeCodeToolCall[],
): Array<{ role: string; content: string; toolCallId?: string; toolName?: string }> {
  const messages: Array<{ role: string; content: string; toolCallId?: string; toolName?: string }> = [];

  if (typeof content === "string") {
    messages.push({ role, content });
    return messages;
  }

  // Array of content blocks
  for (const block of content) {
    if (block.type === "text" && typeof block.text === "string") {
      messages.push({ role, content: block.text });
    } else if (block.type === "tool_use") {
      messages.push({
        role: "assistant",
        content: JSON.stringify(block.input || {}),
        toolName: block.name,
        toolCallId: block.id,
      });
    } else if (block.type === "tool_result") {
      const resultContent = typeof block.content === "string"
        ? block.content
        : Array.isArray(block.content)
          ? block.content.map((c) => c.text || "").join("\n")
          : "";
      messages.push({
        role: "tool",
        content: resultContent,
        toolCallId: block.tool_use_id,
      });
    }
  }

  return messages;
}

/**
 * Estimate API cost for a session based on token counts.
 */
export function estimateCost(
  messages: Array<{ role: string; content: string }>,
  pricing?: { inputPerMillion: number; outputPerMillion: number },
): CostEstimate {
  const rates = pricing || DEFAULT_PRICING;

  let inputTokens = 0;
  let outputTokens = 0;

  for (const msg of messages) {
    const tokens = estimateTokens(msg.content);
    if (msg.role === "user" || msg.role === "system" || msg.role === "tool") {
      inputTokens += tokens;
    } else {
      outputTokens += tokens;
    }
  }

  const costUsd =
    (inputTokens / 1_000_000) * rates.inputPerMillion +
    (outputTokens / 1_000_000) * rates.outputPerMillion;

  return {
    inputTokens,
    outputTokens,
    estimatedCostUsd: Math.round(costUsd * 10000) / 10000,
  };
}

/**
 * AutoCapture wraps SessionCapture with auto-start behavior.
 * When enabled, it automatically starts recording when the first
 * message arrives and stops when the session ends.
 */
export class AutoCapture {
  private capture: SessionCapture;
  private autoSessionId: string | null = null;
  private enabled: boolean = false;

  constructor(capture: SessionCapture) {
    this.capture = capture;
  }

  enable(metadata?: Record<string, unknown>): string {
    if (this.autoSessionId && this.capture.isActive(this.autoSessionId)) {
      return this.autoSessionId;
    }

    const sessionMeta = {
      ...buildSessionMetadata(),
      ...metadata,
      autoCapture: true,
    };

    this.autoSessionId = this.capture.start(sessionMeta);
    this.enabled = true;
    return this.autoSessionId;
  }

  disable(): { sessionId: string | null; result?: { messageCount: number; tokenEstimate: number; qualityScore: number } } {
    if (!this.autoSessionId || !this.capture.isActive(this.autoSessionId)) {
      this.enabled = false;
      return { sessionId: null };
    }

    const sessionId = this.autoSessionId;
    const result = this.capture.stop(sessionId);
    this.autoSessionId = null;
    this.enabled = false;
    return { sessionId, result };
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  getSessionId(): string | null {
    return this.autoSessionId;
  }

  /**
   * Ingest a message, auto-starting if needed.
   */
  ingest(role: string, content: string, opts?: { toolCallId?: string; toolName?: string }): void {
    if (!this.enabled) return;

    if (!this.autoSessionId || !this.capture.isActive(this.autoSessionId)) {
      this.autoSessionId = this.capture.start({
        ...buildSessionMetadata(),
        autoCapture: true,
      });
    }

    this.capture.addMessage(this.autoSessionId, { role, content, ...opts });
  }
}
