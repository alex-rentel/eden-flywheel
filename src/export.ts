/**
 * Export captured sessions as training data in multiple formats.
 *
 * Supported formats:
 * - chatml: OpenAI ChatML format (default) — {"messages": [...]}
 * - alpaca: Alpaca format — {"instruction": "...", "input": "", "output": "..."}
 * - sharegpt: ShareGPT format — {"conversations": [{"from": "...", "value": "..."}]}
 * - raw: Raw JSONL — messages array without system prompt wrapping
 */
import { Storage, type SessionRow, type MessageRow } from "./storage.js";
import { estimateTokens } from "./tokens.js";
import {
  scoreSession,
  sessionFingerprint,
  deduplicateSessions,
  computeDataStats,
  type QualityScore,
  type DataStats,
} from "./quality.js";

export type ExportFormat = "chatml" | "alpaca" | "sharegpt" | "raw";

export interface SFTMessage {
  role: string;
  content: string;
}

export interface SFTExample {
  messages: SFTMessage[];
}

export interface ExportOptions {
  sessionIds?: string[];
  format?: ExportFormat;
  deduplicate?: boolean;
  minQuality?: number;
  stripSystemPrompt?: boolean;
}

export interface FilterOptions {
  hasToolCalls?: boolean;
  noErrors?: boolean;
  minMessages?: number;
  format?: ExportFormat;
  deduplicate?: boolean;
  minQuality?: number;
}

export class Exporter {
  private storage: Storage;

  constructor(storage: Storage) {
    this.storage = storage;
  }

  /**
   * Export sessions as training JSONL string.
   */
  exportSessions(sessionIds?: string[], format?: ExportFormat): string {
    return this.exportWithOptions({ sessionIds, format });
  }

  /**
   * Export with full options.
   */
  exportWithOptions(opts?: ExportOptions): string {
    const format = opts?.format ?? "chatml";
    let sessions = opts?.sessionIds
      ? opts.sessionIds.map((id) => this.storage.getSession(id)).filter(Boolean) as SessionRow[]
      : this.storage.listSessions().filter((s) => s.stopped_at !== null);

    // Load messages for each session
    let sessionsWithMessages = sessions.map((s) => ({
      session: s,
      messages: this.storage.getMessages(s.id),
    }));

    // Quality filtering
    if (opts?.minQuality !== undefined) {
      sessionsWithMessages = sessionsWithMessages.filter(({ session, messages }) => {
        const score = scoreSession(session, messages);
        return score.score >= opts.minQuality!;
      });
    }

    // Deduplication
    if (opts?.deduplicate) {
      const keep = deduplicateSessions(
        sessionsWithMessages.map(({ session, messages }) => ({ id: session.id, messages }))
      );
      sessionsWithMessages = sessionsWithMessages.filter(({ session }) => keep.has(session.id));
    }

    return this._formatSessions(sessionsWithMessages, format, opts?.stripSystemPrompt);
  }

  /**
   * Export with quality filters applied.
   */
  exportFiltered(opts?: FilterOptions): string {
    const format = opts?.format ?? "chatml";
    const sessions = this.storage.getQualitySessions({
      hasToolCalls: opts?.hasToolCalls,
      noErrors: opts?.noErrors,
      minMessages: opts?.minMessages,
    });

    let sessionsWithMessages = sessions.map((s) => ({
      session: s,
      messages: this.storage.getMessages(s.id),
    }));

    // Quality scoring filter
    if (opts?.minQuality !== undefined) {
      sessionsWithMessages = sessionsWithMessages.filter(({ session, messages }) => {
        const score = scoreSession(session, messages);
        return score.score >= opts.minQuality!;
      });
    }

    // Deduplication
    if (opts?.deduplicate) {
      const keep = deduplicateSessions(
        sessionsWithMessages.map(({ session, messages }) => ({ id: session.id, messages }))
      );
      sessionsWithMessages = sessionsWithMessages.filter(({ session }) => keep.has(session.id));
    }

    return this._formatSessions(sessionsWithMessages, format);
  }

  /**
   * Get quality score for a session.
   */
  getSessionQuality(sessionId: string): QualityScore | null {
    const session = this.storage.getSession(sessionId);
    if (!session) return null;
    const messages = this.storage.getMessages(sessionId);
    return scoreSession(session, messages);
  }

  /**
   * Get data statistics across all completed sessions.
   */
  getDataStats(): DataStats {
    const sessions = this.storage.listSessions().filter((s) => s.stopped_at !== null);
    const enriched = sessions.map((s) => {
      const messages = this.storage.getMessages(s.id);
      return {
        session: s,
        messages,
        quality: scoreSession(s, messages),
      };
    });
    return computeDataStats(enriched);
  }

  /**
   * Validate that exported JSONL is loadable by HuggingFace datasets.
   * Returns errors found, empty array = valid.
   */
  validateExport(jsonl: string): string[] {
    const errors: string[] = [];
    const lines = jsonl.split("\n").filter(Boolean);

    if (lines.length === 0) {
      errors.push("Empty export — no training examples");
      return errors;
    }

    for (let i = 0; i < lines.length; i++) {
      try {
        const parsed = JSON.parse(lines[i]);

        // ChatML format validation
        if (parsed.messages) {
          if (!Array.isArray(parsed.messages)) {
            errors.push(`Line ${i + 1}: messages is not an array`);
            continue;
          }
          for (let j = 0; j < parsed.messages.length; j++) {
            const msg = parsed.messages[j];
            if (typeof msg.role !== "string") {
              errors.push(`Line ${i + 1}, message ${j}: missing or invalid role`);
            }
            if (typeof msg.content !== "string") {
              errors.push(`Line ${i + 1}, message ${j}: missing or invalid content`);
            }
          }
        }
        // Alpaca format validation
        else if (parsed.instruction !== undefined) {
          if (typeof parsed.instruction !== "string") {
            errors.push(`Line ${i + 1}: invalid instruction field`);
          }
          if (typeof parsed.output !== "string") {
            errors.push(`Line ${i + 1}: invalid output field`);
          }
        }
        // ShareGPT format validation
        else if (parsed.conversations) {
          if (!Array.isArray(parsed.conversations)) {
            errors.push(`Line ${i + 1}: conversations is not an array`);
          }
        }
        else {
          errors.push(`Line ${i + 1}: unrecognized format`);
        }
      } catch {
        errors.push(`Line ${i + 1}: invalid JSON`);
      }
    }

    return errors;
  }

  // ── Private ─────────────────────────────────────────────────────

  private _formatSessions(
    sessions: Array<{ session: SessionRow; messages: MessageRow[] }>,
    format: ExportFormat,
    stripSystemPrompt?: boolean,
  ): string {
    const lines: string[] = [];

    for (const { session, messages } of sessions) {
      if (messages.length === 0) continue;

      let line: string | null = null;
      switch (format) {
        case "chatml":
          line = this._toChatML(messages, stripSystemPrompt);
          break;
        case "alpaca":
          line = this._toAlpaca(messages);
          break;
        case "sharegpt":
          line = this._toShareGPT(messages);
          break;
        case "raw":
          line = this._toRaw(messages);
          break;
      }
      if (line) lines.push(line);
    }

    return lines.join("\n");
  }

  private _normalizeMessages(messages: MessageRow[]): SFTMessage[] {
    const result: SFTMessage[] = [];

    for (const msg of messages) {
      if (msg.role === "tool") {
        result.push({ role: "tool", content: msg.content });
      } else if (msg.role === "assistant" && msg.tool_name) {
        const toolCall = JSON.stringify({ name: msg.tool_name, arguments: msg.content });
        result.push({ role: "assistant", content: `<tool_call>${toolCall}</tool_call>` });
      } else {
        result.push({ role: msg.role, content: msg.content });
      }
    }

    return result;
  }

  private _toChatML(messages: MessageRow[], stripSystemPrompt?: boolean): string | null {
    const normalized = this._normalizeMessages(messages);
    if (normalized.length < 1) return null;

    const sftMessages: SFTMessage[] = stripSystemPrompt
      ? normalized
      : [{ role: "system", content: "You are a helpful AI coding assistant with access to tools." }, ...normalized];

    if (sftMessages.length < 2) return null;
    return JSON.stringify({ messages: sftMessages });
  }

  private _toAlpaca(messages: MessageRow[]): string | null {
    // Alpaca format: instruction (all user messages), output (final assistant response)
    const userMsgs = messages.filter((m) => m.role === "user");
    const assistantMsgs = messages.filter((m) => m.role === "assistant" && !m.tool_name);

    if (userMsgs.length === 0 || assistantMsgs.length === 0) return null;

    return JSON.stringify({
      instruction: userMsgs.map((m) => m.content).join("\n"),
      input: "",
      output: assistantMsgs[assistantMsgs.length - 1].content,
    });
  }

  private _toShareGPT(messages: MessageRow[]): string | null {
    const normalized = this._normalizeMessages(messages);
    if (normalized.length < 2) return null;

    const conversations = normalized.map((msg) => ({
      from: msg.role === "user" ? "human" : msg.role === "assistant" ? "gpt" : msg.role,
      value: msg.content,
    }));

    return JSON.stringify({ conversations });
  }

  private _toRaw(messages: MessageRow[]): string | null {
    const normalized = this._normalizeMessages(messages);
    if (normalized.length < 1) return null;
    return JSON.stringify({ messages: normalized });
  }
}
