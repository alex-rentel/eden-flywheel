/**
 * SQLite storage for captured sessions and messages.
 */
import Database from "better-sqlite3";
import { randomUUID } from "crypto";
import path from "path";
import os from "os";
import fs from "fs";

export interface SessionRow {
  id: string;
  started_at: string;
  stopped_at: string | null;
  message_count: number;
  token_estimate: number;
  has_tool_calls: number;
  has_errors: number;
  metadata: string;
}

export interface MessageRow {
  id: string;
  session_id: string;
  role: string;
  content: string;
  tool_call_id: string | null;
  tool_name: string | null;
  timestamp: string;
  token_estimate: number;
}

function defaultDbPath(): string {
  const dir = path.join(os.homedir(), ".eden", "flywheel");
  return path.join(dir, "flywheel.db");
}

export class Storage {
  private db: Database.Database;

  constructor(dbPath?: string) {
    const p = dbPath || defaultDbPath();
    const dir = path.dirname(p);
    fs.mkdirSync(dir, { recursive: true });

    this.db = new Database(p);
    this.db.pragma("journal_mode = WAL");
    this.db.pragma("foreign_keys = ON");
    this._migrate();
  }

  private _migrate(): void {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        started_at TEXT NOT NULL DEFAULT (datetime('now')),
        stopped_at TEXT,
        message_count INTEGER NOT NULL DEFAULT 0,
        token_estimate INTEGER NOT NULL DEFAULT 0,
        has_tool_calls INTEGER NOT NULL DEFAULT 0,
        has_errors INTEGER NOT NULL DEFAULT 0,
        metadata TEXT NOT NULL DEFAULT '{}'
      );

      CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL DEFAULT '',
        tool_call_id TEXT,
        tool_name TEXT,
        timestamp TEXT NOT NULL DEFAULT (datetime('now')),
        token_estimate INTEGER NOT NULL DEFAULT 0
      );

      CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
    `);
  }

  // ── Sessions ──────────────────────────────────────────────────

  createSession(metadata?: Record<string, unknown>): string {
    const id = randomUUID();
    this.db
      .prepare(
        "INSERT INTO sessions (id, metadata) VALUES (?, ?)"
      )
      .run(id, JSON.stringify(metadata || {}));
    return id;
  }

  stopSession(sessionId: string): void {
    this.db
      .prepare("UPDATE sessions SET stopped_at = datetime('now') WHERE id = ?")
      .run(sessionId);
  }

  getSession(sessionId: string): SessionRow | undefined {
    return this.db
      .prepare("SELECT * FROM sessions WHERE id = ?")
      .get(sessionId) as SessionRow | undefined;
  }

  listSessions(): SessionRow[] {
    return this.db
      .prepare("SELECT * FROM sessions ORDER BY started_at DESC")
      .all() as SessionRow[];
  }

  // ── Messages ──────────────────────────────────────────────────

  addMessage(
    sessionId: string,
    role: string,
    content: string,
    opts?: { toolCallId?: string; toolName?: string }
  ): string {
    const id = randomUUID();
    const tokenEstimate = Math.ceil(content.length / 4);

    this.db
      .prepare(
        `INSERT INTO messages (id, session_id, role, content, tool_call_id, tool_name, token_estimate)
         VALUES (?, ?, ?, ?, ?, ?, ?)`
      )
      .run(id, sessionId, role, content, opts?.toolCallId ?? null, opts?.toolName ?? null, tokenEstimate);

    // Update session counters
    const isToolCall = role === "assistant" && opts?.toolName;
    const isError = content.toLowerCase().includes("error");

    this.db
      .prepare(
        `UPDATE sessions SET
           message_count = message_count + 1,
           token_estimate = token_estimate + ?,
           has_tool_calls = has_tool_calls + ?,
           has_errors = has_errors + ?
         WHERE id = ?`
      )
      .run(tokenEstimate, isToolCall ? 1 : 0, isError ? 1 : 0, sessionId);

    return id;
  }

  getMessages(sessionId: string): MessageRow[] {
    return this.db
      .prepare("SELECT * FROM messages WHERE session_id = ? ORDER BY timestamp ASC")
      .all(sessionId) as MessageRow[];
  }

  // ── Stats ─────────────────────────────────────────────────────

  getStats(): {
    totalSessions: number;
    totalMessages: number;
    totalTokens: number;
    sessionsWithToolCalls: number;
    sessionsWithErrors: number;
  } {
    const row = this.db
      .prepare(
        `SELECT
           COUNT(*) as totalSessions,
           COALESCE(SUM(message_count), 0) as totalMessages,
           COALESCE(SUM(token_estimate), 0) as totalTokens,
           COALESCE(SUM(CASE WHEN has_tool_calls > 0 THEN 1 ELSE 0 END), 0) as sessionsWithToolCalls,
           COALESCE(SUM(CASE WHEN has_errors > 0 THEN 1 ELSE 0 END), 0) as sessionsWithErrors
         FROM sessions`
      )
      .get() as Record<string, number>;

    return {
      totalSessions: row.totalSessions,
      totalMessages: row.totalMessages,
      totalTokens: row.totalTokens,
      sessionsWithToolCalls: row.sessionsWithToolCalls,
      sessionsWithErrors: row.sessionsWithErrors,
    };
  }

  // ── Filtering ─────────────────────────────────────────────────

  getQualitySessions(opts?: {
    hasToolCalls?: boolean;
    noErrors?: boolean;
    minMessages?: number;
  }): SessionRow[] {
    let where = "WHERE stopped_at IS NOT NULL";
    const params: unknown[] = [];

    if (opts?.hasToolCalls) where += " AND has_tool_calls > 0";
    if (opts?.noErrors) where += " AND has_errors = 0";
    if (opts?.minMessages) {
      where += " AND message_count >= ?";
      params.push(opts.minMessages);
    }

    return this.db
      .prepare(`SELECT * FROM sessions ${where} ORDER BY started_at DESC`)
      .all(...params) as SessionRow[];
  }

  close(): void {
    this.db.close();
  }
}
