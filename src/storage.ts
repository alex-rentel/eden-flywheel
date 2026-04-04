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
  quality_score: number;
  fingerprint: string | null;
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
  const dir = path.join(os.homedir(), ".config", "training-flywheel");
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
        quality_score REAL NOT NULL DEFAULT 0,
        fingerprint TEXT,
        metadata TEXT NOT NULL DEFAULT '{}'
      );

      CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        sequence INTEGER NOT NULL DEFAULT 0,
        role TEXT NOT NULL,
        content TEXT NOT NULL DEFAULT '',
        tool_call_id TEXT,
        tool_name TEXT,
        timestamp TEXT NOT NULL DEFAULT (datetime('now')),
        token_estimate INTEGER NOT NULL DEFAULT 0
      );

      CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

      CREATE TABLE IF NOT EXISTS training_runs (
        id TEXT PRIMARY KEY,
        adapter_path TEXT NOT NULL,
        base_model TEXT NOT NULL,
        iterations INTEGER NOT NULL,
        duration_seconds REAL NOT NULL,
        train_loss REAL,
        eval_loss REAL,
        error TEXT,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
      );
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

    // Get next sequence number for this session
    const seqRow = this.db
      .prepare("SELECT COALESCE(MAX(sequence), -1) + 1 AS next_seq FROM messages WHERE session_id = ?")
      .get(sessionId) as { next_seq: number };
    const seq = seqRow.next_seq;

    this.db
      .prepare(
        `INSERT INTO messages (id, session_id, sequence, role, content, tool_call_id, tool_name, token_estimate)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(id, sessionId, seq, role, content, opts?.toolCallId ?? null, opts?.toolName ?? null, tokenEstimate);

    // Update session counters
    const isToolCall = role === "assistant" && opts?.toolName;
    const isError = role === "tool" && (
      content.toLowerCase().includes("error:") ||
      content.toLowerCase().includes("failed") ||
      content.toLowerCase().includes("exception:")
    );

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
      .prepare("SELECT * FROM messages WHERE session_id = ? ORDER BY sequence ASC")
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

  updateSessionQuality(sessionId: string, score: number, fingerprint: string): void {
    this.db
      .prepare("UPDATE sessions SET quality_score = ?, fingerprint = ? WHERE id = ?")
      .run(score, fingerprint, sessionId);
  }

  // ── Training Runs ──────────────────────────────────────────────

  recordTrainingRun(run: {
    adapterPath: string;
    baseModel: string;
    iterations: number;
    durationSeconds: number;
    trainLoss: number | null;
    evalLoss: number | null;
    error: string | null;
  }): string {
    const id = randomUUID();
    this.db
      .prepare(
        `INSERT INTO training_runs (id, adapter_path, base_model, iterations, duration_seconds, train_loss, eval_loss, error)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`
      )
      .run(id, run.adapterPath, run.baseModel, run.iterations, run.durationSeconds, run.trainLoss, run.evalLoss, run.error);
    return id;
  }

  getTrainingRuns(): Array<{
    id: string;
    adapterPath: string;
    baseModel: string;
    iterations: number;
    durationSeconds: number;
    trainLoss: number | null;
    evalLoss: number | null;
    error: string | null;
    createdAt: string;
  }> {
    const rows = this.db
      .prepare("SELECT * FROM training_runs ORDER BY created_at DESC")
      .all() as Array<{
        id: string;
        adapter_path: string;
        base_model: string;
        iterations: number;
        duration_seconds: number;
        train_loss: number | null;
        eval_loss: number | null;
        error: string | null;
        created_at: string;
      }>;

    return rows.map((r) => ({
      id: r.id,
      adapterPath: r.adapter_path,
      baseModel: r.base_model,
      iterations: r.iterations,
      durationSeconds: r.duration_seconds,
      trainLoss: r.train_loss,
      evalLoss: r.eval_loss,
      error: r.error,
      createdAt: r.created_at,
    }));
  }

  close(): void {
    this.db.close();
  }
}
