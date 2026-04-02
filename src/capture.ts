/**
 * Session capture — records messages for an active session.
 */
import { Storage } from "./storage.js";
import { scoreSession, sessionFingerprint } from "./quality.js";

export interface RecordedMessage {
  role: string;
  content: string;
  toolCallId?: string;
  toolName?: string;
}

export class SessionCapture {
  private activeSessions: Map<string, boolean> = new Map();
  private storage: Storage;

  constructor(storage: Storage) {
    this.storage = storage;
  }

  start(metadata?: Record<string, unknown>): string {
    const sessionId = this.storage.createSession(metadata);
    this.activeSessions.set(sessionId, true);
    return sessionId;
  }

  stop(sessionId: string): { messageCount: number; tokenEstimate: number; qualityScore: number } {
    if (!this.activeSessions.has(sessionId)) {
      throw new Error(`Session ${sessionId} is not active`);
    }

    this.storage.stopSession(sessionId);
    this.activeSessions.delete(sessionId);

    const session = this.storage.getSession(sessionId);
    const messages = this.storage.getMessages(sessionId);

    // Compute quality score and fingerprint
    const quality = scoreSession(session!, messages);
    const fp = sessionFingerprint(messages);
    this.storage.updateSessionQuality(sessionId, quality.score, fp);

    return {
      messageCount: session?.message_count ?? 0,
      tokenEstimate: session?.token_estimate ?? 0,
      qualityScore: quality.score,
    };
  }

  addMessage(sessionId: string, msg: RecordedMessage): string {
    if (!this.activeSessions.has(sessionId)) {
      throw new Error(`Session ${sessionId} is not active or does not exist`);
    }

    return this.storage.addMessage(sessionId, msg.role, msg.content, {
      toolCallId: msg.toolCallId,
      toolName: msg.toolName,
    });
  }

  isActive(sessionId: string): boolean {
    return this.activeSessions.has(sessionId);
  }

  getActiveSessions(): string[] {
    return Array.from(this.activeSessions.keys());
  }
}
