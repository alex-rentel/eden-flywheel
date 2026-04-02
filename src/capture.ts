/**
 * Session capture — records messages for an active session.
 */
import { Storage } from "./storage.js";

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

  stop(sessionId: string): { messageCount: number; tokenEstimate: number } {
    if (!this.activeSessions.has(sessionId)) {
      throw new Error(`Session ${sessionId} is not active`);
    }

    this.storage.stopSession(sessionId);
    this.activeSessions.delete(sessionId);

    const session = this.storage.getSession(sessionId);
    return {
      messageCount: session?.message_count ?? 0,
      tokenEstimate: session?.token_estimate ?? 0,
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
