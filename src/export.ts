/**
 * Export captured sessions as SFT training JSONL.
 *
 * Format:
 * {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}, ...]}
 */
import { Storage, type SessionRow, type MessageRow } from "./storage.js";

export interface SFTMessage {
  role: string;
  content: string;
}

export interface SFTExample {
  messages: SFTMessage[];
}

export class Exporter {
  private storage: Storage;

  constructor(storage: Storage) {
    this.storage = storage;
  }

  /**
   * Export sessions as SFT training JSONL string.
   * Each line is a complete conversation formatted for fine-tuning.
   */
  exportSessions(sessionIds?: string[]): string {
    const sessions = sessionIds
      ? sessionIds.map((id) => this.storage.getSession(id)).filter(Boolean) as SessionRow[]
      : this.storage.listSessions().filter((s) => s.stopped_at !== null);

    const lines: string[] = [];

    for (const session of sessions) {
      const example = this.sessionToSFT(session.id);
      if (example && example.messages.length >= 2) {
        lines.push(JSON.stringify(example));
      }
    }

    return lines.join("\n");
  }

  /**
   * Export with quality filters applied.
   */
  exportFiltered(opts?: {
    hasToolCalls?: boolean;
    noErrors?: boolean;
    minMessages?: number;
  }): string {
    const sessions = this.storage.getQualitySessions(opts);
    const lines: string[] = [];

    for (const session of sessions) {
      const example = this.sessionToSFT(session.id);
      if (example && example.messages.length >= 2) {
        lines.push(JSON.stringify(example));
      }
    }

    return lines.join("\n");
  }

  /**
   * Convert a single session to SFT training format.
   */
  private sessionToSFT(sessionId: string): SFTExample | null {
    const messages = this.storage.getMessages(sessionId);
    if (messages.length === 0) return null;

    const sftMessages: SFTMessage[] = [
      {
        role: "system",
        content: "You are a helpful AI coding assistant with access to tools.",
      },
    ];

    for (const msg of messages) {
      if (msg.role === "tool") {
        // Tool result — include as tool role
        sftMessages.push({
          role: "tool",
          content: msg.content,
        });
      } else if (msg.role === "assistant" && msg.tool_name) {
        // Tool call from assistant
        const toolCall = JSON.stringify({
          name: msg.tool_name,
          arguments: msg.content,
        });
        sftMessages.push({
          role: "assistant",
          content: `<tool_call>${toolCall}</tool_call>`,
        });
      } else {
        sftMessages.push({
          role: msg.role,
          content: msg.content,
        });
      }
    }

    return { messages: sftMessages };
  }
}
