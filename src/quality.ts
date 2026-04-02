/**
 * Quality scoring and deduplication for training data.
 */
import { createHash } from "crypto";
import { type MessageRow, type SessionRow } from "./storage.js";
import { estimateTokens } from "./tokens.js";

export interface QualityScore {
  score: number;           // 0.0 - 1.0
  reasons: string[];       // Why this score
  hasToolCalls: boolean;
  hasErrors: boolean;
  turnCount: number;
  tokenCount: number;
  avgTurnLength: number;
}

/**
 * Score a session's quality for training purposes.
 * Higher = better training example.
 */
export function scoreSession(session: SessionRow, messages: MessageRow[]): QualityScore {
  const reasons: string[] = [];
  let score = 0.5; // baseline

  const hasToolCalls = session.has_tool_calls > 0;
  const hasErrors = session.has_errors > 0;
  const turnCount = messages.length;
  const tokenCount = messages.reduce((sum, m) => sum + estimateTokens(m.content), 0);
  const avgTurnLength = turnCount > 0 ? tokenCount / turnCount : 0;

  // Tool usage is valuable for training
  if (hasToolCalls) {
    score += 0.15;
    reasons.push("contains_tool_calls");
  }

  // Errors reduce quality
  if (hasErrors) {
    score -= 0.15;
    reasons.push("contains_errors");
  }

  // Multi-turn conversations are more valuable than single-turn
  if (turnCount >= 4) {
    score += 0.1;
    reasons.push("multi_turn");
  } else if (turnCount <= 1) {
    score -= 0.2;
    reasons.push("too_short");
  }

  // Successful tool call patterns (tool call followed by tool result followed by assistant)
  let successfulToolPatterns = 0;
  for (let i = 0; i < messages.length - 2; i++) {
    if (
      messages[i].role === "assistant" && messages[i].tool_name &&
      messages[i + 1].role === "tool" &&
      messages[i + 2].role === "assistant" && !messages[i + 2].tool_name
    ) {
      successfulToolPatterns++;
    }
  }
  if (successfulToolPatterns > 0) {
    score += Math.min(0.15, successfulToolPatterns * 0.05);
    reasons.push(`successful_tool_patterns:${successfulToolPatterns}`);
  }

  // Reasonable conversation length (not too short, not too long)
  if (tokenCount >= 100 && tokenCount <= 8000) {
    score += 0.05;
    reasons.push("good_length");
  } else if (tokenCount > 8000) {
    score -= 0.05;
    reasons.push("very_long");
  }

  // Variety in roles (not just user/assistant ping-pong)
  const roles = new Set(messages.map((m) => m.role));
  if (roles.size >= 3) {
    score += 0.05;
    reasons.push("role_variety");
  }

  return {
    score: Math.max(0, Math.min(1, score)),
    reasons,
    hasToolCalls,
    hasErrors,
    turnCount,
    tokenCount,
    avgTurnLength: Math.round(avgTurnLength),
  };
}

/**
 * Compute a fingerprint for deduplication.
 * Uses first user message + last assistant message to detect near-duplicates.
 */
export function sessionFingerprint(messages: MessageRow[]): string {
  const userMessages = messages.filter((m) => m.role === "user");
  const assistantMessages = messages.filter((m) => m.role === "assistant");

  const firstUser = userMessages[0]?.content ?? "";
  const lastAssistant = assistantMessages[assistantMessages.length - 1]?.content ?? "";

  // Normalize: lowercase, collapse whitespace, take first 500 chars
  const normalize = (s: string) =>
    s.toLowerCase().replace(/\s+/g, " ").trim().slice(0, 500);

  const input = normalize(firstUser) + "||" + normalize(lastAssistant);
  return createHash("sha256").update(input).digest("hex").slice(0, 16);
}

/**
 * Deduplicate sessions by fingerprint.
 * Returns the IDs to keep (first occurrence wins).
 */
export function deduplicateSessions(
  sessions: Array<{ id: string; messages: MessageRow[] }>
): Set<string> {
  const seen = new Map<string, string>(); // fingerprint -> session id
  const keep = new Set<string>();

  for (const session of sessions) {
    const fp = sessionFingerprint(session.messages);
    if (!seen.has(fp)) {
      seen.set(fp, session.id);
      keep.add(session.id);
    }
  }

  return keep;
}

/**
 * Compute data statistics across sessions.
 */
export interface DataStats {
  totalSessions: number;
  totalTokens: number;
  avgTokensPerSession: number;
  avgTurnsPerSession: number;
  avgQualityScore: number;
  toolCallDistribution: Record<string, number>;
  turnHistogram: Record<string, number>;
  tokenHistogram: Record<string, number>;
  qualityDistribution: { high: number; medium: number; low: number };
}

export function computeDataStats(
  sessions: Array<{ session: SessionRow; messages: MessageRow[]; quality: QualityScore }>
): DataStats {
  const toolCounts: Record<string, number> = {};
  const turnBuckets: Record<string, number> = { "1-3": 0, "4-10": 0, "11-25": 0, "26-50": 0, "50+": 0 };
  const tokenBuckets: Record<string, number> = { "<100": 0, "100-500": 0, "500-2000": 0, "2000-8000": 0, "8000+": 0 };
  const qualityDist = { high: 0, medium: 0, low: 0 };

  let totalTokens = 0;
  let totalTurns = 0;
  let totalScore = 0;

  for (const { messages, quality } of sessions) {
    totalTokens += quality.tokenCount;
    totalTurns += quality.turnCount;
    totalScore += quality.score;

    // Tool call distribution
    for (const msg of messages) {
      if (msg.tool_name) {
        toolCounts[msg.tool_name] = (toolCounts[msg.tool_name] || 0) + 1;
      }
    }

    // Turn histogram
    const turns = quality.turnCount;
    if (turns <= 3) turnBuckets["1-3"]++;
    else if (turns <= 10) turnBuckets["4-10"]++;
    else if (turns <= 25) turnBuckets["11-25"]++;
    else if (turns <= 50) turnBuckets["26-50"]++;
    else turnBuckets["50+"]++;

    // Token histogram
    const tokens = quality.tokenCount;
    if (tokens < 100) tokenBuckets["<100"]++;
    else if (tokens < 500) tokenBuckets["100-500"]++;
    else if (tokens < 2000) tokenBuckets["500-2000"]++;
    else if (tokens < 8000) tokenBuckets["2000-8000"]++;
    else tokenBuckets["8000+"]++;

    // Quality distribution
    if (quality.score >= 0.7) qualityDist.high++;
    else if (quality.score >= 0.4) qualityDist.medium++;
    else qualityDist.low++;
  }

  const n = sessions.length || 1;

  return {
    totalSessions: sessions.length,
    totalTokens,
    avgTokensPerSession: Math.round(totalTokens / n),
    avgTurnsPerSession: Math.round((totalTurns / n) * 10) / 10,
    avgQualityScore: Math.round((totalScore / n) * 100) / 100,
    toolCallDistribution: toolCounts,
    turnHistogram: turnBuckets,
    tokenHistogram: tokenBuckets,
    qualityDistribution: qualityDist,
  };
}
