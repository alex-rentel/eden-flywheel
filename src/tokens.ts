/**
 * Token estimation without external dependencies.
 *
 * Uses a character-based heuristic calibrated against tiktoken cl100k_base:
 * - English text: ~4 chars per token
 * - Code: ~3.5 chars per token (more special chars)
 * - CJK/Unicode: ~1.5 chars per token
 * - Whitespace-heavy: ~3 chars per token
 */

const CODE_PATTERN = /[{}()\[\];=<>!&|+\-*/\\@#$%^~`]/;
const CJK_PATTERN = /[\u3000-\u9fff\uac00-\ud7af\uff00-\uffef]/;

export function estimateTokens(text: string): number {
  if (!text) return 0;

  const len = text.length;
  if (len === 0) return 0;

  // Sample first 1000 chars to determine content type ratio
  const sample = text.slice(0, 1000);
  let codeChars = 0;
  let cjkChars = 0;

  for (const ch of sample) {
    if (CODE_PATTERN.test(ch)) codeChars++;
    if (CJK_PATTERN.test(ch)) cjkChars++;
  }

  const sampleLen = sample.length;
  const codeRatio = codeChars / sampleLen;
  const cjkRatio = cjkChars / sampleLen;

  // Weighted chars-per-token
  let charsPerToken = 4.0;
  if (cjkRatio > 0.2) charsPerToken = 1.5;
  else if (codeRatio > 0.15) charsPerToken = 3.5;

  return Math.max(1, Math.ceil(len / charsPerToken));
}

export function estimateTokensForMessages(
  messages: Array<{ role: string; content: string }>
): number {
  // Each message has ~4 tokens overhead (role, delimiters)
  let total = 0;
  for (const msg of messages) {
    total += 4 + estimateTokens(msg.content);
  }
  return total;
}
