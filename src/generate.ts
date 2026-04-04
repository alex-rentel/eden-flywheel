/**
 * Synthetic training data generation via OpenRouter (Qwen 3.6 Plus)
 * and validation via Anthropic API.
 */
import { Storage } from "./storage.js";
import { logger } from "./logger.js";

export interface GenerateConfig {
  count: number;
  toolSchemas?: string[];
  difficulty?: "easy" | "medium" | "hard";
}

export interface GenerateResult {
  generated: number;
  stored: number;
  errors: string[];
  batchId: string;
}

export interface ValidateResult {
  sampled: number;
  avgScore: number;
  flagged: number;
  scores: Array<{ sessionId: string; score: number; feedback: string }>;
}

const OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions";
const OPENROUTER_MODEL = "qwen/qwen3.6-plus-preview:free";

const DIFFICULTY_PROMPTS: Record<string, string> = {
  easy: "Generate a simple, single-step tool call conversation. The user asks for one thing, the assistant uses one tool to accomplish it.",
  medium: "Generate a multi-step tool call conversation. The user asks for something that requires 2-3 tool calls in sequence, with the assistant reasoning between calls.",
  hard: "Generate a complex tool call conversation. The user asks for something that requires 3+ tool calls, error handling, and iterative problem-solving. Include at least one tool call that returns unexpected results requiring the assistant to adapt.",
};

function buildSystemPrompt(toolSchemas: string[], difficulty: string): string {
  const difficultyGuide = DIFFICULTY_PROMPTS[difficulty] || DIFFICULTY_PROMPTS.medium;
  const schemaList = toolSchemas.length > 0
    ? `Available tools:\n${toolSchemas.map(s => `- ${s}`).join("\n")}`
    : `Available tools:\n- Read: Read a file. Parameters: { file_path: string }\n- Edit: Edit a file. Parameters: { file_path: string, old_string: string, new_string: string }\n- Bash: Run a shell command. Parameters: { command: string }\n- Grep: Search file contents. Parameters: { pattern: string, path?: string }\n- Glob: Find files by pattern. Parameters: { pattern: string }`;

  return `You are a training data generator for fine-tuning AI coding assistants. Generate realistic Claude Code conversations in ChatML JSON format.

${difficultyGuide}

${schemaList}

Output EXACTLY one JSON object per response with this structure:
{"messages": [{"role": "system", "content": "You are a helpful AI coding assistant with access to tools."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}

Rules:
- Tool calls must use: <tool_call>{"name": "ToolName", "arguments": "..."}</tool_call>
- Tool results come from role "tool"
- Conversations must be realistic coding tasks (debugging, refactoring, file operations)
- Include the assistant's reasoning text before each tool call
- End with the assistant summarizing what was done
- Output ONLY the JSON object, no markdown, no explanation`;
}

function parseGeneratedConversation(text: string): { messages: Array<{ role: string; content: string }> } | null {
  // Try to extract JSON from the response
  let json = text.trim();

  // Strip markdown code blocks if present
  const codeBlockMatch = json.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (codeBlockMatch) {
    json = codeBlockMatch[1].trim();
  }

  try {
    const parsed = JSON.parse(json);
    if (!parsed.messages || !Array.isArray(parsed.messages)) return null;

    // Validate structure
    const hasUser = parsed.messages.some((m: { role: string }) => m.role === "user");
    const hasAssistant = parsed.messages.some((m: { role: string }) => m.role === "assistant");
    if (!hasUser || !hasAssistant) return null;

    // Validate each message has role and content
    for (const msg of parsed.messages) {
      if (typeof msg.role !== "string" || typeof msg.content !== "string") return null;
    }

    return parsed;
  } catch {
    return null;
  }
}

/**
 * Generate synthetic training data using OpenRouter / Qwen 3.6 Plus.
 */
export async function generateTrainingData(
  storage: Storage,
  config: GenerateConfig,
  apiKey?: string,
): Promise<GenerateResult> {
  const key = apiKey || process.env.OPENROUTER_API_KEY;
  if (!key) {
    return {
      generated: 0,
      stored: 0,
      errors: ["OPENROUTER_API_KEY not set. Set it in env or flywheel config."],
      batchId: "",
    };
  }

  const difficulty = config.difficulty || "medium";
  const systemPrompt = buildSystemPrompt(config.toolSchemas || [], difficulty);
  const errors: string[] = [];
  let stored = 0;

  // Create a batch ID for tracking
  const batchId = `gen-${Date.now()}`;

  for (let i = 0; i < config.count; i++) {
    try {
      const response = await fetch(OPENROUTER_URL, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${key}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: OPENROUTER_MODEL,
          messages: [
            { role: "system", content: systemPrompt },
            { role: "user", content: `Generate training example ${i + 1}/${config.count}. Difficulty: ${difficulty}. Create a unique, realistic coding scenario.` },
          ],
          temperature: 0.9,
          max_tokens: 4096,
        }),
      });

      if (!response.ok) {
        const errText = await response.text();
        errors.push(`OpenRouter API error (${response.status}): ${errText.slice(0, 200)}`);
        continue;
      }

      const data = await response.json() as {
        choices?: Array<{ message?: { content?: string } }>;
        error?: { message?: string };
      };

      if (data.error) {
        errors.push(`OpenRouter error: ${data.error.message || JSON.stringify(data.error)}`);
        continue;
      }

      const content = data.choices?.[0]?.message?.content;
      if (!content) {
        errors.push(`Empty response for example ${i + 1}`);
        continue;
      }

      const parsed = parseGeneratedConversation(content);
      if (!parsed) {
        errors.push(`Failed to parse response ${i + 1} as valid ChatML`);
        continue;
      }

      // Store as a session
      const sessionId = storage.createSession({
        source: "synthetic",
        batchId,
        difficulty,
        generatedAt: new Date().toISOString(),
      });

      for (const msg of parsed.messages) {
        if (msg.role === "system") continue; // Skip system prompt
        storage.addMessage(sessionId, msg.role, msg.content);
      }

      storage.stopSession(sessionId);
      stored++;
    } catch (err) {
      errors.push(`Request ${i + 1} failed: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  logger.info("Synthetic data generation complete", { batchId, generated: config.count, stored, errors: errors.length });

  return {
    generated: config.count,
    stored,
    errors,
    batchId,
  };
}

/**
 * Validate training examples using Anthropic API (Claude).
 */
export async function validateBatch(
  storage: Storage,
  batchId: string,
  sampleSize: number,
  apiKey?: string,
): Promise<ValidateResult> {
  const key = apiKey || process.env.ANTHROPIC_API_KEY;
  if (!key) {
    return {
      sampled: 0,
      avgScore: 0,
      flagged: 0,
      scores: [],
    };
  }

  // Find sessions from this batch
  const allSessions = storage.listSessions();
  const batchSessions = allSessions.filter(s => {
    const meta = JSON.parse(s.metadata || "{}");
    return meta.batchId === batchId;
  });

  if (batchSessions.length === 0) {
    return { sampled: 0, avgScore: 0, flagged: 0, scores: [] };
  }

  // Sample N sessions
  const shuffled = [...batchSessions].sort(() => Math.random() - 0.5);
  const sampled = shuffled.slice(0, Math.min(sampleSize, shuffled.length));

  const scores: Array<{ sessionId: string; score: number; feedback: string }> = [];

  for (const session of sampled) {
    const messages = storage.getMessages(session.id);
    const conversation = messages.map(m => `${m.role}: ${m.content}`).join("\n\n");

    try {
      const response = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "x-api-key": key,
          "anthropic-version": "2023-06-01",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "claude-haiku-4-5-20251001",
          max_tokens: 512,
          messages: [
            {
              role: "user",
              content: `Rate this AI coding assistant conversation as training data on a scale of 1-5.

Criteria:
1. Accuracy: Are tool calls and responses realistic?
2. Format: Does it follow proper ChatML tool-calling conventions?
3. Naturalness: Does the conversation flow naturally?

Conversation:
${conversation}

Reply with ONLY a JSON object: {"score": <1-5>, "feedback": "<brief explanation>"}`,
            },
          ],
        }),
      });

      if (!response.ok) continue;

      const data = await response.json() as {
        content?: Array<{ type: string; text?: string }>;
      };

      const text = data.content?.[0]?.text || "";
      try {
        const result = JSON.parse(text);
        scores.push({
          sessionId: session.id,
          score: result.score || 0,
          feedback: result.feedback || "",
        });
      } catch {
        // Try to extract score from text
        const scoreMatch = text.match(/"score"\s*:\s*(\d)/);
        scores.push({
          sessionId: session.id,
          score: scoreMatch ? parseInt(scoreMatch[1]) : 0,
          feedback: text.slice(0, 200),
        });
      }
    } catch (err) {
      logger.warn("Validation request failed", {
        sessionId: session.id,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }

  const totalScore = scores.reduce((sum, s) => sum + s.score, 0);
  const avgScore = scores.length > 0 ? totalScore / scores.length : 0;
  const flagged = scores.filter(s => s.score <= 2).length;

  return {
    sampled: scores.length,
    avgScore: Math.round(avgScore * 100) / 100,
    flagged,
    scores,
  };
}

// Exported for testing
export { parseGeneratedConversation, buildSystemPrompt };
