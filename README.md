# eden-flywheel

**MCP server that turns your AI conversations into training data for better local models.**

The closed loop: use AI → capture trajectories → fine-tune → deploy better model → repeat.

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Node.js](https://img.shields.io/badge/node-%3E%3D18-brightgreen)](https://nodejs.org)
[![Tests](https://img.shields.io/badge/tests-139%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-90%25%2B-brightgreen)]()

## What it does

Every conversation with your AI agent is a potential training example. eden-flywheel captures these interactions, scores them for quality, deduplicates, formats them as SFT training data, and triggers fine-tuning runs — all through the Model Context Protocol.

```
You use Claude Code / Cursor / any MCP client
    │
    ▼
eden-flywheel captures the session (manual or auto)
    │
    ▼
Quality scoring: tool usage, error rate, turn depth
    │
    ▼
Deduplication via SHA-256 fingerprinting
    │
    ▼
Export as SFT training data (ChatML / Alpaca / ShareGPT / Raw)
    │
    ▼
Fine-tune via local MLX LoRA
    │
    ▼
Evaluate & promote the new adapter
    │
    ▼
You use the BETTER model tomorrow
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MCP Client                               │
│              (Claude Code, Cursor, etc.)                        │
└─────────────────┬───────────────────────────────────────────────┘
                  │ stdio (JSON-RPC)
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                     eden-flywheel MCP Server                    │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ 12 Tools │  │4 Resources│  │ Quality  │  │  AutoCapture  │  │
│  │          │  │           │  │ Pipeline │  │  (Claude Code) │  │
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘  └──────┬────────┘  │
│       │              │              │               │           │
│  ┌────▼──────────────▼──────────────▼───────────────▼────────┐  │
│  │                    Storage (SQLite + WAL)                  │  │
│  │              ~/.eden-flywheel/flywheel.db                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌───────────────────┐  ┌─────────────────────────────────────┐ │
│  │  Config + Logger   │  │  Training (mlx-lm subprocess)      │ │
│  │  ~/.eden-flywheel/ │  │  ~/.eden-models/                   │ │
│  └───────────────────┘  └─────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
npm install

# Build
npm run build

# Run the MCP server
npm start

# Run with verbose logging
npm run start:verbose

# Run tests
npm test

# Run tests with coverage
npm run test:coverage
```

### Add to Claude Code

Add to your `~/.claude/mcp.json`:

```json
{
  "mcpServers": {
    "eden-flywheel": {
      "command": "node",
      "args": ["/path/to/eden-flywheel/dist/index.js"]
    }
  }
}
```

## MCP Tools (12)

### Session Recording

| Tool | Description |
|---|---|
| `flywheel_record_start` | Start recording a session with optional metadata |
| `flywheel_record_stop` | Stop recording and compute quality score |
| `flywheel_list` | List all recorded sessions with stats |

### Auto-Capture (Claude Code)

| Tool | Description |
|---|---|
| `flywheel_autocapture` | Enable/disable automatic Claude Code session capture |
| `flywheel_ingest` | Ingest a raw Claude Code message into the active auto-capture session |
| `flywheel_cost` | Estimate API cost for a session based on token usage |

### Export & Quality

| Tool | Description |
|---|---|
| `flywheel_export` | Export sessions as training JSONL (ChatML/Alpaca/ShareGPT/Raw) |
| `flywheel_filter` | Export filtered data by quality, tool usage, errors, message count |
| `flywheel_status` | Show capture stats, quality distribution, data health |

### Training Pipeline

| Tool | Description |
|---|---|
| `flywheel_train` | Trigger MLX LoRA fine-tuning with exported data |
| `flywheel_eval` | Evaluate a trained adapter (loss metrics or size heuristic) |
| `flywheel_promote` | Deploy a trained adapter as the active model |

## MCP Resources (4)

| URI | Description |
|---|---|
| `flywheel://status` | Current capture status and session statistics |
| `flywheel://sessions` | List of all recorded sessions |
| `flywheel://latest-export` | Most recent export as ChatML JSONL |
| `flywheel://training-history` | Training run history and active adapter info |

## Export Formats

### ChatML (default)
Standard chat format for SFT. System prompt + user/assistant turns with tool calls inlined as XML.

```json
{"messages": [
  {"role": "system", "content": "You are a helpful coding assistant."},
  {"role": "user", "content": "Read the file src/main.ts"},
  {"role": "assistant", "content": "<tool_call>\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"src/main.ts\"}}\n</tool_call>"},
  {"role": "assistant", "content": "Here's the content of src/main.ts..."}
]}
```

### Alpaca
Instruction/input/output format. First user message becomes instruction, last plain assistant message becomes output.

```json
{"instruction": "Read the file src/main.ts", "input": "", "output": "Here's the content..."}
```

### ShareGPT
Multi-turn conversation format with human/gpt roles.

```json
{"conversations": [
  {"from": "human", "value": "Read the file src/main.ts"},
  {"from": "gpt", "value": "Here's the content..."}
]}
```

### Raw
Preserves all message roles including tool messages with XML-wrapped tool calls.

## Quality Scoring

Sessions are scored 0-1 based on:

| Signal | Weight | Description |
|---|---|---|
| Tool calls present | +0.15 | Sessions with tool usage are more valuable |
| Errors present | -0.15 | Error-heavy sessions are penalized |
| Multi-turn (4+) | +0.10 | Deeper conversations are preferred |
| Successful tool patterns | +0.05 each | Tool call → result → response chains |
| Good length (10-100 msgs) | +0.05 | Not too short, not too long |
| Role variety (3+ roles) | +0.05 | User + assistant + tool diversity |
| Very long (>50K tokens) | penalty | Extremely long sessions flagged |

Sessions are deduplicated via SHA-256 fingerprinting of normalized content.

## Configuration

Config file at `~/.eden-flywheel/config.json`:

```json
{
  "logLevel": "info",
  "dbPath": "~/.eden-flywheel/flywheel.db",
  "autoCapture": true,
  "defaultExportFormat": "chatml",
  "pricing": {
    "inputPerMToken": 3.0,
    "outputPerMToken": 15.0
  }
}
```

### CLI Flags

```bash
eden-flywheel --verbose          # Debug logging
eden-flywheel --quiet            # Error-only logging
eden-flywheel --log-level warn   # Specific log level
eden-flywheel --db-path /tmp/f.db  # Custom database path
```

## Project Structure

```
src/
├── index.ts          # MCP server entry point (12 tools, 4 resources)
├── storage.ts        # SQLite storage layer (WAL mode, foreign keys)
├── capture.ts        # Session recording lifecycle
├── export.ts         # Multi-format JSONL export + validation
├── quality.ts        # Quality scoring, fingerprinting, dedup, stats
├── tokens.ts         # Token estimation heuristic
├── training.ts       # MLX LoRA fine-tuning integration
├── autocapture.ts    # Claude Code auto-capture + cost estimation
├── errors.ts         # Typed error hierarchy
├── logger.ts         # Structured JSON logging (stderr)
└── config.ts         # File + CLI config resolution

tests/
├── storage.test.ts       # Storage CRUD and edge cases
├── capture.test.ts       # Session lifecycle
├── export.test.ts        # Multi-format export
├── server.test.ts        # MCP tool E2E via InMemoryTransport
├── quality.test.ts       # Scoring, dedup, fingerprinting
├── training.test.ts      # Training pipeline
├── autocapture.test.ts   # Auto-capture + Claude Code parsing
├── resources.test.ts     # MCP resource endpoints
├── hardening.test.ts     # Typed errors, logger, config
├── stress.test.ts        # Volume, concurrency, malformed input
└── coverage-boost.test.ts # Edge case coverage
```

## Testing

```bash
npm test                    # 139 tests
npm run test:coverage       # With V8 coverage (90%+ statements)
npm run typecheck           # TypeScript strict mode
```

**Coverage:** 90.7% statements, 86.3% branches, 93.1% functions, 92.3% lines.

## Training Targets

| Target | Hardware | Time | Use case |
|---|---|---|---|
| Local MLX LoRA | Mac (M1+) | ~2 hours | Quick iteration, small adapters |
| Great Lakes HPC | UMich A100s | ~8 hours | Full fine-tune, larger models |
| Cloud GPU | Any provider | Varies | If no local GPU |

## Why this matters

Claude Code, Cursor, OpenCode — they all use the same model every day. Your model never learns YOUR codebase, YOUR patterns, YOUR tools. The flywheel changes that:

- **Week 1:** Base Qwen3-4B, 97.5% tool accuracy
- **Week 2:** Fine-tuned on 200 of YOUR sessions → 98.5% on YOUR tasks
- **Week 4:** Fine-tuned on 1000 sessions → model knows your project structure
- **Month 3:** Model anticipates what you need before you ask

## Related

- [eden-memory](https://github.com/alex-rentel/eden-memory) — Persistent cross-session memory
- [eden-models](https://github.com/alex-rentel/eden-models) — Full training pipeline for Great Lakes HPC

## License

Apache 2.0
