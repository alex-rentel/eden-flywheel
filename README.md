# training-flywheel

**A standalone MCP server for Claude Code that captures your coding sessions as fine-tuning data for local models.**

The closed loop: use AI → capture → fine-tune → deploy → repeat.

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Node.js](https://img.shields.io/badge/node-%3E%3D18-brightgreen)](https://nodejs.org)
[![Tests](https://img.shields.io/badge/tests-174%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-88%25%2B-brightgreen)]()

## What it does

Every conversation with your AI agent is a potential training example. training-flywheel captures these interactions, scores them for quality, deduplicates, formats them as SFT training data, and triggers fine-tuning runs — all through the Model Context Protocol.

```
You use Claude Code daily
    │
    ▼
flywheel captures the session (manual or auto)
    │
    ▼
Quality scoring: tool usage, error rate, turn depth
    │
    ▼
Deduplication via SHA-256 fingerprinting
    │
    ▼
Export as SFT training data (ChatML / Alpaca / ShareGPT / Raw)
    │                                      │
    ▼                                      ▼
Generate synthetic data             10% held-out eval split
(Qwen 3.6 Plus via OpenRouter)
    │
    ▼
Fine-tune via local MLX LoRA
    │
    ▼
Evaluate on held-out eval set
    │
    ▼
Promote & deploy to Ollama
    │
    ▼
You use the BETTER model tomorrow
```

## Installation

### Add to Claude Code

Add to your MCP configuration (`~/.claude/mcp.json` or equivalent):

```json
{
  "mcpServers": {
    "training-flywheel": {
      "command": "npx",
      "args": ["-y", "training-flywheel"],
      "env": {
        "FLYWHEEL_DB": "~/.config/training-flywheel/flywheel.db",
        "OPENROUTER_API_KEY": "your-key-here"
      }
    }
  }
}
```

Or install locally:

```bash
git clone https://github.com/alex-rentel/training-flywheel.git
cd training-flywheel
npm install && npm run build
```

Then point MCP at the built entry:

```json
{
  "mcpServers": {
    "training-flywheel": {
      "command": "node",
      "args": ["/path/to/training-flywheel/dist/index.js"]
    }
  }
}
```

## Usage: The Full Loop

### 1. Capture sessions

```
flywheel_autocapture { enabled: true }   // auto-capture all messages
flywheel_record_start {}                 // or manual recording
```

### 2. Export training data

```
flywheel_export { evalSplit: true, outputPath: "/tmp/train.jsonl" }
// Writes train.jsonl + train.eval.jsonl (10% held-out)
```

### 3. Generate synthetic data (optional)

```
flywheel_generate { count: 50, difficulty: "medium" }
// Uses Qwen 3.6 Plus (free on OpenRouter) to create more training examples
```

### 4. Train a LoRA adapter

```
flywheel_train {
  baseModel: "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
  trainData: "/tmp/train.jsonl"
}
```

### 5. Evaluate

```
flywheel_eval {
  baseModel: "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
  adapterPath: "~/.config/training-flywheel/models/adapters/lora-...",
  testData: "/tmp/train.eval.jsonl"
}
```

### 6. Promote & Deploy

```
flywheel_promote { adapterPath: "..." }
// GGUF adapters auto-deploy to Ollama
// MLX adapters use mlx-lm directly
```

## MCP Tools (14)

### Session Recording

| Tool | Description |
|---|---|
| `flywheel_record_start` | Start recording a session with optional metadata |
| `flywheel_record_stop` | Stop recording and compute quality score |
| `flywheel_list` | List all recorded sessions with stats |

### Auto-Capture (Claude Code)

| Tool | Description |
|---|---|
| `flywheel_autocapture` | Enable/disable automatic session capture |
| `flywheel_ingest` | Ingest a raw message into the active auto-capture session |
| `flywheel_cost` | Estimate API cost for a session |

### Export & Quality

| Tool | Description |
|---|---|
| `flywheel_export` | Export sessions as training JSONL with optional 10% eval split |
| `flywheel_filter` | Export filtered data by quality, tool usage, errors |
| `flywheel_status` | Show capture stats, quality distribution, data health |

### Synthetic Data

| Tool | Description |
|---|---|
| `flywheel_generate` | Generate synthetic training data via Qwen 3.6 Plus (free OpenRouter) |
| `flywheel_validate` | Score synthetic data quality using Claude |

### Training Pipeline

| Tool | Description |
|---|---|
| `flywheel_train` | Trigger MLX LoRA fine-tuning |
| `flywheel_eval` | Evaluate adapter on held-out test data |
| `flywheel_promote` | Deploy adapter to Ollama or active slot |

## MCP Resources (4)

| URI | Description |
|---|---|
| `flywheel://status` | Current capture status and statistics |
| `flywheel://sessions` | All recorded sessions |
| `flywheel://latest-export` | Most recent ChatML JSONL export |
| `flywheel://training-history` | Training runs and active adapter |

## Supported Training Backends

| Backend | Hardware | Use case |
|---|---|---|
| **Local MLX LoRA** (mlx-lm) | Mac M1+ | Quick iteration, small adapters |
| **Local nanochat** (mlx-nanochat) | Mac M1+ | Full SFT with MLX |
| **Remote HPC** (eden-models + Slurm) | GPU cluster | Large-scale fine-tuning |

## Supported Deployment Targets

| Target | Notes |
|---|---|
| **Ollama** | Auto-deploys GGUF adapters via Modelfile |
| **mlx-lm** | Direct inference with MLX safetensors adapters |

## Configuration

Config file at `~/.config/training-flywheel/config.json`:

```json
{
  "logLevel": "info",
  "dbPath": "~/.config/training-flywheel/flywheel.db",
  "autoCapture": true,
  "defaultExportFormat": "chatml"
}
```

### CLI Flags

```bash
training-flywheel --verbose            # Debug logging
training-flywheel --quiet              # Error-only logging
training-flywheel --log-level warn     # Specific log level
training-flywheel --db-path /tmp/f.db  # Custom database path
```

### Environment Variables

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | Required for `flywheel_generate` (synthetic data) |
| `ANTHROPIC_API_KEY` | Required for `flywheel_validate` (quality scoring) |
| `FLYWHEEL_DB` | Override database path |

## Quality Scoring

Sessions are scored 0-1 based on:

| Signal | Weight | Description |
|---|---|---|
| Tool calls present | +0.15 | Sessions with tool usage are more valuable |
| Errors present | -0.15 | Error-heavy sessions are penalized |
| Multi-turn (4+) | +0.10 | Deeper conversations are preferred |
| Successful tool patterns | +0.05 each | Tool call → result → response chains |
| Good length | +0.05 | Not too short, not too long |
| Role variety (3+ roles) | +0.05 | User + assistant + tool diversity |

## Export Formats

- **ChatML** (default): Standard SFT format with system prompt + tool calls as XML
- **Alpaca**: Instruction/input/output for simple fine-tuning
- **ShareGPT**: Multi-turn human/gpt conversation format
- **Raw**: Preserves all roles including tool messages

## Related Projects

- [eden-models](https://github.com/alex-rentel/eden-models) — HPC training pipeline (Slurm configs, data preprocessing)
- [mlx-nanochat](https://github.com/alex-rentel/mlx-nanochat) — Local MLX fine-tuning
- [mlx-turboquant](https://github.com/alex-rentel/mlx-turboquant) — KV cache compression for deployed models

## License

Apache 2.0
