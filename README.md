# eden-flywheel

**MCP server that turns your AI conversations into training data for better local models.**

The closed loop: use AI → capture trajectories → fine-tune → deploy better model → repeat.

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

## What it does

Every conversation with your AI agent is a potential training example. eden-flywheel captures these interactions, filters for quality, formats them as training data, and can trigger fine-tuning runs.

```
You use Claude Code / Cursor / any agent
    │
    ▼
eden-flywheel captures the session
    │
    ▼
Filters: keep good tool calls, discard garbage
    │
    ▼
Formats as SFT training data (JSONL)
    │
    ▼
Triggers fine-tune on Great Lakes / local MLX
    │
    ▼
New model deployed via Ollama
    │
    ▼
You use the BETTER model tomorrow
```

## MCP Tools Exposed

| Tool | Description |
|---|---|
| `flywheel_status` | Show captured sessions, data quality stats |
| `flywheel_export` | Export recent sessions as training JSONL |
| `flywheel_filter` | Filter exported data by quality score |
| `flywheel_train` | Trigger a fine-tune run (local MLX or remote HPC) |
| `flywheel_eval` | Evaluate current model vs fine-tuned candidate |
| `flywheel_promote` | Deploy the fine-tuned model as new default |
| `trajectory_record` | Start recording current session |
| `trajectory_stop` | Stop recording and save |

## Why this matters

Claude Code, Cursor, OpenCode — they all use the same model every day. Your model never learns YOUR codebase, YOUR patterns, YOUR tools. The flywheel changes that:

- **Week 1:** Base Qwen3-4B, 97.5% tool accuracy
- **Week 2:** Fine-tuned on 200 of YOUR sessions → 98.5% on YOUR tasks
- **Week 4:** Fine-tuned on 1000 sessions → model knows your project structure
- **Month 3:** Model anticipates what you need before you ask

## Training Targets

| Target | Hardware | Time | Use case |
|---|---|---|---|
| Local MLX LoRA | Your Mac (M1+) | ~2 hours | Quick iteration, small adapters |
| Great Lakes HPC | UMich A100s | ~8 hours | Full fine-tune, larger models |
| Cloud GPU | Any provider | Varies | If no local GPU |

## Extracted from

- `eden/flywheel.py` (256 lines) — Flywheel orchestration
- `eden/training.py` (479 lines) — MLX LoRA/DPO training engine
- `eden/trajectory.py` (144 lines) — Session trajectory export

## Related

- [eden-memory](https://github.com/alex-rentel/eden-memory) — Persistent cross-session memory
- [eden-models](https://github.com/alex-rentel/eden-models) — Full training pipeline for Great Lakes HPC

## License

Apache 2.0
