# Contributing to eden-flywheel

Thanks for your interest. eden-flywheel is a Model Context Protocol server that captures AI-agent conversations as training data, scores them, deduplicates, and triggers fine-tuning runs. This file is a quick orientation.

## Dev environment

Node.js 18+, npm. macOS / Linux / Windows.

```bash
git clone https://github.com/alex-rentel/eden-flywheel.git
cd eden-flywheel
npm install
```

## Running checks locally

```bash
npm run typecheck   # tsc --noEmit
npm run build       # tsc
npm test            # vitest run, 174 tests
npm run test:coverage  # with coverage
```

CI (`.github/workflows/ci.yml`) runs the same three on Node 18, 20, and 22.

## Layout

| Path | What lives here |
|---|---|
| `src/index.ts` | MCP server entry point |
| `src/config.ts` | Config file support (`~/.config/eden-flywheel/config.json`) |
| `src/storage.ts` | SQLite storage layer for sessions and quality scores |
| `src/training.ts` | LoRA adapter management |
| `src/logger.ts` | Structured JSON logging |
| `src/errors.ts` | Typed error hierarchy |
| `tests/*.test.ts` | Vitest tests, including a `hardening.test.ts` with security/path-traversal coverage |

## Style

- TypeScript strict mode (see `tsconfig.json`).
- 2-space indent, LF endings (matches `.editorconfig`).
- Tests run with `vitest`; prefer co-locating tests under `tests/` rather than alongside source.

## Releases

This is an MCP server, not a library. Releases happen by:

1. Bump `version` in `package.json`.
2. Tag the commit (`git tag -a v0.x.y -m "..."`) and push.
3. Optionally publish to npm (`npm publish` — requires npm credentials and 2FA).

## Issues / questions

Open an issue at https://github.com/alex-rentel/eden-flywheel/issues.
