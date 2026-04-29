# Security policy

## Reporting a vulnerability

Email **alex@renaissanceintelligence.ai** with the details. Avoid filing a public GitHub issue for anything you believe could be exploited — open a private channel first.

A useful report includes:

- The version you reproduced against (`package.json` → `version`, plus the commit SHA).
- A minimal repro: relevant MCP messages, config file, the failure mode.
- Node.js version (`node --version`).

You should expect a first reply within a few days.

## Supported versions

Only the latest minor line gets security fixes. Pre-1.0 versions are unsupported.

## Scope

In scope: anything that lets crafted MCP input crash the server, leak data across sessions, write outside the configured storage directory (`~/.config/eden-flywheel/`), or execute arbitrary code via the LoRA adapter loading path. Path-traversal and command-injection cases are explicitly relevant — see `tests/hardening.test.ts` for the surface area we already cover.

Out of scope:

- Quality scoring being "wrong" on a particular conversation — that's a tuning issue, not a security issue. Open a public GitHub issue.
- Issues in the MCP SDK or in tools that consume eden-flywheel's output — report upstream.
- Anything that requires the attacker to already have write access to `~/.config/eden-flywheel/` or the user's npm install path.
