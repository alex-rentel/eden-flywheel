/**
 * Configuration file support for training-flywheel.
 *
 * Reads from ~/.config/training-flywheel/config.json (JSON for simplicity, no yaml dep).
 */
import fs from "fs";
import path from "path";
import os from "os";
import { type LogLevel } from "./logger.js";

export interface FlywheelConfig {
  dbPath?: string;
  logLevel?: LogLevel;
  autoCapture?: boolean;
  defaultExportFormat?: "chatml" | "alpaca" | "sharegpt" | "raw";
  pricing?: {
    inputPerMillion?: number;
    outputPerMillion?: number;
  };
  training?: {
    defaultModel?: string;
    defaultIterations?: number;
    defaultBatchSize?: number;
    defaultLoraRank?: number;
  };
}

const CONFIG_DIR = path.join(os.homedir(), ".config", "training-flywheel");
const CONFIG_PATH = path.join(CONFIG_DIR, "config.json");

let cachedConfig: FlywheelConfig | null = null;

export function loadConfig(): FlywheelConfig {
  if (cachedConfig) return cachedConfig;

  try {
    if (fs.existsSync(CONFIG_PATH)) {
      const raw = fs.readFileSync(CONFIG_PATH, "utf-8");
      cachedConfig = JSON.parse(raw) as FlywheelConfig;
      return cachedConfig;
    }
  } catch (err) {
    process.stderr.write(`[training-flywheel] Warning: failed to parse config at ${CONFIG_PATH}: ${err instanceof Error ? err.message : String(err)}. Using defaults.\n`);
  }

  cachedConfig = {};
  return cachedConfig;
}

export function saveConfig(config: FlywheelConfig): void {
  fs.mkdirSync(CONFIG_DIR, { recursive: true });
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2) + "\n");
  cachedConfig = config;
}

export function getConfigPath(): string {
  return CONFIG_PATH;
}

export function resetConfigCache(): void {
  cachedConfig = null;
}

/**
 * Parse CLI args: --verbose, --quiet, --db-path, --log-level
 */
export function parseCliArgs(argv: string[]): Partial<FlywheelConfig> {
  const overrides: Partial<FlywheelConfig> = {};

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--verbose") {
      overrides.logLevel = "debug";
    } else if (arg === "--quiet") {
      overrides.logLevel = "error";
    } else if (arg === "--log-level" && i + 1 < argv.length) {
      overrides.logLevel = argv[++i] as LogLevel;
    } else if (arg === "--db-path" && i + 1 < argv.length) {
      overrides.dbPath = argv[++i];
    }
  }

  return overrides;
}

/**
 * Merge config file with CLI overrides.
 */
export function resolveConfig(cliOverrides?: Partial<FlywheelConfig>): FlywheelConfig {
  const fileConfig = loadConfig();
  return { ...fileConfig, ...cliOverrides };
}
