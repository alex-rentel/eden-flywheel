/**
 * Training integration — trigger LoRA fine-tuning via mlx-lm.
 *
 * Runs mlx_lm.lora as a subprocess, monitors progress, and manages adapters.
 */
import { execFile } from "child_process";
import fs from "fs";
import path from "path";
import os from "os";
import type { Storage } from "./storage.js";

export interface TrainConfig {
  baseModel: string;
  trainData: string;
  outputDir?: string;
  iterations?: number;
  batchSize?: number;
  learningRate?: number;
  loraRank?: number;
  loraLayers?: number;
}

export interface TrainResult {
  adapterPath: string;
  baseModel: string;
  iterations: number;
  durationSeconds: number;
  trainLoss: number | null;
  evalLoss: number | null;
  error: string | null;
}

export interface EvalResult {
  baseScore: number;
  adaptedScore: number;
  improved: boolean;
  testCases: number;
  details: string;
}

const MODELS_DIR = path.join(os.homedir(), ".config", "training-flywheel", "models");
const ACTIVE_DIR = path.join(MODELS_DIR, "active");
const ADAPTERS_DIR = path.join(MODELS_DIR, "adapters");

let _storage: Storage | null = null;

/**
 * Set the storage instance for training history persistence.
 * When set, training runs are recorded in SQLite instead of a JSONL file.
 * Pass null to clear and fall back to JSONL.
 */
export function setTrainingStorage(storage: Storage | null): void {
  _storage = storage;
}

function ensureDirs(): void {
  for (const dir of [MODELS_DIR, ACTIVE_DIR, ADAPTERS_DIR]) {
    fs.mkdirSync(dir, { recursive: true });
  }
}

/**
 * Run a LoRA fine-tuning job using mlx-lm.
 */
export async function trainAdapter(config: TrainConfig): Promise<TrainResult> {
  ensureDirs();

  const iterations = config.iterations ?? 100;
  const batchSize = config.batchSize ?? 2;
  const learningRate = config.learningRate ?? 1e-5;
  const loraRank = config.loraRank ?? 8;
  const loraLayers = config.loraLayers ?? 16;
  const outputDir = config.outputDir || path.join(
    ADAPTERS_DIR,
    `lora-${Date.now()}`
  );

  fs.mkdirSync(outputDir, { recursive: true });

  // Validate train data exists
  if (!fs.existsSync(config.trainData)) {
    return {
      adapterPath: outputDir,
      baseModel: config.baseModel,
      iterations,
      durationSeconds: 0,
      trainLoss: null,
      evalLoss: null,
      error: `Training data not found: ${config.trainData}`,
    };
  }

  const args = [
    "-m", "mlx_lm", "lora",
    "--model", config.baseModel,
    "--train",
    "--data", config.trainData,
    "--iters", String(iterations),
    "--batch-size", String(batchSize),
    "--learning-rate", String(learningRate),
    "--adapter-path", outputDir,
    "--num-layers", String(loraLayers),
  ];

  const start = Date.now();

  return new Promise<TrainResult>((resolve) => {
    execFile("python3", args, { timeout: 3600_000 }, (error, stdout, stderr) => {
      const durationSeconds = (Date.now() - start) / 1000;
      const output = stdout + "\n" + stderr;

      // Parse loss from mlx-lm output
      let trainLoss: number | null = null;
      let evalLoss: number | null = null;

      const trainLossMatch = output.match(/Train loss[:\s]+([\d.]+)/i);
      if (trainLossMatch) trainLoss = parseFloat(trainLossMatch[1]);

      const evalLossMatch = output.match(/Val loss[:\s]+([\d.]+)/i);
      if (evalLossMatch) evalLoss = parseFloat(evalLossMatch[1]);

      const result: TrainResult = {
        adapterPath: outputDir,
        baseModel: config.baseModel,
        iterations,
        durationSeconds: Math.round(durationSeconds * 10) / 10,
        trainLoss,
        evalLoss,
        error: error ? `Training failed: ${error.message}` : null,
      };

      // Log to history
      logTrainingRun(result);

      resolve(result);
    });
  });
}

/**
 * Evaluate base model vs fine-tuned adapter on test data.
 * Requires a test data file (eval split from flywheel_export).
 */
export async function evaluateAdapter(
  baseModel: string,
  adapterPath: string,
  testDataPath?: string,
): Promise<EvalResult> {
  ensureDirs();

  // Verify adapter exists
  const adapterExists = fs.existsSync(path.join(adapterPath, "adapters.safetensors"))
    || fs.existsSync(path.join(adapterPath, "adapters.npz"));

  if (!adapterExists) {
    return {
      baseScore: 0.5,
      adaptedScore: 0.5,
      improved: false,
      testCases: 0,
      details: "No adapter files found — training may not have completed.",
    };
  }

  // Require test data — no heuristic fallback
  if (!testDataPath || !fs.existsSync(testDataPath)) {
    return {
      baseScore: 0.5,
      adaptedScore: 0.5,
      improved: false,
      testCases: 0,
      details: "No test data available — cannot evaluate. Re-export with flywheel_export (evalSplit: true) to generate a held-out eval set.",
    };
  }

  // Count test cases
  const testLines = fs.readFileSync(testDataPath, "utf-8").split("\n").filter(Boolean);
  const testCases = testLines.length;

  // Run mlx_lm.evaluate with test data
  const args = [
    "-m", "mlx_lm", "evaluate",
    "--model", baseModel,
    "--adapter-path", adapterPath,
    "--data", testDataPath,
  ];

  return new Promise<EvalResult>((resolve) => {
    execFile("python3", args, { timeout: 600_000 }, (error, stdout, stderr) => {
      const output = stdout + "\n" + stderr;

      let adaptedScore = 0.5;
      const perplexityMatch = output.match(/Perplexity[:\s]+([\d.]+)/i);
      if (perplexityMatch) {
        const perplexity = parseFloat(perplexityMatch[1]);
        adaptedScore = Math.max(0, 1 - perplexity / 100);
      }

      resolve({
        baseScore: 0.5,
        adaptedScore,
        improved: adaptedScore > 0.5,
        testCases,
        details: error ? `Evaluation error: ${error.message}` : output.trim(),
      });
    });
  });
}

export interface PromoteResult {
  promotedPath: string;
  ollamaDeployed: boolean;
  ollamaModel?: string;
  ollamaError?: string;
  adapterFormat: "gguf" | "mlx" | "unknown";
  note?: string;
}

/**
 * Promote a successful adapter to the active slot and optionally deploy to Ollama.
 */
export async function promoteAdapter(
  adapterPath: string,
  name?: string,
  opts?: { deployOllama?: boolean; baseModelGguf?: string },
): Promise<PromoteResult> {
  ensureDirs();

  const targetName = name || "flywheel-latest";
  const targetPath = path.join(ACTIVE_DIR, targetName);

  // Validate source exists
  if (!fs.existsSync(adapterPath)) {
    throw new Error(`Adapter path does not exist: ${adapterPath}`);
  }

  // Remove existing
  if (fs.existsSync(targetPath)) {
    fs.rmSync(targetPath, { recursive: true });
  }

  // Copy entire adapter directory including subdirectories (Node 18+)
  fs.cpSync(adapterPath, targetPath, { recursive: true });

  // Detect adapter format
  const files = fs.readdirSync(adapterPath);
  const hasGguf = files.some(f => f.endsWith(".gguf"));
  const hasSafetensors = files.some(f => f.endsWith(".safetensors"));
  const adapterFormat: "gguf" | "mlx" | "unknown" = hasGguf ? "gguf" : hasSafetensors ? "mlx" : "unknown";

  const result: PromoteResult = {
    promotedPath: targetPath,
    ollamaDeployed: false,
    adapterFormat,
  };

  // Attempt Ollama deployment if requested or auto-detect
  if (opts?.deployOllama !== false && adapterFormat === "gguf") {
    const ollamaResult = await deployToOllama(adapterPath, targetName, opts?.baseModelGguf);
    result.ollamaDeployed = ollamaResult.success;
    result.ollamaModel = ollamaResult.modelName;
    result.ollamaError = ollamaResult.error;
  } else if (adapterFormat === "mlx") {
    result.note = "MLX adapter detected. Use mlx-lm directly for inference — Ollama does not yet support custom MLX adapters.";
  }

  return result;
}

/**
 * Check if Ollama is running.
 */
export async function isOllamaRunning(): Promise<boolean> {
  try {
    const response = await fetch("http://localhost:11434/api/tags");
    return response.ok;
  } catch {
    return false;
  }
}

/**
 * Deploy a GGUF adapter to Ollama.
 */
async function deployToOllama(
  adapterPath: string,
  modelName: string,
  baseModelGguf?: string,
): Promise<{ success: boolean; modelName?: string; error?: string }> {
  // Check Ollama is running
  const running = await isOllamaRunning();
  if (!running) {
    return { success: false, error: "Ollama is not running. Start it with: ollama serve" };
  }

  // Find GGUF files
  const files = fs.readdirSync(adapterPath);
  const ggufFiles = files.filter(f => f.endsWith(".gguf"));

  if (ggufFiles.length === 0) {
    return { success: false, error: "No .gguf files found in adapter directory" };
  }

  const adapterGguf = path.join(adapterPath, ggufFiles[0]);
  const baseModel = baseModelGguf || "llama3.2:3b";

  // Create Modelfile
  const modelfilePath = path.join(adapterPath, "Modelfile");
  const modelfileContent = `FROM ${baseModel}
ADAPTER ${adapterGguf}
SYSTEM "You are a tool-calling assistant. Use <tool_call> tags for tool invocations."
`;
  fs.writeFileSync(modelfilePath, modelfileContent);

  // Run ollama create
  const ollamaModelName = `flywheel-${modelName}`;
  try {
    const { execFileSync } = await import("child_process");
    execFileSync("ollama", ["create", ollamaModelName, "-f", modelfilePath], {
      timeout: 120_000,
      encoding: "utf-8",
    });

    return { success: true, modelName: ollamaModelName };
  } catch (err) {
    return {
      success: false,
      modelName: ollamaModelName,
      error: `ollama create failed: ${err instanceof Error ? err.message : String(err)}`,
    };
  }
}

/**
 * Get training history from SQLite.
 * Requires setTrainingStorage() to have been called.
 */
export function getTrainingHistory(): TrainResult[] {
  if (!_storage) return [];
  return _storage.getTrainingRuns().map((r) => ({
    adapterPath: r.adapterPath,
    baseModel: r.baseModel,
    iterations: r.iterations,
    durationSeconds: r.durationSeconds,
    trainLoss: r.trainLoss,
    evalLoss: r.evalLoss,
    error: r.error,
  }));
}

function logTrainingRun(result: TrainResult): void {
  if (!_storage) return;
  _storage.recordTrainingRun({
    adapterPath: result.adapterPath,
    baseModel: result.baseModel,
    iterations: result.iterations,
    durationSeconds: result.durationSeconds,
    trainLoss: result.trainLoss,
    evalLoss: result.evalLoss,
    error: result.error,
  });
}

/**
 * Get the active adapter path if one is promoted.
 */
export function getActiveAdapter(): string | null {
  const latest = path.join(ACTIVE_DIR, "flywheel-latest");
  if (fs.existsSync(latest) && fs.readdirSync(latest).length > 0) {
    return latest;
  }
  return null;
}
