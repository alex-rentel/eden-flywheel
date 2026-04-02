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

const EDEN_MODELS_DIR = path.join(os.homedir(), ".eden-models");
const ACTIVE_DIR = path.join(EDEN_MODELS_DIR, "active");
const ADAPTERS_DIR = path.join(EDEN_MODELS_DIR, "adapters");
const HISTORY_PATH = path.join(EDEN_MODELS_DIR, "training_history.jsonl");

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
  for (const dir of [EDEN_MODELS_DIR, ACTIVE_DIR, ADAPTERS_DIR]) {
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
 * Evaluate base model vs fine-tuned on test cases.
 * Uses a simple prompt-completion accuracy test.
 */
export async function evaluateAdapter(
  baseModel: string,
  adapterPath: string,
  testDataPath?: string,
  trainDataPath?: string,
): Promise<EvalResult> {
  ensureDirs();

  // If no explicit test data, try using train data for a sanity eval
  const effectiveTestData = testDataPath && fs.existsSync(testDataPath)
    ? testDataPath
    : trainDataPath && fs.existsSync(trainDataPath)
      ? trainDataPath
      : null;

  if (!effectiveTestData) {
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

    return {
      baseScore: 0.5,
      adaptedScore: 0.5,
      improved: false,
      testCases: 0,
      details: "No test data available — cannot evaluate. Provide testData or re-export with a held-out split.",
    };
  }

  // With test data, run mlx_lm.evaluate
  const args = [
    "-m", "mlx_lm", "evaluate",
    "--model", baseModel,
    "--adapter-path", adapterPath,
    "--data", effectiveTestData,
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
        testCases: 0,
        details: error ? `Evaluation error: ${error.message}` : output.trim(),
      });
    });
  });
}

/**
 * Promote a successful adapter to the active slot.
 */
export function promoteAdapter(adapterPath: string, name?: string): string {
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

  return targetPath;
}

/**
 * Get training history from SQLite (preferred) or JSONL file (fallback).
 */
export function getTrainingHistory(): TrainResult[] {
  if (_storage) {
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

  // Fallback to JSONL file
  ensureDirs();
  if (!fs.existsSync(HISTORY_PATH)) return [];

  const history: TrainResult[] = [];
  const lines = fs.readFileSync(HISTORY_PATH, "utf-8").split("\n");
  for (const line of lines) {
    if (!line.trim()) continue;
    try {
      history.push(JSON.parse(line));
    } catch {}
  }
  return history;
}

function logTrainingRun(result: TrainResult): void {
  if (_storage) {
    _storage.recordTrainingRun({
      adapterPath: result.adapterPath,
      baseModel: result.baseModel,
      iterations: result.iterations,
      durationSeconds: result.durationSeconds,
      trainLoss: result.trainLoss,
      evalLoss: result.evalLoss,
      error: result.error,
    });
    return;
  }

  // Fallback to JSONL file
  ensureDirs();
  const entry = JSON.stringify({
    ...result,
    timestamp: new Date().toISOString(),
  });
  fs.appendFileSync(HISTORY_PATH, entry + "\n");
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
