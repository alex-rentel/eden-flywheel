/**
 * Typed error hierarchy for eden-flywheel.
 */

export class FlywheelError extends Error {
  constructor(message: string, public readonly code: string) {
    super(message);
    this.name = "FlywheelError";
  }
}

export class SessionNotFoundError extends FlywheelError {
  constructor(sessionId: string) {
    super(`Session not found: ${sessionId}`, "SESSION_NOT_FOUND");
    this.name = "SessionNotFoundError";
  }
}

export class SessionNotActiveError extends FlywheelError {
  constructor(sessionId: string) {
    super(`Session is not active: ${sessionId}`, "SESSION_NOT_ACTIVE");
    this.name = "SessionNotActiveError";
  }
}

export class ExportError extends FlywheelError {
  constructor(message: string) {
    super(message, "EXPORT_ERROR");
    this.name = "ExportError";
  }
}

export class TrainingError extends FlywheelError {
  constructor(message: string) {
    super(message, "TRAINING_ERROR");
    this.name = "TrainingError";
  }
}

export class StorageError extends FlywheelError {
  constructor(message: string) {
    super(message, "STORAGE_ERROR");
    this.name = "StorageError";
  }
}

export class ConfigError extends FlywheelError {
  constructor(message: string) {
    super(message, "CONFIG_ERROR");
    this.name = "ConfigError";
  }
}
