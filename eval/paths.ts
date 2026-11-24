import { homedir } from "node:os";
import { join } from "node:path";

/** Local-only eval data (the pseudonym → session file index); never inside the repository. */
export const EVAL_DIR = process.env.OMP_PROGRESS_EVAL_DIR ?? join(homedir(), ".cache", "omp-session-progress-eval");
export const SESSIONS_DIR = join(homedir(), ".omp", "agent", "sessions");
