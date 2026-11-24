import type { Evidence } from '../../src/hybrid/runtime/evidence';

// Point and LunaInput are the only inputs visible to predictors. Truth and
// observed call outcomes belong to training/scoring, never to either forecast.
export interface Span { family: string; durationS: number; completed: boolean }
export interface Point {
  atMs: number;
  elapsedS: number;
  evidence: Evidence;
  tool: { id: number; family: string; ageS: number } | null;
}
export interface Truth { remainingS: number; toolLeftS: number | null }
export interface Prompt { points: Point[]; truth: Truth[]; missingPoints?: number }
export interface Session { id: string; spans: Span[]; prompts: Prompt[] }
export interface LunaInput {
  phase: Evidence['phase']; request: Evidence['request']; hasCurrentPlan: boolean;
  open: number; done: number; openKinds: Evidence['openKinds']; currentKinds: Evidence['currentKinds']; recent: Evidence['recent'];
  cleanExecAfterEdit: number; errors: number; wrapSignal: boolean; family: string; localPostMin: number;
}
export interface LunaAnswer { postMin: number | null; latencyMs: number; model: string; cost: number }
