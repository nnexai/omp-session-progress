import type { Evidence } from '../../src/hybrid/runtime/evidence';

export type Band = 0 | 1 | 2 | 3; // 0, 1, 2–3, 4+ subsequent stage starts
export type Stage = 'none' | 'edit' | 'verify' | 'delegate' | 'release';
export interface Prefix {
  evidence: Evidence;
  stage: Stage;
  episodesSeen: number;
  verificationsSeen: number;
  elapsedSinceStageS: number;
  verifying: boolean;
}
export interface Sample { prefix: Prefix; band: Band; remainingMin: number; weight: number }
export interface Example { prefix:Prefix; gold:{ band:Band; todoBand:Band; remainingMin:number } }
export interface Trial { session:string; duration:number; missing:number; rows:Example[] }
