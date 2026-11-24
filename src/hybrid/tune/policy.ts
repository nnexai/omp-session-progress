import type { Evidence } from '../runtime/evidence';

// Baseline: identical to the existing displayed ETA, without network access.
export const enabled = false;
export const instruction = `Classify the work that remains in an interactive coding prompt, not the difficulty of its original request. Reply with one label only: CLOSE (only handoff or final checks remain), BOUNDED (one identifiable implementation or verification step), MULTI (several substantive steps remain), EXTENDED (a substantial multi-stage implementation, investigation or verification campaign remains), or UNKNOWN (not identifiable). A running tool alone does not establish what follows it. A successful command is not necessarily final verification. Use current-plan evidence in preference to generic request categories. The local estimate and interval are minutes of active working time; you need not agree with them.`;
export type Workload = 'CLOSE'|'BOUNDED'|'MULTI'|'EXTENDED'|'UNKNOWN';
export function query(e:Evidence, lastS:number|null) {
  return e.elapsedS >= 60 && (lastS===null || e.elapsedS-lastS>=300);
}
export function packet(e:Evidence) {
  return JSON.stringify(e, (_k,v) => typeof v==='number'?Math.round(v*10)/10:v);
}
export function combine(e:Evidence, workload:Workload, ageS:number) {
  const factor = { CLOSE:0.65, BOUNDED:0.9, MULTI:1.35, EXTENDED:1.8, UNKNOWN:1 }[workload];
  return e.localMin * (1 + (factor-1)*Math.max(0,1-ageS/360));
}
