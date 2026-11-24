// Aggregate-only grading. Predictions and labels are transient and never emitted.
export interface Slice {
 count: number; error: number; baselineError: number; major: number; baselineMajor: number; over15: number; baselineOver15: number;
}
export const slice = (): Slice => ({ count: 0, error: 0, baselineError: 0, major: 0, baselineMajor: 0, over15: 0, baselineOver15: 0 });
export function observe(s: Slice, prediction: number, baseline: number, truth: number): void {
 for (const n of [prediction, baseline, truth]) if (!Number.isFinite(n) || n < 0) throw new Error('Invalid ETA or truth');
 const error = Math.abs(prediction - truth), baselineError = Math.abs(baseline - truth);
 s.count++; s.error += error; s.baselineError += baselineError;
 s.major += Number(error >= 15); s.baselineMajor += Number(baselineError >= 15); s.over15 += Number(prediction >= 15); s.baselineOver15 += Number(baseline >= 15);
}
export function mean(s: Slice, key: 'error' | 'baselineError' | 'major' | 'baselineMajor' | 'over15' | 'baselineOver15'): number {
 return s.count ? s[key] / s.count : 0;
}
export function score(primary: Slice, shorts: Slice[], slow: Slice, long30: boolean): Record<string, number> {
 if (!primary.count || shorts.some(s => !s.count)) throw new Error('Missing checkpoint coverage');
 const shortCount = shorts.reduce((n, s) => n + s.count, 0);
 return {
  mae_min: mean(primary, 'error'), baseline_mae_min: mean(primary, 'baselineError'),
  major_miss: mean(primary, 'major'), baseline_major_miss: mean(primary, 'baselineMajor'),
  primary_checkpoints: primary.count, primary_major_count: primary.major,
  short_catastrophe: shorts.length ? shorts.reduce((n, s) => n + mean(s, 'over15'), 0) / shorts.length : 0,
  short_baseline_catastrophe: shorts.length ? shorts.reduce((n, s) => n + mean(s, 'baselineOver15'), 0) / shorts.length : 0,
  short_prompts: shorts.length, short_checkpoints: shortCount,
  short_catastrophe_count: shorts.reduce((n, s) => n + s.over15, 0),
  slow_mae_min: mean(slow, 'error'), slow_baseline_mae_min: mean(slow, 'baselineError'), slow_checkpoints: slow.count,
  slow_major_miss: mean(slow, 'major'), slow_baseline_major_miss: mean(slow, 'baselineMajor'), slow_major_count: slow.major,
  long30: Number(long30), long30_mae_min: long30 ? mean(primary, 'error') : 0,
  long30_checkpoints: long30 ? primary.count : 0,
  long30_baseline_mae_min: long30 ? mean(primary, 'baselineError') : 0, long30_major_miss: long30 ? mean(primary, 'major') : 0,
 };
}

if (import.meta.main) {
 const assert = (ok: boolean) => { if (!ok) throw new Error('Grading self-check failed'); };
 const run = (kind: 'oracle' | 'repeat') => {
  const primary = slice(), short = slice(), slow = slice();
  for (const truth of [20, 10]) observe(primary, kind === 'oracle' ? truth : 30, 30, truth);
  observe(short, kind === 'oracle' ? 0.5 : 30, 30, 0.5);
  observe(slow, kind === 'oracle' ? 20 : 30, 30, 20);
  return score(primary, [short], slow, true);
 };
 const oracle = run('oracle'), repeat = run('repeat');
 assert(oracle.mae_min === 0 && oracle.short_catastrophe === 0 && oracle.slow_mae_min === 0);
 assert(repeat.mae_min === 15 && repeat.major_miss === 0.5 && repeat.short_catastrophe === 1);
 assert(JSON.stringify(repeat) === JSON.stringify(run('repeat')));
 const empty = slice(); observe(empty, 1, 1, 1);
 const noShort = score(empty, [], slice(), false);
 assert(noShort.short_catastrophe === 0 && noShort.short_checkpoints === 0);
 for (const invalid of [Number.NaN, null, -1, Infinity]) {
  let rejected = false; try { observe(slice(), invalid as number, 1, 1); } catch { rejected = true; }
  assert(rejected);
 }
 console.log('Grading self-check: oracle, null coverage, invalid ETA and repeat passed');
}
