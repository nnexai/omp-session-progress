import type { Evidence } from '../../src/hybrid/runtime/evidence';
import type { Point, Session } from './types';

// Thresholds and distances are fixed before evaluating the holdout. A call is
// an independent survival observation; multiple calls from one session alone
// are not enough to establish a family-specific duration distribution.
const MIN_CALLS = 8;
const MIN_SESSIONS = 3;
const MAX_POST_MIN = 7 * 24 * 60;

type Duration = { durationS: number; completed: boolean; session: number };
type Features = {
  family: string; phase: Evidence['phase']; request: number; openKinds: number;
  currentKinds: number; recent: number; plan: boolean; wrap: boolean;
  elapsed: number; age: number; local: number; open: number; done: number;
  errors: number; clean: number;
};
type PostSample = { x: Features; postMin: number };
type PostPrompt = { session: number; samples: PostSample[] };

export type TimingModel = { durations: Duration[]; families: Map<string, Duration[]>; prompts: PostPrompt[] };

function broader(family: string): string {
  return family.startsWith('execute-') ? 'execute' : family;
}

// Evidence exposes only these literal categories and tool families. Bitsets
// preserve the original set/Jaccard semantics, including duplicate entries.
const bits: Record<Evidence['request'][number] | Evidence['recent'][number], number> = {
  investigation: 1 << 0, implementation: 1 << 1, verification: 1 << 2,
  endToEnd: 1 << 3, release: 1 << 4, review: 1 << 5, deliver: 1 << 6,
  execute: 1 << 7, inspect: 1 << 8, edit: 1 << 9, delegate: 1 << 10,
  plan: 1 << 11, ask: 1 << 12, other: 1 << 13,
};
function mask(values: ReadonlyArray<Evidence['request'][number] | Evidence['recent'][number]>): number {
  let result = 0;
  for (const value of values) result |= bits[value];
  return result;
}

function features(p: Point): Features {
  const e = p.evidence;
  return {
    family: p.tool!.family, phase: e.phase, request: mask(e.request), openKinds: mask(e.openKinds),
    currentKinds: mask(e.currentKinds), recent: mask(e.recent), plan: e.hasCurrentPlan, wrap: e.wrapSignal,
    elapsed: Math.log1p(Math.max(0, p.elapsedS) / 60),
    age: Math.log1p(Math.max(0, p.tool!.ageS) / 60),
    // This is a predictor of post-tool work, not an additive component.
    local: Math.log1p(Math.max(0, e.localMin)),
    open: Math.log1p(Math.max(0, e.open)), done: Math.log1p(Math.max(0, e.done)),
    errors: Math.log1p(Math.max(0, e.errors)), clean: Math.log1p(Math.max(0, e.cleanExecAfterEdit)),
  };
}

export function fitTiming(sessions: Session[]): TimingModel {
  const durations: Duration[] = [];
  const families = new Map<string, Duration[]>();
  const prompts: PostPrompt[] = [];
  sessions.forEach((session, sessionIndex) => {
    for (const span of session.spans) {
      if (!Number.isFinite(span.durationS) || span.durationS < 0) continue;
      const row = { durationS: span.durationS, completed: span.completed, session: sessionIndex };
      durations.push(row);
      const keys = new Set([span.family, broader(span.family)]);
      for (const key of keys) {
        let group = families.get(key);
        if (!group) families.set(key, group = []);
        group.push(row);
      }
    }
    for (const prompt of session.prompts) {
      const samples: PostSample[] = [];
      const seen = new Set<string>();
      for (let i = 0; i < prompt.points.length; i++) {
        const point = prompt.points[i]!;
        const truth = prompt.truth[i];
        if (!point.tool || point.tool.ageS < 30 || !truth || truth.toolLeftS == null ||
            !Number.isFinite(truth.toolLeftS) || !Number.isFinite(truth.remainingS) ||
            truth.toolLeftS < 0 || truth.remainingS < truth.toolLeftS) continue;
        // Sparse, prefix-defined logarithmic landmarks per call. The nearest
        // matching landmark per prompt gets one vote at prediction time.
        const bucket = `${point.tool.id}:${Math.floor(Math.log(Math.max(30, point.tool.ageS) / 30) / Math.log(1.5))}:${Math.floor(Math.log1p(Math.max(0, point.elapsedS) / 30) / Math.log(1.5))}`;
        if (seen.has(bucket)) continue;
        seen.add(bucket);
        samples.push({ x: features(point), postMin: Math.min(MAX_POST_MIN, (truth.remainingS - truth.toolLeftS) / 60) });
      }
      if (samples.length) prompts.push({ session: sessionIndex, samples });
    }
  });
  // Sort once, not once per query, for the conditional Kaplan–Meier scan.
  durations.sort((a, b) => a.durationS - b.durationS);
  for (const group of families.values()) group.sort((a, b) => a.durationS - b.durationS);
  return { durations, families, prompts };
}

// Condition on survival strictly beyond the observed age, then calculate the product-limit
// median. Censors remove risk only after events at the same time. If survival
// never reaches one half, the median is not identified: do not impute a finish.
function survivalMedian(rows: Duration[], ageS: number): { enough: boolean; leftS: number | null } {
  let lo = 0, hi = rows.length;
  while (lo < hi) {
    const mid = lo + Math.floor((hi - lo) / 2);
    if (rows[mid]!.durationS <= ageS) lo = mid + 1;
    else hi = mid;
  }
  const start = lo;
  const atRisk = rows.length - start;
  const sessions = new Set<number>();
  for (let i = start; i < rows.length; i++) sessions.add(rows[i]!.session);
  if (atRisk < MIN_CALLS || sessions.size < MIN_SESSIONS) return { enough: false, leftS: null };
  let risk = atRisk, survival = 1;
  for (let i = start; i < rows.length;) {
    const t = rows[i]!.durationS;
    let events = 0, removed = 0;
    while (i < rows.length && rows[i]!.durationS === t) {
      if (rows[i]!.completed) events++;
      removed++;
      i++;
    }
    survival *= 1 - events / risk;
    if (survival <= 0.5) return { enough: true, leftS: Math.max(0, t - ageS) };
    risk -= removed;
  }
  return { enough: true, leftS: null };
}

function setDistance(a: number, b: number): number {
  let union = a | b, intersection = a & b, unionCount = 0, intersectionCount = 0;
  while (union) { union &= union - 1; unionCount++; }
  while (intersection) { intersection &= intersection - 1; intersectionCount++; }
  return unionCount ? 1 - intersectionCount / unionCount : 0;
}

function distance(a: Features, b: Features): number {
  return (a.family === b.family ? 0 : broader(a.family) === broader(b.family) ? 0.4 : 1.2) +
    (a.phase === b.phase ? 0 : 0.6) + (a.plan === b.plan ? 0 : 0.3) + (a.wrap === b.wrap ? 0 : 0.4) +
    0.5 * setDistance(a.request, b.request) + 0.4 * setDistance(a.openKinds, b.openKinds) +
    0.4 * setDistance(a.currentKinds, b.currentKinds) + 0.2 * setDistance(a.recent, b.recent) +
    0.5 * Math.abs(a.elapsed - b.elapsed) + 0.3 * Math.abs(a.age - b.age) +
    0.6 * Math.abs(a.local - b.local) + 0.3 * Math.abs(a.open - b.open) +
    0.2 * Math.abs(a.done - b.done) + 0.15 * Math.abs(a.errors - b.errors) +
    0.15 * Math.abs(a.clean - b.clean);
}

function postEstimate(model: TimingModel, p: Point): number | null {
  const x = features(p);
  const neighbors: { distance: number; postMin: number; session: number }[] = [];
  for (const prompt of model.prompts) {
    let best: PostSample | undefined, bestDistance = Infinity;
    for (const sample of prompt.samples) {
      const d = distance(x, sample.x);
      if (d < bestDistance) { best = sample; bestDistance = d; }
    }
    if (best) neighbors.push({ distance: bestDistance, postMin: best.postMin, session: prompt.session });
  }
  neighbors.sort((a, b) => a.distance - b.distance);
  const selected = neighbors.slice(0, 40);
  if (selected.length < MIN_CALLS || new Set(selected.map(n => n.session)).size < MIN_SESSIONS) return null;
  // One vote per prompt, smoothly downweighted by feature distance. The
  // weighted median is robust to large post-tool durations and label outliers.
  const weighted = selected.map(n => ({ value: n.postMin, weight: 1 / (1 + n.distance) }));
  weighted.sort((a, b) => a.value - b.value);
  const half = weighted.reduce((sum, n) => sum + n.weight, 0) / 2;
  let cumulative = 0;
  for (const n of weighted) {
    cumulative += n.weight;
    if (cumulative >= half) return n.value;
  }
  return null;
}

export function timing(model: TimingModel, p: Point): { toolMin: number; postMin: number; totalMin: number } | null {
  if (!p.tool || !Number.isFinite(p.tool.ageS) || p.tool.ageS < 30) return null;
  const family = p.tool.family;
  const groups = [...new Set([family, broader(family)])].map(key => model.families.get(key) ?? []);
  groups.push(model.durations);
  for (const group of groups) {
    const result = survivalMedian(group, p.tool.ageS);
    if (!result.enough) continue;
    if (result.leftS == null) return null;
    const postMin = postEstimate(model, p);
    if (postMin == null) return null;
    const toolMin = result.leftS / 60;
    return { toolMin, postMin, totalMin: toolMin + postMin };
  }
  return null;
}
