import type { Input, Row } from './data';

// Absolute remaining-time deadlines in seconds. Each leaf estimates all of
// these events jointly, so its CDF is monotone without post-hoc calibration.
const DEADLINES = [15, 30, 60, 120, 240, 480, 900, 1800, 3600, 7200, 14400, 28800];
const TREE_COUNT = 5;
const MAX_DEPTH = 3;
const MIN_LEAF = 30;
const PRIOR_WEIGHT = 8;

type Leaf = { cdf: number[] };
type Branch = { feature: number; cut: number; left: Node; right: Node };
type Node = Leaf | Branch;
export type M = { trees: Node[] };
type Example = { row: Row; weight: number; bucket: number };

// The first three indices are explicit anchors; the rest address x. No baseline
// estimate is ever used to train either the partitions or their distributions.
function value(r: Input, feature: number): number {
 const v = feature === 0 ? Math.log1p(Math.max(0, r.e))
  : feature === 1 ? Math.log1p(Math.max(0, r.tools))
  : feature === 2 ? Number(r.todo) : r.x[feature - 3];
 return Number.isFinite(v) ? v : 0;
}

function histogram(rows: Example[]): number[] {
 const bins = Array<number>(DEADLINES.length + 1).fill(0);
 for (const r of rows) bins[r.bucket] += r.weight;
 return bins;
}

function cdf(bins: number[], prior?: number[]): number[] {
 const total = bins.reduce((a, b) => a + b, 0);
 let sum = 0;
 return DEADLINES.map((_, i) => {
  sum += bins[i];
  return prior ? (sum + PRIOR_WEIGHT * prior[i]) / (total + PRIOR_WEIGHT) : sum / total;
 });
}

// Sum of binary Brier impurities over deadlines, omitting the constant term.
function score(bins: number[], total: number): number {
 if (total <= 0) return 0;
 let cumulative = 0;
 let result = 0;
 for (let i = 0; i < DEADLINES.length; i++) {
  cumulative += bins[i];
  result += cumulative * cumulative / total;
 }
 return result;
}

function grow(rows: Example[], prior: number[], features: number[], depth: number): Node {
 const all = histogram(rows);
 const leaf: Leaf = { cdf: cdf(all, prior) };
 if (depth >= MAX_DEPTH || rows.length < MIN_LEAF * 2) return leaf;
 const total = all.reduce((a, b) => a + b, 0);
 const parentScore = score(all, total);
 let bestScore = parentScore;
 let bestFeature = -1;
 let bestCut = 0;

 for (const feature of features) {
  const ordered = rows.map(r => ({ r, v: value(r.row, feature) })).sort((a, b) => a.v - b.v);
  const left = Array<number>(all.length).fill(0);
  const right = all.slice();
  let leftWeight = 0;
  // Evaluate at most seven equal-count boundaries, moving a single sweep.
  let cursor = 0;
  for (let q = 1; q < 8; q++) {
   const end = Math.floor(ordered.length * q / 8);
   while (cursor < end) {
    const item = ordered[cursor++].r;
    left[item.bucket] += item.weight;
    right[item.bucket] -= item.weight;
    leftWeight += item.weight;
   }
   if (end < MIN_LEAF || ordered.length - end < MIN_LEAF || ordered[end - 1].v === ordered[end].v) continue;
   const s = score(left, leftWeight) + score(right, total - leftWeight);
   if (s > bestScore + 1e-9) {
    bestScore = s;
    bestFeature = feature;
    bestCut = ordered[end - 1].v;
   }
  }
 }
 if (bestFeature < 0) return leaf;
 const left: Example[] = [];
 const right: Example[] = [];
 for (const row of rows) (value(row.row, bestFeature) <= bestCut ? left : right).push(row);
 return { feature: bestFeature, cut: bestCut,
  left: grow(left, prior, features, depth + 1), right: grow(right, prior, features, depth + 1) };
}

export function fit(rows: Row[]): M {
 if (!rows.length) return { trees: [] };
 // Temper dense long-prompt checkpoint sequences without discarding them.
 const counts = new Map<string, number>();
 for (const r of rows) {
  const key = `${r.sid}:${r.pid}`;
  counts.set(key, (counts.get(key) ?? 0) + 1);
 }
 const examples: Example[] = rows.map(row => {
  const found = DEADLINES.findIndex(h => row.y <= h);
  return { row, weight: 1 / Math.sqrt(counts.get(`${row.sid}:${row.pid}`)!),
   bucket: found < 0 ? DEADLINES.length : found };
 });
 const prior = cdf(histogram(examples));
 const width = rows[0].x.length;
 const trees: Node[] = [];
 for (let t = 0; t < TREE_COUNT; t++) {
  // A deterministic feature subspace makes the small ensemble diverse. The
  // step is coprime to 140, covering the complete production vector over trees.
  const features = [0, 1, 2];
  for (let i = 0; i < Math.min(28, width); i++) features.push(3 + ((t * 28 + i) * 37 % width));
  trees.push(grow(examples, prior, features, 0));
 }
 return { trees };
}

function median(c: number[]): number {
 let previousP = 0;
 let previousH = 0;
 for (let i = 0; i < DEADLINES.length; i++) {
  if (c[i] >= 0.5) {
   const fraction = (0.5 - previousP) / (c[i] - previousP);
   // Interpolate on a smoothed log-time scale, appropriate to wide tail bins.
   return 30 * Math.expm1(Math.log1p(previousH / 30) + fraction *
    (Math.log1p(DEADLINES[i] / 30) - Math.log1p(previousH / 30)));
  }
  previousP = c[i];
  previousH = DEADLINES[i];
 }
 // The median is beyond the observed grid; avoid inventing an unbounded tail.
 return DEADLINES[DEADLINES.length - 1];
}

export function predict(model: M, r: Input): number {
 const baseline = Number.isFinite(r.b) ? Math.max(0, r.b) : 0;
 if (!model.trees.length) return baseline;
 const combined = Array<number>(DEADLINES.length).fill(0);
 for (let node of model.trees) {
  while ('feature' in node) node = value(r, node.feature) <= node.cut ? node.left : node.right;
  for (let i = 0; i < combined.length; i++) combined[i] += node.cdf[i] / model.trees.length;
 }
 const estimate = median(combined);
 // Bound correction in log space, and protect very short baseline predictions
 // from a long-task tail dominating the blend. Long tasks retain a somewhat
 // larger upward correction than downward correction.
 const bounded = Math.max((baseline + 30) / 2, Math.min((baseline + 30) * 3, estimate + 30));
 const blend = baseline < 60 ? 0.1 : bounded > baseline + 30 ? 0.25 : 0.15;
 let result = Math.exp((1 - blend) * Math.log(baseline + 30) + blend * Math.log(bounded)) - 30;
 if (baseline < 60) result = Math.min(result, baseline + 15);
 return Number.isFinite(result) ? Math.max(0, result) : baseline;
}
