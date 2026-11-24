import { evalHead, trainGBDTHead, type GBDTHead, type TrainHeadOptions } from '../../src/gbdt';
import type { Band, Prefix, Sample } from './types';

export interface MilestoneModel {
  direct(p: Prefix): number;
  oracle(p: Prefix, band: Band): number;
  oracleFinal(p: Prefix, final: boolean): number;
  forecast(p: Prefix): { eta: number; probabilities: number[] };
  prior: number[];
}

const bands: Band[] = [0, 1, 2, 3];
const stages = ['none', 'edit', 'verify', 'delegate', 'release'] as const;
const phases = ['idle', 'watching', 'planning', 'exploring', 'editing', 'verifying', 'delegating', 'waiting', 'retrying', 'wrapping up', 'done'] as const;
const categories = ['investigation', 'implementation', 'verification', 'endToEnd', 'release', 'review', 'deliver'] as const;
const families = ['execute', 'inspect', 'edit', 'delegate', 'plan', 'ask', 'other'] as const;
const headOptions: TrainHeadOptions = {
  anchor: 'anchor', nTrees: 70, lr: 0.055, maxDepth: 4, minLeaf: 18,
  l2: 10, colSample: 0.65, seed: 7,
};

const nonnegative = (n: number) => Number.isFinite(n) ? Math.max(0, n) : 0;
const logCount = (n: number) => Math.log1p(nonnegative(n));
const present = (values: readonly string[], value: string) => values.includes(value) ? 1 : 0;

// Only checkpoint-visible features are used by either head or the classifier.
function features(p: Prefix): number[] {
  const e = p.evidence;
  return [
    logCount(e.elapsedS), logCount(e.localMin), logCount(e.lowMin ?? 0), logCount(e.highMin ?? 0),
    Number(e.lowMin !== null), Number(e.highMin !== null),
    logCount(e.requestedItems), Number(e.continuation), Number(e.hasCurrentPlan),
    logCount(e.open), logCount(e.done), logCount(e.calls), logCount(e.errors),
    logCount(e.cleanExecAfterEdit), logCount(e.edits), logCount(e.delegates),
    logCount(p.episodesSeen), logCount(p.verificationsSeen), logCount(p.elapsedSinceStageS),
    Number(p.verifying), Number(e.slowTool), Number(e.wrapSignal),
    ...stages.map(s => Number(p.stage === s)),
    ...phases.map(s => Number(e.phase === s)),
    ...categories.map(s => present(e.request, s)),
    ...categories.map(s => present(e.openKinds, s)),
    ...categories.map(s => present(e.currentKinds, s)),
    ...families.map(s => present(e.recent, s)),
    ...families.map(s => e.running.filter(r => r.kind === s).length),
    logCount(Math.max(0, ...e.running.map(r => nonnegative(r.ageS)))),
    ...categories.map(s => Number(e.running.some(r => r.intentKinds.includes(s)))),
  ];
}

function anchor(p: Prefix): number {
  return Math.log(nonnegative(p.evidence.localMin) * 60 + 30);
}

function predict(head: GBDTHead, p: Prefix, x: number[]): number {
  // The training target is log(remaining seconds + 30), matching the existing
  // GBDT's composite objective. Clamp before exponentiation for finite output.
  const z = Math.max(0, Math.min(20, anchor(p) + evalHead(head, x)));
  return Math.max(0, (Math.exp(z) - 30) / 60);
}

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

export function fit(samples: Sample[]): MilestoneModel {
  const rows = samples.filter(s => Number.isFinite(s.weight) && s.weight > 0 && Number.isFinite(s.remainingMin) && s.remainingMin >= 0);
  if (!rows.length) throw new Error('Cannot fit milestones without positively weighted samples');
  const xs = rows.map(s => features(s.prefix));
  const targets = rows.map(s => Math.log(s.remainingMin * 60 + 30));
  const anchors = rows.map(s => anchor(s.prefix));
  const elapsed = rows.map(s => nonnegative(s.prefix.evidence.elapsedS));
  const weights = rows.map(s => s.weight);
  // The constant column preserves the exact feature-sampling sequence and
  // dimensionality; only the oracle receives the future milestone band.
  const directHead = trainGBDTHead(xs.map(x => [...x, 0]), targets, anchors, elapsed, weights, headOptions);
  const oracleHead = trainGBDTHead(xs.map((x, i) => [...x, rows[i]!.band]), targets, anchors, elapsed, weights, headOptions);
  const finalHead = trainGBDTHead(xs.map((x, i) => [...x, rows[i]!.band === 0 ? 0 : 1]), targets, anchors, elapsed, weights, headOptions);

  // Weighted multinomial logistic regression with standardized prefix features,
  // L2 shrinkage, fixed iterations and a weakly smoothed weighted class prior.
  const total = weights.reduce((a, b) => a + b, 0);
  const counts = bands.map(b => rows.reduce((n, s) => n + (s.band === b ? s.weight : 0), 0));
  const prior = counts.map(n => (n + total * 0.0025) / (total * 1.01));
  const d = xs[0]!.length;
  const mean = Array<number>(d).fill(0);
  const scale = Array<number>(d).fill(0);
  for (let i = 0; i < xs.length; i++) {
    for (let j = 0; j < d; j++) mean[j]! += weights[i]! / total * xs[i]![j]!;
  }
  for (let i = 0; i < xs.length; i++) {
    for (let j = 0; j < d; j++) scale[j]! += weights[i]! / total * (xs[i]![j]! - mean[j]!) ** 2;
  }
  for (let j = 0; j < d; j++) scale[j] = Math.max(0.1, Math.sqrt(scale[j]!));
  const standardize = (x: number[]) => x.map((v, j) => Math.max(-5, Math.min(5, (v - mean[j]!) / scale[j]!)));
  const standardized = xs.map(standardize);
  const coefficients = bands.map(() => Array<number>(d).fill(0));
  const intercepts = prior.map(Math.log);
  const probabilities = (x: number[]) => softmax(bands.map(b => {
    let z = intercepts[b]!;
    for (let j = 0; j < d; j++) z += coefficients[b]![j]! * x[j]!;
    return z;
  }));
  for (let iter = 0; iter < 160; iter++) {
    const gradient = bands.map(() => Array<number>(d).fill(0));
    const biasGradient = Array<number>(4).fill(0);
    for (let i = 0; i < rows.length; i++) {
      const x = standardized[i]!;
      const prob = probabilities(x);
      for (const b of bands) {
        const error = (prob[b]! - Number(rows[i]!.band === b)) * weights[i]! / total;
        biasGradient[b]! += error;
        for (let j = 0; j < d; j++) gradient[b]![j]! += error * x[j]!;
      }
    }
    for (const b of bands) {
      intercepts[b]! -= 0.08 * biasGradient[b]!;
      for (let j = 0; j < d; j++) coefficients[b]![j]! -= 0.08 * (gradient[b]![j]! + 0.05 * coefficients[b]![j]!);
    }
  }

  const direct = (p: Prefix) => predict(directHead, p, [...features(p), 0]);
  const oracle = (p: Prefix, band: Band) => predict(oracleHead, p, [...features(p), band]);
  const oracleFinal = (p: Prefix, final: boolean) => predict(finalHead, p, [...features(p), final ? 0 : 1]);
  return {
    direct, oracle, oracleFinal, prior,
    forecast(p) {
      const x = features(p);
      const probs = probabilities(standardize(x));
      const eta = bands.reduce((sum, b) => sum + probs[b]! * predict(oracleHead, p, [...x, b]), 0);
      return { eta, probabilities: probs };
    },
  };
}
