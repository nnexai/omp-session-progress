import { evalHead, trainGBDTHead, type GBDTHead } from '../../src/gbdt';
import type { Input, Row } from './data';

// The extra pseudo-turn and 30 seconds regularize the rate for zero/small
// future-turn counts. Clipping limits the influence of exceptional long tails.
const MAX_TURNS = 100;
const MAX_REMAINING = 8 * 60 * 60;
const MAX_PER_PROMPT = 12;

type Model = { turns: GBDTHead | null; rate: GBDTHead | null };

function features(r: Input): number[] {
	return [...r.x, Math.log1p(Math.max(0, r.e)), Math.log1p(Math.max(0, r.tools)), r.todo ? 1 : 0];
}

export function fit(rows: Row[]): Model {
	if (rows.length === 0) return { turns: null, rate: null };
	const prompts = new Map<string, Row[]>();
	for (const r of rows) {
		const key = `${r.sid}:${r.pid}`;
		const group = prompts.get(key);
		if (group) group.push(r);
		else prompts.set(key, [r]);
	}
	const X: number[][] = [];
	const turnTargets: number[] = [];
	const rateTargets: number[] = [];
	const weights: number[] = [];
	for (const group of prompts.values()) {
		// Evenly cover each prompt's observed checkpoints, irrespective of its length.
		group.sort((a, b) => a.e - b.e);
		const n = Math.min(MAX_PER_PROMPT, group.length);
		for (let i = 0; i < n; i++) {
			const r = group[n === 1 ? 0 : Math.round(i * (group.length - 1) / (n - 1))]!;
			const turns = Math.min(MAX_TURNS, Math.max(0, r.turns));
			const remaining = Math.min(MAX_REMAINING, Math.max(0, r.y));
			X.push(features(r));
			turnTargets.push(Math.log1p(turns));
			rateTargets.push(Math.log((remaining + 30) / (turns + 1)));
			weights.push(1 / n);
		}
	}
	const zero = X.map(() => 0);
	const options = {
		anchor: 'zero' as const, nTrees: 48, lr: 0.08, maxDepth: 3,
		minLeaf: 20, l2: 1, colSample: 0.8, seed: 1543,
		// A near-median quantile avoids the ETA-specific composite loss in trainGBDTHead.
		quantile: 0.49, maxBins: 32,
	};
	return {
		turns: trainGBDTHead(X, turnTargets, zero, zero, weights, options),
		rate: trainGBDTHead(X, rateTargets, zero, zero, weights, { ...options, seed: 2909 }),
	};
}

export function predict(model: Model, r: Input): number {
	const baseline = Number.isFinite(r.b) ? Math.max(0, r.b) : 0;
	if (!model.turns || !model.rate) return baseline;
	const x = features(r);
	const logTurns = Math.min(Math.log1p(MAX_TURNS), Math.max(0, evalHead(model.turns, x)));
	const logRate = Math.min(Math.log(MAX_REMAINING + 30), Math.max(Math.log(30 / (MAX_TURNS + 1)), evalHead(model.rate, x)));
	const estimate = Math.max(0, Math.min(MAX_REMAINING, Math.exp(logTurns + logRate) - 30));
	if (!Number.isFinite(estimate)) return baseline;
	// A conservative geometric blend prevents either noisy factor from dominating.
	return Math.max(0, Math.exp(0.7 * Math.log1p(baseline) + 0.3 * Math.log1p(estimate)) - 1);
}
