// Compact, zero-dependency histogram-binned Gradient Boosted Decision Trees
// (GBDT) optimizing exact composite L1 loss on log remaining time + progress share.
//
// Compact tree representation:
//   - Leaf node: a number `v` (rounded to 4 decimals)
//   - Split node: `[featureIndex, threshold, leftChild, rightChild]`
// Inference runs in < 0.1 ms per call with zero allocations.

import type { SurvivalNetModel } from "./survival-net";
import type { TextRidgeModel } from "./text-concepts";

export type CompactNode = number | [number, number, CompactNode, CompactNode];

export type AnchorKind = "zero" | "elapsed" | "anchor";

export type HeadRole = "core" | "early" | "long";

export interface GBDTHead {
	anchor: AnchorKind;
	/** Optional survival-regime role (defaults to "core" for flat averaging). */
	role?: HeadRole;
	base: number;
	lr: number;
	trees: CompactNode[];
}

export interface EnsembleModel {
	version: number;
	featureCount: number;
	heads: GBDTHead[];
	/** Optional calibrated lower/upper quantile heads (25th / 75th percentile of z = ln(remS + 30)). */
	qLow?: GBDTHead;
	qHigh?: GBDTHead;
	/** Optional 4-channel 40-concept Ridge duration priors. */
	textModel?: TextRidgeModel;
	/** Optional Truncated Log-Normal Survival ResMLP heads blended smoothly as elapsed grows. */
	survivalNets?: SurvivalNetModel[];
	/** Optional 8-octave empirical mean forecast slippage for causal online centering. */
	slipCenter?: number[];
}

export interface EvalContext {
	elapsedS?: number;
	toolCalls?: number;
	hasActiveTodo?: boolean;
}

export function evalTree(node: CompactNode, x: number[]): number {
	let cur = node;
	while (typeof cur !== "number") {
		cur = x[cur[0]]! <= cur[1] ? cur[2] : cur[3];
	}
	return cur;
}

export function evalHead(head: GBDTHead, x: number[]): number {
	let v = head.base;
	for (let i = 0; i < head.trees.length; i++) {
		v += head.lr * evalTree(head.trees[i]!, x);
	}
	return v;
}

function anchorValue(anchor: AnchorKind, logElapsed: number, anchorZ: number): number {
	return anchor === "zero" ? 0 : anchor === "elapsed" ? logElapsed : anchorZ;
}

const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));

/**
 * Evaluate the ensemble in log-remaining space `z = ln(remS + 30)`.
 * When survival-regime heads (`role: "early"` / `"long"`) are present alongside
 * core heads, applies smooth C-infinity causal transitions in elapsed active time:
 *   - Early regime (first ~25s before tool/plan buildup): tilts toward unbiased all-prompt head.
 *   - Long regime (past ~12m elapsed): transitions smoothly toward `t >= 300s` survival specialists.
 * Backward compatible with flat `version: 2` models (averages all heads equally).
 */
export function evalEnsembleZ(
	model: EnsembleModel,
	x: number[],
	logElapsed: number,
	anchorZ: number,
	ctx?: EvalContext,
): number {
	let coreSum = 0;
	let coreN = 0;
	let earlySum = 0;
	let earlyN = 0;
	let longSum = 0;
	let longN = 0;

	for (let i = 0; i < model.heads.length; i++) {
		const h = model.heads[i]!;
		const z = anchorValue(h.anchor, logElapsed, anchorZ) + evalHead(h, x);
		const role = h.role ?? "core";
		if (role === "long") {
			longSum += z;
			longN++;
		} else if (role === "early") {
			// Early-role head is trained on all completed prompts >5s; it participates in core AND anchors early regime.
			coreSum += z;
			coreN++;
			earlySum += z;
			earlyN++;
		} else {
			coreSum += z;
			coreN++;
		}
	}

	const zCore = coreN > 0 ? coreSum / coreN : longN > 0 ? longSum / longN : logElapsed;
	if (!ctx || ctx.elapsedS === undefined) {
		if (longN === 0) return zCore;
		return (coreSum + longSum) / (coreN + longN);
	}

	const e = Math.max(0, ctx.elapsedS);
	let z = zCore;
	if (longN > 0) {
		const zLong = longSum / longN;
		// Smooth transition toward long-survival specialists once elapsed passes ~12m (720s).
		const wLong = 0.65 * sigmoid((Math.log(e + 1) - Math.log(720)) / 0.28);
		z = (1 - wLong) * z + wLong * zLong;
	}
	if (earlyN > 0 && !ctx.hasActiveTodo) {
		const zEarly = earlySum / earlyN;
		const tools = ctx.toolCalls ?? 0;
		// Gentle early-regime tilt only when very early (<25s) and low tool count (<=2) without an active todo plan.
		const wEarly = 0.5 * sigmoid(-(e - 25) / 7) * sigmoid(-(tools - 2.5) / 0.8);
		z = wEarly * zEarly + (1 - wEarly) * z;
	}
	return z;
}

export function evalQuantileRangeZ(
	model: EnsembleModel,
	x: number[],
	logElapsed: number,
	anchorZ: number,
	medianZ: number,
): { lowZ: number; highZ: number } {
	const rawLow = model.qLow ? anchorValue(model.qLow.anchor, logElapsed, anchorZ) + evalHead(model.qLow, x) : medianZ - 0.5;
	const rawHigh = model.qHigh ? anchorValue(model.qHigh.anchor, logElapsed, anchorZ) + evalHead(model.qHigh, x) : medianZ + 0.5;
	return {
		lowZ: Math.min(rawLow, medianZ),
		highZ: Math.max(rawHigh, medianZ),
	};
}

// ---- Training (used by eval/train.ts) ----

const DEFAULT_MAX_BINS = 64;
const round4 = (n: number) => Math.round(n * 10_000) / 10_000;

function buildBins(X: number[][], maxBins: number) {
	const n = X.length;
	const nf = X[0]!.length;
	const thresholds: number[][] = [];
	const binned: Uint8Array[] = [];
	const vals = new Float64Array(n);
	for (let f = 0; f < nf; f++) {
		for (let i = 0; i < n; i++) vals[i] = X[i]![f]!;
		vals.sort();
		const thr: number[] = [];
		for (let b = 1; b < maxBins; b++) {
			const v = vals[Math.floor((b * n) / maxBins)]!;
			if (thr.length === 0 || v > thr[thr.length - 1]! + 1e-12) thr.push(v);
		}
		thresholds.push(thr);
		const col = new Uint8Array(n);
		const tl = thr.length;
		for (let i = 0; i < n; i++) {
			const v = X[i]![f]!;
			let lo = 0;
			let hi = tl;
			while (lo < hi) {
				const mid = (lo + hi) >> 1;
				if (v <= thr[mid]!) hi = mid;
				else lo = mid + 1;
			}
			col[i] = lo;
		}
		binned.push(col);
	}
	return { thresholds, binned };
}

/** O(n) weighted quantile via quickselect over TypedArrays (no GC churn). */
function weightedQuantileFast(vals: Float64Array, wts: Float64Array, n: number, q: number): number {
	const v = new Float64Array(n);
	const w = new Float64Array(n);
	let total = 0;
	for (let i = 0; i < n; i++) {
		v[i] = vals[i]!;
		w[i] = wts[i]!;
		total += w[i]!;
	}
	const target = q * total;
	let lo = 0;
	let hi = n - 1;
	let accLeft = 0;
	while (lo < hi) {
		const mid = (lo + hi) >> 1;
		const a = v[lo]!;
		const b = v[mid]!;
		const c = v[hi]!;
		const pivot = a < b ? (b < c ? b : a < c ? c : a) : (a < c ? a : b < c ? c : b);
		let i = lo;
		let j = hi;
		while (i <= j) {
			while (v[i]! < pivot) i++;
			while (v[j]! > pivot) j--;
			if (i <= j) {
				const tv = v[i]!;
				v[i] = v[j]!;
				v[j] = tv;
				const tw = w[i]!;
				w[i] = w[j]!;
				w[j] = tw;
				i++;
				j--;
			}
		}
		let leftW = 0;
		for (let k = lo; k <= j; k++) leftW += w[k]!;
		if (accLeft + leftW >= target) {
			hi = j;
		} else {
			accLeft += leftW;
			let eqW = 0;
			for (let k = j + 1; k < i; k++) eqW += w[k]!;
			if (accLeft + eqW >= target) return pivot;
			accLeft += eqW;
			lo = i;
		}
	}
	return v[lo]!;
}

export interface TrainHeadOptions {
	anchor: AnchorKind;
	role?: HeadRole;
	nTrees: number;
	lr: number;
	maxDepth: number;
	minLeaf: number;
	l2: number;
	colSample: number;
	seed: number;
	/** Quantile to target (0.5 = median / composite score; e.g. 0.25 or 0.75 for range bounds). */
	quantile?: number;
	/** Histogram bin count per feature (defaults to 64). */
	maxBins?: number;
}

export function trainGBDTHead(
	X: number[][],
	yTarget: number[],
	anchorValues: number[],
	elapsedS: number[],
	sampleWt: number[],
	opts: TrainHeadOptions,
): GBDTHead {
	const q = opts.quantile ?? 0.5;
	const maxBins = opts.maxBins ?? DEFAULT_MAX_BINS;
	const n = X.length;
	const nf = X[0]!.length;
	const { thresholds, binned } = buildBins(X, maxBins);
	const yResidInit = new Float64Array(n);
	const swArr = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		yResidInit[i] = yTarget[i]! - anchorValues[i]!;
		swArr[i] = sampleWt[i]!;
	}
	const base = round4(weightedQuantileFast(yResidInit, swArr, n, q));
	const pred = new Float64Array(n).fill(base);
	const trees: CompactNode[] = [];
	let rng = opts.seed >>> 0;
	const rand = () => {
		rng = (rng * 1664525 + 1013904223) >>> 0;
		return rng / 0x100000000;
	};
	const grad = new Float64Array(n);
	const resid = new Float64Array(n);
	const wt = new Float64Array(n);
	const leafVals = new Float64Array(n);
	const leafWts = new Float64Array(n);
	const binG = new Float64Array(maxBins);
	const binW = new Float64Array(maxBins);
	const binC = new Int32Array(maxBins);
	const allFeats = Array.from({ length: nf }, (_, i) => i);
	const invLn8 = 1 / Math.log(8);
	const isMedian = Math.abs(q - 0.5) < 1e-6;

	const buildTree = (indices: Int32Array, depth: number, feats: number[]): CompactNode => {
		const m = indices.length;
		const makeLeaf = () => {
			for (let i = 0; i < m; i++) {
				const idx = indices[i]!;
				leafVals[i] = resid[idx]!;
				leafWts[i] = wt[idx]!;
			}
			return round4(weightedQuantileFast(leafVals, leafWts, m, q));
		};
		if (depth >= opts.maxDepth || m < 2 * opts.minLeaf) return makeLeaf();

		let sumG = 0;
		let sumW = 0;
		for (let i = 0; i < m; i++) {
			const idx = indices[i]!;
			sumG += grad[idx]!;
			sumW += wt[idx]!;
		}
		const baseGain = (sumG * sumG) / (sumW + opts.l2);
		let bestGain = 1e-6;
		let bestF = -1;
		let bestBin = -1;

		for (let fi = 0; fi < feats.length; fi++) {
			const f = feats[fi]!;
			const thr = thresholds[f]!;
			const nb = thr.length + 1;
			if (nb <= 1) continue;
			const col = binned[f]!;
			binG.fill(0, 0, nb);
			binW.fill(0, 0, nb);
			binC.fill(0, 0, nb);
			for (let i = 0; i < m; i++) {
				const idx = indices[i]!;
				const b = col[idx]!;
				binG[b]! += grad[idx]!;
				binW[b]! += wt[idx]!;
				binC[b]! += 1;
			}
			let leftG = 0;
			let leftW = 0;
			let leftC = 0;
			for (let b = 0; b < nb - 1; b++) {
				leftG += binG[b]!;
				leftW += binW[b]!;
				leftC += binC[b]!;
				if (leftC < opts.minLeaf || m - leftC < opts.minLeaf) continue;
				const rightG = sumG - leftG;
				const rightW = sumW - leftW;
				const gain = (leftG * leftG) / (leftW + opts.l2) + (rightG * rightG) / (rightW + opts.l2) - baseGain;
				if (gain > bestGain) {
					bestGain = gain;
					bestF = f;
					bestBin = b;
				}
			}
		}
		if (bestF === -1) return makeLeaf();

		const col = binned[bestF]!;
		let lc = 0;
		for (let i = 0; i < m; i++) if (col[indices[i]!]! <= bestBin) lc++;
		const leftIdx = new Int32Array(lc);
		const rightIdx = new Int32Array(m - lc);
		let li = 0;
		let ri = 0;
		for (let i = 0; i < m; i++) {
			const idx = indices[i]!;
			if (col[idx]! <= bestBin) leftIdx[li++] = idx;
			else rightIdx[ri++] = idx;
		}
		return [
			bestF,
			// Equality belongs to the left bin. Rounding a log-valued cut down sends
			// every training observation at the boundary to the opposite leaf at runtime.
			thresholds[bestF]![bestBin]!,
			buildTree(leftIdx, depth + 1, feats),
			buildTree(rightIdx, depth + 1, feats),
		];
	};

	for (let t = 0; t < opts.nTrees; t++) {
		for (let i = 0; i < n; i++) {
			const r = yResidInit[i]! - pred[i]!;
			resid[i] = r;
			const dir = r > 1e-9 ? q : r < -1e-9 ? q - 1 : 0;
			let w = swArr[i]!;
			if (isMedian) {
				const zHat = anchorValues[i]! + pred[i]!;
				const e = elapsedS[i]!;
				const ez = Math.exp(Math.min(10, Math.max(0, zHat)));
				const remHat = Math.max(0, ez - 30);
				const dpdz = (e * ez) / Math.pow(e + remHat + 1e-6, 2);
				w *= invLn8 + dpdz;
			}
			wt[i] = w;
			grad[i] = dir * w;
		}
		const feats = allFeats.filter(() => rand() < opts.colSample);
		const rowTmp: number[] = [];
		for (let i = 0; i < n; i++) if (rand() < 0.78) rowTmp.push(i);
		const rowIdx = Int32Array.from(rowTmp.length >= 2 * opts.minLeaf ? rowTmp : Array.from({ length: n }, (_, i) => i));
		const tree = buildTree(rowIdx, 0, feats.length >= 6 ? feats : allFeats);
		trees.push(tree);
		for (let i = 0; i < n; i++) pred[i]! += opts.lr * evalTree(tree, X[i]!);
	}
	return opts.role ? { anchor: opts.anchor, role: opts.role, base, lr: opts.lr, trees } : { anchor: opts.anchor, base, lr: opts.lr, trees };
}
