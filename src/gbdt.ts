// Compact, zero-dependency histogram-binned Gradient Boosted Decision Trees
// (GBDT) optimizing exact composite L1 loss on log remaining time + progress share.
//
// Compact tree representation:
//   - Leaf node: a number `v` (rounded to 4 decimals)
//   - Split node: `[featureIndex, threshold, leftChild, rightChild]`
// Inference runs in < 0.05 ms per call with zero allocations.

export type CompactNode = number | [number, number, CompactNode, CompactNode];

export type AnchorKind = "zero" | "elapsed" | "anchor";

export interface GBDTHead {
	anchor: AnchorKind;
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

export function evalEnsembleZ(model: EnsembleModel, x: number[], logElapsed: number, anchorZ: number): number {
	let sum = 0;
	for (let i = 0; i < model.heads.length; i++) {
		const h = model.heads[i]!;
		sum += anchorValue(h.anchor, logElapsed, anchorZ) + evalHead(h, x);
	}
	return sum / model.heads.length;
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

const MAX_BINS = 32;
const round4 = (n: number) => Math.round(n * 10_000) / 10_000;

function buildBins(X: number[][]) {
	const n = X.length;
	const nf = X[0]!.length;
	const thresholds: number[][] = [];
	const binned: Uint8Array[] = [];
	for (let f = 0; f < nf; f++) {
		const vals = new Float64Array(n);
		for (let i = 0; i < n; i++) vals[i] = X[i]![f]!;
		vals.sort();
		const thr: number[] = [];
		for (let b = 1; b < MAX_BINS; b++) {
			const v = vals[Math.floor((b * n) / MAX_BINS)]!;
			if (thr.length === 0 || v > thr[thr.length - 1]! + 1e-12) thr.push(v);
		}
		thresholds.push(thr);
		const col = new Uint8Array(n);
		for (let i = 0; i < n; i++) {
			const v = X[i]![f]!;
			let lo = 0;
			let hi = thr.length;
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

function weightedQuantile(vals: number[], wts: number[], q: number): number {
	const idx = vals.map((_, i) => i).sort((a, b) => vals[a]! - vals[b]!);
	let total = 0;
	for (let i = 0; i < wts.length; i++) total += wts[i]!;
	let acc = 0;
	for (const i of idx) {
		acc += wts[i]!;
		if (acc >= q * total) return vals[i]!;
	}
	return vals[idx[idx.length - 1]!]!;
}

function buildHistTree(
	thresholds: number[][],
	binned: Uint8Array[],
	grad: Float64Array,
	resid: Float64Array,
	wt: Float64Array,
	indices: Int32Array,
	depth: number,
	maxDepth: number,
	minLeaf: number,
	l2: number,
	feats: number[],
	quantile: number,
): CompactNode {
	const n = indices.length;
	const leaf = () => {
		const rVals = new Array<number>(n);
		const rWts = new Array<number>(n);
		for (let i = 0; i < n; i++) {
			const idx = indices[i]!;
			rVals[i] = resid[idx]!;
			rWts[i] = wt[idx]!;
		}
		return round4(weightedQuantile(rVals, rWts, quantile));
	};
	if (depth >= maxDepth || n < 2 * minLeaf) return leaf();

	let sumG = 0;
	let sumW = 0;
	for (let i = 0; i < n; i++) {
		const idx = indices[i]!;
		sumG += grad[idx]!;
		sumW += wt[idx]!;
	}
	const baseGain = (sumG * sumG) / (sumW + l2);
	let bestGain = 1e-6;
	let bestF = -1;
	let bestBin = -1;
	const binG = new Float64Array(MAX_BINS);
	const binW = new Float64Array(MAX_BINS);
	const binC = new Int32Array(MAX_BINS);

	for (const f of feats) {
		const thr = thresholds[f]!;
		const nb = thr.length + 1;
		if (nb <= 1) continue;
		const col = binned[f]!;
		binG.fill(0, 0, nb);
		binW.fill(0, 0, nb);
		binC.fill(0, 0, nb);
		for (let i = 0; i < n; i++) {
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
			if (leftC < minLeaf || n - leftC < minLeaf) continue;
			const rightG = sumG - leftG;
			const rightW = sumW - leftW;
			const gain = (leftG * leftG) / (leftW + l2) + (rightG * rightG) / (rightW + l2) - baseGain;
			if (gain > bestGain) {
				bestGain = gain;
				bestF = f;
				bestBin = b;
			}
		}
	}
	if (bestF === -1) return leaf();

	const col = binned[bestF]!;
	let lc = 0;
	for (let i = 0; i < n; i++) if (col[indices[i]!]! <= bestBin) lc++;
	const leftIdx = new Int32Array(lc);
	const rightIdx = new Int32Array(n - lc);
	let li = 0;
	let ri = 0;
	for (let i = 0; i < n; i++) {
		const idx = indices[i]!;
		if (col[idx]! <= bestBin) leftIdx[li++] = idx;
		else rightIdx[ri++] = idx;
	}
	return [
		bestF,
		// Equality belongs to the left bin. Rounding a log-valued cut down sends
		// every training observation at the boundary to the opposite leaf at runtime.
		thresholds[bestF]![bestBin]!,
		buildHistTree(thresholds, binned, grad, resid, wt, leftIdx, depth + 1, maxDepth, minLeaf, l2, feats, quantile),
		buildHistTree(thresholds, binned, grad, resid, wt, rightIdx, depth + 1, maxDepth, minLeaf, l2, feats, quantile),
	];
}

export interface TrainHeadOptions {
	anchor: AnchorKind;
	nTrees: number;
	lr: number;
	maxDepth: number;
	minLeaf: number;
	l2: number;
	colSample: number;
	seed: number;
	/** Quantile to target (0.5 = median / composite score; e.g. 0.25 or 0.75 for range bounds). */
	quantile?: number;
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
	const n = X.length;
	const nf = X[0]!.length;
	const { thresholds, binned } = buildBins(X);
	const yResidInit = yTarget.map((y, i) => y - anchorValues[i]!);
	const base = round4(weightedQuantile(yResidInit, sampleWt, q));
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
	const allFeats = Array.from({ length: nf }, (_, i) => i);

	for (let t = 0; t < opts.nTrees; t++) {
		for (let i = 0; i < n; i++) {
			const r = yResidInit[i]! - pred[i]!;
			resid[i] = r;
			const dir = r > 1e-9 ? q : r < -1e-9 ? q - 1 : 0;
			let w = sampleWt[i]!;
			if (Math.abs(q - 0.5) < 1e-6) {
				const zHat = anchorValues[i]! + pred[i]!;
				const e = elapsedS[i]!;
				const ez = Math.exp(Math.min(10, Math.max(0, zHat)));
				const remHat = Math.max(0, ez - 30);
				const dpdz = (e * ez) / Math.pow(e + remHat + 1e-6, 2);
				w *= 1 / Math.log(8) + dpdz;
			}
			wt[i] = w;
			grad[i] = dir * w;
		}
		const feats = allFeats.filter(() => rand() < opts.colSample);
		const rowTmp: number[] = [];
		for (let i = 0; i < n; i++) if (rand() < 0.78) rowTmp.push(i);
		const rowIdx = Int32Array.from(rowTmp.length >= 2 * opts.minLeaf ? rowTmp : Array.from({ length: n }, (_, i) => i));
		const tree = buildHistTree(
			thresholds,
			binned,
			grad,
			resid,
			wt,
			rowIdx,
			0,
			opts.maxDepth,
			opts.minLeaf,
			opts.l2,
			feats.length >= 6 ? feats : allFeats,
			q,
		);
		trees.push(tree);
		for (let i = 0; i < n; i++) pred[i]! += opts.lr * evalTree(tree, X[i]!);
	}
	return { anchor: opts.anchor, base, lr: opts.lr, trees };
}
