// Compact pure-TypeScript Exact Truncated Log-Normal Survival Residual MLP.
// Models total active prompt duration ln(T + 30) ~ N(mu(x), sigma(x)^2) via
// exact conditional survival negative log-likelihood ln p(T | T > e, x), then
// computes the closed-form conditional remaining median:
//   T_50(e) = exp(mu + sigma * Phi^-1(0.5 + 0.5 * Phi((ln(e+30) - mu)/sigma))) - 30
// Runs in < 0.04 ms with zero external dependencies and gives genuine cross-family
// (smooth neural vs axis-aligned tree) diversity alongside the GBDT ensemble.

import { SMOOTH_S } from "./features";

export interface SurvivalNetModel {
	d: number;
	h: number;
	muIn: number[];
	invSdIn: number[];
	W1: number[];
	b1: number[];
	W2: number[];
	b2: number[];
	Wo: number[];
	bo: number[];
}

const INV_SQRT2PI = 1 / Math.sqrt(2 * Math.PI);
const normPdf = (x: number) => INV_SQRT2PI * Math.exp(-0.5 * x * x);

/** Abramowitz & Stegun 7.1.26 standard normal CDF (max error < 1.5e-7). */
function normCdf(x: number): number {
	const sign = x < 0 ? -1 : 1;
	const z = Math.abs(x) / Math.SQRT2;
	const t = 1 / (1 + 0.3275911 * z);
	const y =
		1 -
		(((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t) *
			Math.exp(-z * z);
	return 0.5 * (1 + sign * y);
}

/** Acklam rational approximation to standard normal inverse CDF (error < 1e-9). */
function normInv(p: number): number {
	const pp = Math.max(1e-7, Math.min(1 - 1e-7, p));
	const a = [
		-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.383577518672690e2, -3.066479806614716e1,
		2.506628277459239e0,
	];
	const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
	const c = [
		-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838e0, -2.549732539343734e0, 4.374664141464968e0,
		2.938163982698783e0,
	];
	const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996e0, 3.754408661907416e0];
	const pLow = 0.02425;
	const pHigh = 1 - pLow;
	if (pp < pLow) {
		const q = Math.sqrt(-2 * Math.log(pp));
		return (
			(((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) /
			((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1)
		);
	}
	if (pp <= pHigh) {
		const q = pp - 0.5;
		const r = q * q;
		return (
			((((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r + a[5]!) * q) /
			(((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1)
		);
	}
	const q = Math.sqrt(-2 * Math.log(1 - pp));
	return -(
		(((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) /
		((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1)
	);
}

const invMills = (u: number) => (u > 6 ? u + 1 / u : normPdf(u) / Math.max(1e-12, 1 - normCdf(u)));
const silu = (x: number) => x / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
const dsilu = (x: number) => {
	const s = 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
	return s * (1 + x * (1 - s));
};
const symlog = (v: number) => Math.sign(v) * Math.log1p(Math.abs(v));
const round4 = (n: number) => Math.round(n * 10_000) / 10_000;

export function truncLogNormMedZ(mu: number, sigma: number, elapsedS: number): number {
	const logE = Math.log(elapsedS + SMOOTH_S);
	const p0 = normCdf((logE - mu) / sigma);
	const zT = mu + sigma * normInv(p0 + 0.5 * (1 - p0));
	const totS = Math.exp(Math.min(10.8, Math.max(0, zT))) - SMOOTH_S;
	return Math.log(Math.max(0, totS - elapsedS) + SMOOTH_S);
}

/** Evaluate one Truncated Log-Normal Survival ResMLP, returning conditional median z = ln(remS + 30). */
export function evalSurvivalNetZ(m: SurvivalNetModel, xRaw: number[], elapsedS: number): number {
	const { d, h, muIn, invSdIn, W1, b1, W2, b2, Wo, bo } = m;
	const xn = new Float64Array(d);
	for (let j = 0; j < d; j++) {
		const v = (symlog(xRaw[j] ?? 0) - muIn[j]!) * invSdIn[j]!;
		xn[j] = v > 4 ? 4 : v < -4 ? -4 : v;
	}
	const h1 = new Float64Array(h);
	for (let i = 0; i < h; i++) {
		let s = b1[i]!;
		const off = i * d;
		for (let j = 0; j < d; j++) s += W1[off + j]! * xn[j]!;
		h1[i] = silu(s);
	}
	const h2 = new Float64Array(h);
	for (let i = 0; i < h; i++) {
		let s = b2[i]!;
		const off = i * h;
		for (let j = 0; j < h; j++) s += W2[off + j]! * h1[j]!;
		h2[i] = h1[i]! + silu(s);
	}
	let o0 = bo[0]!;
	let o1 = bo[1]!;
	for (let j = 0; j < h; j++) {
		o0 += Wo[j]! * h2[j]!;
		o1 += Wo[h + j]! * h2[j]!;
	}
	const mu = Math.max(3.4, Math.min(9.8, Math.log(elapsedS + SMOOTH_S) + o0));
	const sigma = Math.exp(Math.max(-1.3, Math.min(0.8, o1)));
	return truncLogNormMedZ(mu, sigma, elapsedS);
}

export interface TrainSurvivalNetOptions {
	hidden: number;
	epochs: number;
	batchSize: number;
	lr: number;
	wd: number;
	dropout: number;
	seed: number;
}

export function trainSurvivalNet(
	X: number[][],
	totalS: number[],
	elapsedS: number[],
	sampleWt: number[],
	opts: TrainSurvivalNetOptions,
): SurvivalNetModel {
	const n = X.length;
	const d = X[0]!.length;
	const h = opts.hidden;
	const muIn = new Float64Array(d);
	const invSdIn = new Float64Array(d);
	let wSum = 0;
	for (let i = 0; i < n; i++) wSum += sampleWt[i]!;
	for (let i = 0; i < n; i++) {
		const w = sampleWt[i]! / wSum;
		const xi = X[i]!;
		for (let j = 0; j < d; j++) muIn[j]! += w * symlog(xi[j]!);
	}
	for (let i = 0; i < n; i++) {
		const w = sampleWt[i]! / wSum;
		const xi = X[i]!;
		for (let j = 0; j < d; j++) {
			const df = symlog(xi[j]!) - muIn[j]!;
			invSdIn[j]! += w * df * df;
		}
	}
	for (let j = 0; j < d; j++) invSdIn[j] = 1 / Math.max(0.05, Math.sqrt(invSdIn[j]!));

	const Xs = new Float32Array(n * d);
	const logT = new Float64Array(n);
	const logE = new Float64Array(n);
	const sw = new Float64Array(n);
	for (let i = 0; i < n; i++) {
		const xi = X[i]!;
		const off = i * d;
		for (let j = 0; j < d; j++) {
			const v = (symlog(xi[j]!) - muIn[j]!) * invSdIn[j]!;
			Xs[off + j] = v > 4 ? 4 : v < -4 ? -4 : v;
		}
		logT[i] = Math.log(totalS[i]! + SMOOTH_S);
		logE[i] = Math.log(elapsedS[i]! + SMOOTH_S);
		sw[i] = (sampleWt[i]! * n) / wSum;
	}

	let rng = opts.seed >>> 0;
	const rand = () => {
		rng = (rng * 1664525 + 1013904223) >>> 0;
		return rng / 0x100000000;
	};
	const randn = () => Math.sqrt(-2 * Math.log(Math.max(1e-12, rand()))) * Math.cos(2 * Math.PI * rand());

	const sW1 = h * d;
	const sb1 = h;
	const sW2 = h * h;
	const sb2 = h;
	const sWo = 2 * h;
	const sbo = 2;
	const nP = sW1 + sb1 + sW2 + sb2 + sWo + sbo;
	const params = new Float64Array(nP);
	const grads = new Float64Array(nP);
	const mA = new Float64Array(nP);
	const vA = new Float64Array(nP);
	const oW1 = 0;
	const ob1 = sW1;
	const oW2 = ob1 + sb1;
	const ob2 = oW2 + sW2;
	const oWo = ob2 + sb2;
	const obo = oWo + sWo;

	const initW = (off: number, len: number, sc: number) => {
		for (let i = 0; i < len; i++) params[off + i] = randn() * sc;
	};
	initW(oW1, sW1, Math.sqrt(2 / d));
	initW(oW2, sW2, Math.sqrt(1 / h) * 0.4);
	initW(oWo, sWo, Math.sqrt(1 / h) * 0.25);
	params[obo] = 0.5;
	params[obo + 1] = -0.15;

	const order = new Int32Array(n);
	for (let i = 0; i < n; i++) order[i] = i;
	const a1 = new Float64Array(h);
	const h1 = new Float64Array(h);
	const a2 = new Float64Array(h);
	const h2 = new Float64Array(h);
	const mask1 = new Float64Array(h);
	const dh2 = new Float64Array(h);
	const da2 = new Float64Array(h);
	const dh1 = new Float64Array(h);
	const da1 = new Float64Array(h);
	const keep = 1 - opts.dropout;
	const invKeep = 1 / keep;
	let step = 0;
	const totalSteps = opts.epochs * Math.ceil(n / opts.batchSize);

	for (let ep = 0; ep < opts.epochs; ep++) {
		for (let i = n - 1; i > 0; i--) {
			const j = Math.floor(rand() * (i + 1));
			const tmp = order[i]!;
			order[i] = order[j]!;
			order[j] = tmp;
		}
		for (let start = 0; start < n; start += opts.batchSize) {
			const end = Math.min(n, start + opts.batchSize);
			const invBs = 1 / (end - start);
			grads.fill(0);
			for (let bi = start; bi < end; bi++) {
				const idx = order[bi]!;
				const xOff = idx * d;
				const wi = sw[idx]! * invBs;
				for (let i = 0; i < h; i++) {
					let s = params[ob1 + i]!;
					const row = oW1 + i * d;
					for (let j = 0; j < d; j++) s += params[row + j]! * Xs[xOff + j]!;
					a1[i] = s;
					const mk = rand() < keep ? invKeep : 0;
					mask1[i] = mk;
					h1[i] = silu(s) * mk;
				}
				for (let i = 0; i < h; i++) {
					let s = params[ob2 + i]!;
					const row = oW2 + i * h;
					for (let j = 0; j < h; j++) s += params[row + j]! * h1[j]!;
					a2[i] = s;
					h2[i] = h1[i]! + silu(s);
				}
				let o0 = params[obo]!;
				let o1 = params[obo + 1]!;
				for (let j = 0; j < h; j++) {
					o0 += params[oWo + j]! * h2[j]!;
					o1 += params[oWo + h + j]! * h2[j]!;
				}
				const sig = Math.exp(Math.max(-1.3, Math.min(0.8, o1)));
				const mu = logE[idx]! + o0;
				const zT = (logT[idx]! - mu) / sig;
				const u0 = (logE[idx]! - mu) / sig;
				const lam = invMills(u0);
				const dMu = wi * Math.max(-5, Math.min(5, -(zT - lam) / sig));
				const dLS = wi * Math.max(-5, Math.min(5, 1 - zT * zT + lam * u0)) * (o1 > -1.3 && o1 < 0.8 ? 1 : 0.1);
				grads[obo]! += dMu;
				grads[obo + 1]! += dLS;
				for (let j = 0; j < h; j++) {
					grads[oWo + j]! += dMu * h2[j]!;
					grads[oWo + h + j]! += dLS * h2[j]!;
					dh2[j] = dMu * params[oWo + j]! + dLS * params[oWo + h + j]!;
				}
				for (let i = 0; i < h; i++) {
					da2[i] = dh2[i]! * dsilu(a2[i]!);
					dh1[i] = dh2[i]!;
					grads[ob2 + i]! += da2[i]!;
				}
				for (let i = 0; i < h; i++) {
					const g = da2[i]!;
					const row = oW2 + i * h;
					for (let j = 0; j < h; j++) {
						grads[row + j]! += g * h1[j]!;
						dh1[j]! += g * params[row + j]!;
					}
				}
				for (let i = 0; i < h; i++) {
					da1[i] = dh1[i]! * mask1[i]! * dsilu(a1[i]!);
					grads[ob1 + i]! += da1[i]!;
					const g = da1[i]!;
					const row = oW1 + i * d;
					for (let j = 0; j < d; j++) grads[row + j]! += g * Xs[xOff + j]!;
				}
			}
			step++;
			const lrT = opts.lr * 0.5 * (1 + Math.cos((Math.PI * step) / totalSteps));
			let gn = 0;
			for (let i = 0; i < nP; i++) gn += grads[i]! * grads[i]!;
			const clip = gn > 25 ? 5 / Math.sqrt(gn) : 1;
			const b1c = 1 - Math.pow(0.9, step);
			const b2c = 1 - Math.pow(0.999, step);
			for (let i = 0; i < nP; i++) {
				const g = grads[i]! * clip;
				mA[i] = 0.9 * mA[i]! + 0.1 * g;
				vA[i] = 0.999 * vA[i]! + 0.001 * g * g;
				const isW = i < ob1 || (i >= oW2 && i < ob2) || (i >= oWo && i < obo);
				if (isW) params[i]! *= 1 - lrT * opts.wd;
				params[i]! -= (lrT * (mA[i]! / b1c)) / (Math.sqrt(vA[i]! / b2c) + 1e-8);
			}
		}
	}

	const pack = (arr: Float64Array, off = 0, len = arr.length) => {
		const out = new Array<number>(len);
		for (let i = 0; i < len; i++) out[i] = round4(arr[off + i]!);
		return out;
	};
	return {
		d,
		h,
		muIn: pack(muIn),
		invSdIn: pack(invSdIn),
		W1: pack(params, oW1, sW1),
		b1: pack(params, ob1, sb1),
		W2: pack(params, oW2, sW2),
		b2: pack(params, ob2, sb2),
		Wo: pack(params, oWo, sWo),
		bo: pack(params, obo, sbo),
	};
}
