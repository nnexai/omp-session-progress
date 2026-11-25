#!/usr/bin/env bun
// Fit the local v4 hybrid survival-regime GBDT + Truncated Log-Normal Survival ResMLP
// ensemble, 4-channel public-concept Ridge text priors, causal slippage centering,
// and out-of-fold 6×6 range calibration table on historic OMP sessions.
// Training is 100% local; model artifacts contain only numeric weights/splits, never session text.
import { closeSync, openSync, readFileSync, readdirSync, readSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { applyEvent, emptyFacts, type Facts } from "../src/facts";
import { extractFeatures, SMOOTH_S } from "../src/features";
import { type EnsembleModel, evalEnsembleZ, type TrainHeadOptions, trainGBDTHead } from "../src/gbdt";
import { evalSurvivalNetZ, trainSurvivalNet } from "../src/survival-net";
import { conceptCounts, conceptVec, evalRidge, fitRidge, type TextRidgeModel } from "../src/text-concepts";
import { activeSegments, extractSession, type Trace } from "./extract";
import { fit as fitRangeModel } from "./horizon-study/ranges";
import { allDenseLandmarks, landmarks, type Sample } from "./landmarks";
import { SESSIONS_DIR } from "./paths";

const project = resolve(import.meta.dir, "..");
const excludedSessions = new Set<string>();
if (!process.argv.includes("--include-all")) {
	const state = JSON.parse(readFileSync(join(project, ".omp/ratchet/progress-time/_state.json"), "utf8"));
	if (!state.test_ids?.length) throw new Error("Cannot train without the held-out session split");
	for (const id of state.test_ids as string[]) excludedSessions.add(id.replace(/-p\d+$/, ""));
}

const t0 = performance.now();
const header = Buffer.alloc(4096);
const excludedProjects = ["omp-session-progress", "omp-progress-smoke", "omp-progress-bench", "omp-progress-eval"];

interface LoadedSession {
	id: string;
	trace: Trace;
	maxDurS: number;
	fold: number;
}

const loaded: LoadedSession[] = [];
for (const dir of readdirSync(SESSIONS_DIR).sort()) {
	const full = join(SESSIONS_DIR, dir);
	if (!statSync(full).isDirectory()) continue;
	for (const file of readdirSync(full).sort()) {
		if (!file.endsWith(".jsonl")) continue;
		const path = join(full, file);
		const st = statSync(path);
		if (!st.isFile() || st.size < 200 || st.size > 64 * 1024 * 1024) continue;
		const fd = openSync(path, "r");
		try {
			const head = header.toString("utf8", 0, readSync(fd, header));
			if (excludedProjects.some(s => head.includes(s))) continue;
		} finally {
			closeSync(fd);
		}
		const trace = extractSession(path);
		if (excludedSessions.has(trace.id)) continue;
		let maxDurS = 0;
		let hasPrompt = false;
		for (const p of trace.prompts) {
			if (p.outcome !== "completed") continue;
			const durS = activeSegments(trace.events, p).reduce((sum, s) => sum + s.ms, 0) / 1000;
			if (durS > 5) {
				hasPrompt = true;
				if (durS > maxDurS) maxDurS = durS;
			}
		}
		if (hasPrompt) loaded.push({ id: trace.id, trace, maxDurS, fold: 0 });
	}
}

// Deterministic stratified 5-fold assignment by session max prompt duration.
loaded.sort((a, b) => b.maxDurS - a.maxDurS || a.id.localeCompare(b.id));
for (let i = 0; i < loaded.length; i++) loaded[i]!.fold = i % 5;

type TrainSample = Sample | "allDense";
interface BaseRow {
	fold: number;
	xBase: number[];
	tP: Float64Array;
	tT: Float64Array;
	tI: Float64Array;
	tR: Float64Array;
	e: number;
	y: number;
	totalS: number;
	todoAnchor: number;
}
interface TrainRow extends BaseRow {
	sample: TrainSample;
	bucket: number;
	promptWt: number;
}
interface HorizonRow extends BaseRow {
	sid: string;
	pid: number;
	toolCalls: number;
	hasActiveTodo: boolean;
}

function textChannels(facts: Facts) {
	const effPrompt =
		facts.isApprovalPrompt && facts.prevAssistText ? `${facts.promptText}\n${facts.prevAssistText}` : facts.promptText;
	const remTodoTexts = facts.todoUpdated
		? facts.todos.filter(t => t.status !== "done" && t.status !== "dropped").map(t => t.text)
		: [];
	return {
		tP: conceptVec(effPrompt),
		tT: conceptCounts(remTodoTexts),
		tI: conceptCounts(facts.recentIntents),
		tR: conceptCounts(facts.recentProse),
	};
}

const trainRows: TrainRow[] = [];
const horizonRows: HorizonRow[] = [];
let clockPrompts = 0;

for (const sess of loaded) {
	const facts = emptyFacts();
	let cursor = 0;
	for (let pi = 0; pi < sess.trace.prompts.length; pi++) {
		const prompt = sess.trace.prompts[pi]!;
		if (prompt.outcome !== "completed") continue;
		const segments = activeSegments(sess.trace.events, prompt);
		const totalMs = segments.reduce((sum, s) => sum + s.ms, 0);
		const totalS = totalMs / 1000;
		if (totalS <= 5) continue;
		clockPrompts++;
		const nTools = prompt.toolStartIndices.length;
		const bucket = totalS < 60 ? 0 : totalS < 180 ? 1 : totalS <= 900 ? 2 : 3;
		const structured = bucket === 0 ? totalS >= 15 && nTools >= 2 : nTools >= 5;

		type PointReq = {
			kind: TrainSample | "horizon";
			at: number;
			cut: number;
			elapsedS: number;
			remainingS: number;
			promptWt: number;
		};
		const reqs: PointReq[] = [];
		for (const p of landmarks(segments, structured)) {
			reqs.push({ kind: p.sample, at: p.at, cut: p.cut, elapsedS: p.elapsedS, remainingS: p.remainingS, promptWt: 1 });
		}
		const ad = allDenseLandmarks(segments);
		for (const p of ad) {
			reqs.push({
				kind: "allDense",
				at: p.at,
				cut: p.cut,
				elapsedS: p.elapsedS,
				remainingS: p.remainingS,
				promptWt: 1 / ad.length,
			});
		}
		let segIdx = 0;
		let segElapsed = 0;
		for (let target = 5_000; target < totalMs; target = target === 5_000 ? 15_000 : target + 15_000) {
			while (segIdx < segments.length && segElapsed + segments[segIdx]!.ms < target) segElapsed += segments[segIdx++]!.ms;
			const seg = segments[segIdx]!;
			reqs.push({
				kind: "horizon",
				at: Math.round(seg.t + target - segElapsed),
				cut: seg.index + 1,
				elapsedS: target / 1000,
				remainingS: Math.max(0, (totalMs - target) / 1000),
				promptWt: 1,
			});
		}
		reqs.sort((a, b) => a.elapsedS - b.elapsedS);

		for (const p of reqs) {
			while (cursor < p.cut && cursor < sess.trace.events.length) {
				const ev = sess.trace.events[cursor++]!;
				applyEvent(facts, ev.event, ev.t);
			}
			const feat = extractFeatures(facts, p.at);
			// First 145 features are 140 base + 5 Markov; last 5 are filled from out-of-fold TextRidge.
			const xBase = feat.x.slice(0, feat.x.length - 5);
			const ch = textChannels(facts);
			if (p.kind === "horizon") {
				horizonRows.push({
					fold: sess.fold,
					sid: sess.id,
					pid: pi,
					xBase,
					...ch,
					e: p.elapsedS,
					y: p.remainingS,
					totalS,
					todoAnchor: feat.anchorZ,
					toolCalls: feat.toolCalls,
					hasActiveTodo: feat.hasActiveTodo,
				});
			} else {
				trainRows.push({
					fold: sess.fold,
					sample: p.kind,
					xBase,
					...ch,
					e: p.elapsedS,
					y: p.remainingS,
					totalS,
					todoAnchor: feat.anchorZ,
					bucket,
					promptWt: p.promptWt,
				});
			}
		}
	}
}

const round4 = (n: number) => Math.round(n * 10_000) / 10_000;

function fitTextModel(rows: TrainRow[]): TextRidgeModel {
	const ad = rows.filter(r => r.sample === "allDense");
	const cl = rows.filter(r => r.sample === "clock");
	return {
		bP: fitRidge(ad.map(r => r.tP), Float64Array.from(ad.map(r => Math.log(r.totalS + SMOOTH_S))), Float64Array.from(ad.map(r => r.promptWt)), 35),
		bT: fitRidge(cl.map(r => r.tT), Float64Array.from(cl.map(r => Math.log(r.y + SMOOTH_S))), Float64Array.from(cl.map(() => 1)), 45),
		bI: fitRidge(cl.map(r => r.tI), Float64Array.from(cl.map(r => Math.log(r.y + SMOOTH_S))), Float64Array.from(cl.map(() => 1)), 45),
		bR: fitRidge(cl.map(r => r.tR), Float64Array.from(cl.map(r => Math.log(r.y + SMOOTH_S))), Float64Array.from(cl.map(() => 1)), 45),
	};
}

function fullX(r: BaseRow, tm: TextRidgeModel): number[] {
	const logE = Math.log(r.e + SMOOTH_S);
	const zp = evalRidge(tm.bP, r.tP);
	return [...r.xBase, zp, zp - logE, evalRidge(tm.bT, r.tT), evalRidge(tm.bI, r.tI), evalRidge(tm.bR, r.tR)];
}

const common = { lr: 0.045, colSample: 0.45, l2: 12, maxBins: 64 };
const quantiles = { anchor: "elapsed", nTrees: 45, lr: 0.055, maxDepth: 3, minLeaf: 24, l2: 12, colSample: 0.5, maxBins: 64 } as const;

function trainPointModels(rows: TrainRow[], innerTms: TextRidgeModel[], includeQuantiles: boolean) {
	const counts = [0, 0, 0, 0];
	for (const r of rows) if (r.sample === "sparse") counts[r.bucket]! += 0.25;
	const main = counts[1]! + counts[2]! + counts[3]!;
	const structuredWt = (r: TrainRow) => (r.bucket === 0 ? 0.15 : Math.sqrt(main / (3 * counts[r.bucket]!)));
	const xOf = (r: TrainRow) => fullX(r, innerTms[r.fold]!);
	const fitG = (sub: TrainRow[], weightFn: (r: TrainRow) => number, options: TrainHeadOptions) =>
		trainGBDTHead(
			sub.map(xOf),
			sub.map(r => Math.log(r.y + SMOOTH_S)),
			sub.map(r => (options.anchor === "zero" ? 0 : options.anchor === "elapsed" ? Math.log(r.e + SMOOTH_S) : r.todoAnchor)),
			sub.map(r => r.e),
			sub.map(weightFn),
			options,
		);

	const sparse = rows.filter(r => r.sample === "sparse");
	const dense = rows.filter(r => r.sample === "dense");
	const allDense = rows.filter(r => r.sample === "allDense");
	const clock = rows.filter(r => r.sample === "clock");
	const longClock = clock.filter(r => r.e >= 300);

	const heads = [
		fitG(sparse, structuredWt, { ...common, anchor: "zero", role: "core", nTrees: 110, maxDepth: 4, minLeaf: 18, seed: 7 }),
		fitG(dense, structuredWt, { ...common, anchor: "elapsed", role: "core", nTrees: 130, maxDepth: 5, minLeaf: 38, seed: 36 }),
		fitG(allDense, r => r.promptWt, { ...common, anchor: "zero", role: "early", nTrees: 130, maxDepth: 5, minLeaf: 22, seed: 101 }),
		fitG(clock, () => 1, { ...common, anchor: "elapsed", role: "core", nTrees: 130, maxDepth: 5, minLeaf: 24, seed: 173 }),
		fitG(longClock, () => 1, { ...common, anchor: "zero", role: "long", nTrees: 110, maxDepth: 4, minLeaf: 18, seed: 293 }),
		fitG(longClock, () => 1, { ...common, anchor: "elapsed", role: "long", nTrees: 110, maxDepth: 4, minLeaf: 18, seed: 257 }),
	];

	const survRows = [...allDense, ...clock];
	const survX = survRows.map(xOf);
	const survTot = survRows.map(r => r.totalS);
	const survEl = survRows.map(r => r.e);
	const survWt = survRows.map(r => (r.sample === "allDense" ? r.promptWt * 4.0 : 0.45));
	const survOpts = { hidden: 64, epochs: 22, batchSize: 128, lr: 0.0025, wd: 0.03, dropout: 0.15 };
	const survivalNets = [
		trainSurvivalNet(survX, survTot, survEl, survWt, { ...survOpts, seed: 1001 }),
		trainSurvivalNet(survX, survTot, survEl, survWt, { ...survOpts, seed: 2001 }),
	];

	const qLow = includeQuantiles ? fitG(allDense, r => r.promptWt, { ...quantiles, seed: 401, quantile: 0.25 }) : undefined;
	const qHigh = includeQuantiles ? fitG(clock, () => 1, { ...quantiles, seed: 409, quantile: 0.75 }) : undefined;
	return { heads, survivalNets, qLow, qHigh };
}

function evalBaseZ(model: Pick<EnsembleModel, "version" | "featureCount" | "heads" | "survivalNets">, row: HorizonRow, tm: TextRidgeModel): number {
	const x = fullX(row, tm);
	const logE = Math.log(row.e + SMOOTH_S);
	const gbdtZ = evalEnsembleZ(model, x, logE, row.todoAnchor, {
		elapsedS: row.e,
		toolCalls: row.toolCalls,
		hasActiveTodo: row.hasActiveTodo,
	});
	if (!model.survivalNets?.length) return gbdtZ;
	let sum = 0;
	for (const net of model.survivalNets) sum += evalSurvivalNetZ(net, x, row.e);
	const mlpZ = sum / model.survivalNets.length;
	const wMlp = 0.15 / (1 + Math.exp(-(row.e - 45) / 15));
	return (1 - wMlp) * gbdtZ + wMlp * mlpZ;
}

// 1. Out-of-fold 5-fold pass on dev sessions to fit causal `slipCenter` and 6×6 `range-data.json`.
console.log(`Loaded ${loaded.length} dev sessions (${clockPrompts} prompts, ${horizonRows.length} horizon ckpts). Running 5-fold OOF calibration...`);
const oofBaseZ = new Float64Array(horizonRows.length);
for (let fold = 0; fold < 5; fold++) {
	const tr = trainRows.filter(r => r.fold !== fold);
	const foldTm = fitTextModel(tr);
	const innerTms = [0, 1, 2, 3, 4].map(inf => (inf === fold ? foldTm : fitTextModel(tr.filter(r => r.fold !== inf))));
	const foldModel = trainPointModels(tr, innerTms, false);
	const env: EnsembleModel = { version: 4, featureCount: 150, heads: foldModel.heads, survivalNets: foldModel.survivalNets };
	for (let i = 0; i < horizonRows.length; i++) {
		const row = horizonRows[i]!;
		if (row.fold === fold) oofBaseZ[i] = evalBaseZ(env, row, foldTm);
	}
	console.log(`  OOF fold ${fold + 1}/5 complete (${((performance.now() - t0) / 1000).toFixed(1)}s)`);
}

// Causal session-bias + raw slippage across OOF horizon checkpoints (in chronological session + prompt order).
const eBin = (e: number) => Math.min(7, Math.max(0, Math.floor(Math.log2(Math.max(1, e / 30)))));
const slipSumAll = new Float64Array(8);
const slipCntAll = new Float64Array(8);
const slipSumFold = Array.from({ length: 5 }, () => new Float64Array(8));
const slipCntFold = Array.from({ length: 5 }, () => new Float64Array(8));
const oofSessBias = new Float64Array(horizonRows.length);
const oofRawSlip = new Float64Array(horizonRows.length);
const oofHasSlip = new Uint8Array(horizonRows.length);

let curSid = "";
let curPid = -1;
let biasHist: number[] = [];
let curSum = 0;
let curN = 0;
let lms: Array<{ e: number; totS: number }> = [];

for (let i = 0; i < horizonRows.length; i++) {
	const r = horizonRows[i]!;
	if (r.sid !== curSid) {
		curSid = r.sid;
		curPid = -1;
		biasHist = [];
		curSum = 0;
		curN = 0;
		lms = [];
	}
	if (r.pid !== curPid) {
		if (curPid >= 0 && curN > 0) {
			biasHist.push(Math.max(-2, Math.min(2, curSum / curN)));
			if (biasHist.length > 8) biasHist.shift();
		}
		curPid = r.pid;
		curSum = 0;
		curN = 0;
		lms = [];
	}
	if (biasHist.length > 0) {
		let s = 0;
		let w = 0;
		for (let k = 0; k < biasHist.length; k++) {
			const wk = Math.pow(0.8, biasHist.length - 1 - k);
			s += wk * biasHist[k]!;
			w += wk;
		}
		oofSessBias[i] = s / (w + 2.5);
	}
	const baseZ = oofBaseZ[i]!;
	const remS = Math.max(0, Math.exp(Math.min(10.5, Math.max(0, baseZ))) - SMOOTH_S);
	const totS = r.e + remS;
	let prevRef: { e: number; totS: number } | null = null;
	for (const lm of lms) {
		if (lm.e <= r.e * 0.55) prevRef = lm;
		else break;
	}
	if (prevRef) {
		const sl = Math.max(-2, Math.min(2, Math.log((totS + SMOOTH_S) / (prevRef.totS + SMOOTH_S))));
		oofRawSlip[i] = sl;
		oofHasSlip[i] = 1;
		const b = eBin(r.e);
		slipSumAll[b]! += sl;
		slipCntAll[b]! += 1;
		for (let f = 0; f < 5; f++) {
			if (f !== r.fold) {
				slipSumFold[f]![b]! += sl;
				slipCntFold[f]![b]! += 1;
			}
		}
	}
	if (lms.length === 0 || r.e - lms[lms.length - 1]!.e >= 14) lms.push({ e: r.e, totS });
	curSum += Math.log(r.y + SMOOTH_S) - baseZ;
	curN++;
}

const slipCenter = Array.from({ length: 8 }, (_, b) => (slipCntAll[b]! > 0 ? round4(slipSumAll[b]! / slipCntAll[b]!) : 0));

const rangeRows = horizonRows.map((r, i) => {
	const b = eBin(r.e);
	const foldMean = slipCntFold[r.fold]![b]! > 0 ? slipSumFold[r.fold]![b]! / slipCntFold[r.fold]![b]! : 0;
	const cSlip = oofHasSlip[i] && r.e >= 60 ? oofRawSlip[i]! - foldMean : 0;
	const z = oofBaseZ[i]! + 0.6 * oofSessBias[i]! + 0.25 * cSlip;
	const predS = Math.max(0, Math.exp(Math.min(10.5, Math.max(0, z))) - SMOOTH_S);
	return { e: r.e, b: predS, y: r.y, sid: r.sid, pid: r.pid };
});
const rangeModel = fitRangeModel(rangeRows);

// 2. Final production model trained on all dev sessions (with strictly out-of-fold inner TextRidge inputs).
console.log(`Training final v4 model on all ${loaded.length} dev sessions...`);
const finalTextModel = fitTextModel(trainRows);
const finalInnerTms = [0, 1, 2, 3, 4].map(f => fitTextModel(trainRows.filter(r => r.fold !== f)));
const finalFitted = trainPointModels(trainRows, finalInnerTms, true);

const model: EnsembleModel = {
	version: 4,
	featureCount: trainRows[0]!.xBase.length + 5,
	heads: finalFitted.heads,
	qLow: finalFitted.qLow,
	qHigh: finalFitted.qHigh,
	textModel: finalTextModel,
	survivalNets: finalFitted.survivalNets,
	slipCenter,
};

const json = JSON.stringify(model);
const outPath = join(project, "src", "model-data.ts");
writeFileSync(
	outPath,
	`// Local v4 hybrid GBDT + Truncated Log-Normal Survival ResMLP ensemble generated by \`bun eval/train.ts\`.\n` +
		`// ${loaded.length} sessions; ${clockPrompts} completed prompts; ${excludedSessions.size} held-out sessions excluded.\n` +
		`import type { EnsembleModel } from "./gbdt";\n\n` +
		`export const MODEL_DATA: EnsembleModel = ${json};\n`,
);

const rangePath = join(project, "src", "range-data.json");
writeFileSync(rangePath, JSON.stringify(rangeModel));

console.log(
	`Trained v4 on ${loaded.length} sessions, ${clockPrompts} completed prompts (${model.featureCount} features); ${excludedSessions.size} held-out sessions excluded in ${((performance.now() - t0) / 1000).toFixed(1)}s.`,
);
console.log(`slipCenter: [${slipCenter.join(", ")}]`);
console.log(`Wrote ${outPath} (${(json.length / 1024).toFixed(1)} KB) and ${rangePath}.`);
