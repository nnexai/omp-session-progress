#!/usr/bin/env bun
// Trains the local zero-LLM progress estimator (3-anchor GBDT ensemble) from
// historic local OMP sessions and writes `src/model-data.ts`.
//
// Strictly excludes every session present in the held-out test split
// (`.omp/ratchet/progress-time/_state.json` test_ids) unless `--include-all`
// is explicitly passed, so held-out evaluation stays 100% leak-free.
//
//   bun eval/train.ts [--include-all]

import { existsSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { applyEvent, emptyFacts } from "../src/facts";
import { extractFeatures, SMOOTH_S } from "../src/features";
import { type AnchorKind, type EnsembleModel, trainGBDTHead } from "../src/gbdt";
import { activeSegments, extractSession, type PromptLabel, type TraceEvent } from "./extract";
import type { Probe } from "./grade";
import { SESSIONS_DIR } from "./paths";

const EXCLUDE_CWD = ["omp-session-progress", "omp-progress-smoke", "omp-progress-bench", "omp-progress-eval"];
const FRACTIONS = [0.15, 0.4, 0.65, 0.9];
const SUB60_WEIGHT = 0.15;
const BAL_POWER = 0.5;

const project = resolve(import.meta.dir, "..");
const includeAll = process.argv.includes("--include-all");

const excludedSessions = new Set<string>();
const statePath = join(project, ".omp", "ratchet", "progress-time", "_state.json");
if (!includeAll && existsSync(statePath)) {
	const state = JSON.parse(readFileSync(statePath, "utf8"));
	for (const id of (state.test_ids ?? []) as string[]) {
		excludedSessions.add(id.replace(/-p\d+$/, ""));
	}
}

const officialProbes = new Map<string, Probe[]>();
const casesPath = join(import.meta.dir, "cases.jsonl");
if (existsSync(casesPath)) {
	for (const line of readFileSync(casesPath, "utf8").trim().split("\n").filter(Boolean)) {
		const c = JSON.parse(line);
		officialProbes.set(c.id, c.probes);
	}
}

function makeProbes(events: TraceEvent[], p: PromptLabel): Probe[] {
	const segments = activeSegments(events, p);
	const total = segments.reduce((a, s) => a + s.ms, 0);
	const out: Probe[] = [];
	for (const f of FRACTIONS) {
		const target = f * total;
		let acc = 0;
		for (const s of segments) {
			if (s.ms > 0 && acc + s.ms >= target) {
				out.push({ at: Math.round(s.t + (target - acc)), cut: s.index + 1, elapsedS: target / 1000, remainingS: (total - target) / 1000 });
				break;
			}
			acc += s.ms;
		}
	}
	return out;
}

interface TrainRow {
	x: number[];
	logElapsed: number;
	anchorZ: number;
	elapsedS: number;
	yLogRem: number;
	bucket: "sub60" | "short" | "medium" | "long";
}

const rows: TrainRow[] = [];
let sessionsUsed = 0;
let eligiblePrompts = 0;
let sub60Prompts = 0;

for (const dir of readdirSync(SESSIONS_DIR)) {
	const full = join(SESSIONS_DIR, dir);
	if (!statSync(full).isDirectory()) continue;
	for (const file of readdirSync(full)) {
		if (!file.endsWith(".jsonl")) continue;
		const path = join(full, file);
		const st = statSync(path);
		if (st.size > 64 * 1024 * 1024 || st.size < 200) continue;
		const head = readFileSync(path, "utf8").slice(0, 4096);
		if (EXCLUDE_CWD.some(s => head.includes(s))) continue;

		const trace = extractSession(path);
		if (excludedSessions.has(trace.id)) continue;

		let usedSession = false;
		for (let pi = 0; pi < trace.prompts.length; pi++) {
			const p = trace.prompts[pi]!;
			const caseId = `${trace.id}-p${pi}`;
			const offProbes = officialProbes.get(caseId);
			if (!offProbes && p.outcome !== "completed") continue;

			const activeS = activeSegments(trace.events, p).reduce((a, s) => a + s.ms, 0) / 1000;
			const isEligible = Boolean(offProbes) || (p.toolStartIndices.length >= 5 && activeS >= 60);
			const isSub60 = !isEligible && p.toolStartIndices.length >= 2 && activeS >= 15 && activeS < 60;
			if (!isEligible && !isSub60) continue;

			const probes = offProbes ?? makeProbes(trace.events, p);
			if (probes.length === 0) continue;

			const bucket: TrainRow["bucket"] = isSub60 ? "sub60" : activeS < 180 ? "short" : activeS <= 900 ? "medium" : "long";
			const facts = emptyFacts();
			let cursor = 0;
			for (const pr of probes) {
				while (cursor < pr.cut && cursor < trace.events.length) {
					const ev = trace.events[cursor]!;
					applyEvent(facts, ev.event, ev.t);
					cursor++;
				}
				const feat = extractFeatures(facts, pr.at);
				rows.push({
					x: feat.x,
					logElapsed: feat.logElapsed,
					anchorZ: feat.anchorZ,
					elapsedS: pr.elapsedS,
					yLogRem: Math.log(pr.remainingS + SMOOTH_S),
					bucket,
				});
			}
			usedSession = true;
			if (isEligible) eligiblePrompts++;
			else sub60Prompts++;
		}
		if (usedSession) sessionsUsed++;
	}
}

const mainRows = rows.filter(r => r.bucket !== "sub60");
const counts = { short: 0, medium: 0, long: 0 };
for (const r of mainRows) counts[r.bucket as keyof typeof counts]++;

const wts = rows.map(r => {
	if (r.bucket === "sub60") return SUB60_WEIGHT;
	const balW = mainRows.length / (3 * counts[r.bucket as keyof typeof counts]);
	return Math.pow(balW, BAL_POWER);
});

const X = rows.map(r => r.x);
const yZ = rows.map(r => r.yLogRem);
const el = rows.map(r => r.elapsedS);
const anchors: AnchorKind[] = ["zero", "elapsed", "anchor"];

const heads = anchors.map((anchor, idx) => {
	const ancVals = rows.map(r => (anchor === "zero" ? 0 : anchor === "elapsed" ? r.logElapsed : r.anchorZ));
	return trainGBDTHead(X, yZ, ancVals, el, wts, {
		anchor,
		nTrees: 70,
		lr: 0.055,
		maxDepth: 4,
		minLeaf: 18,
		l2: 10,
		colSample: 0.65,
		seed: 29 * idx + 7,
	});
});

const elapsedAnchor = rows.map(r => r.logElapsed);
const qLow = trainGBDTHead(X, yZ, elapsedAnchor, el, wts, {
	anchor: "elapsed",
	nTrees: 40,
	lr: 0.06,
	maxDepth: 3,
	minLeaf: 22,
	l2: 12,
	colSample: 0.65,
	seed: 401,
	quantile: 0.25,
});
const qHigh = trainGBDTHead(X, yZ, elapsedAnchor, el, wts, {
	anchor: "elapsed",
	nTrees: 40,
	lr: 0.06,
	maxDepth: 3,
	minLeaf: 22,
	l2: 12,
	colSample: 0.65,
	seed: 409,
	quantile: 0.75,
});

const model: EnsembleModel = {
	version: 1,
	featureCount: X[0]!.length,
	heads,
	qLow,
	qHigh,
};

const json = JSON.stringify(model);
const outPath = join(project, "src", "model-data.ts");
writeFileSync(
	outPath,
	`// Pre-trained 3-anchor GBDT ensemble generated by \`bun eval/train.ts\`.\n` +
		`// Trained on ${sessionsUsed} local OMP sessions (${eligiblePrompts} prompts >=60s + ${sub60Prompts} short prompts 15-60s;\n` +
		`// ${excludedSessions.size} held-out test sessions strictly excluded).\n` +
		`import type { EnsembleModel } from "./gbdt";\n\n` +
		`export const MODEL_DATA: EnsembleModel = ${json};\n`,
);

console.log(
	`Trained on ${sessionsUsed} sessions (${eligiblePrompts} eligible + ${sub60Prompts} sub-60s prompts = ${rows.length} probes; excluded ${excludedSessions.size} held-out sessions).`,
);
console.log(`Wrote ${outPath} (${(json.length / 1024).toFixed(1)} KB).`);
