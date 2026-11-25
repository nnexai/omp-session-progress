#!/usr/bin/env bun
// Fit the local survival-regime GBDT ensemble on historic OMP sessions.
// Training is 100% local; the model artifact contains numeric splits/leaves, never session contents.
import { closeSync, openSync, readFileSync, readdirSync, readSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { applyEvent, emptyFacts } from "../src/facts";
import { extractFeatures, SMOOTH_S } from "../src/features";
import { type EnsembleModel, type TrainHeadOptions, trainGBDTHead } from "../src/gbdt";
import { activeSegments, extractSession } from "./extract";
import { allDenseLandmarks, landmarks, type Sample } from "./landmarks";
import { SESSIONS_DIR } from "./paths";

const project = resolve(import.meta.dir, "..");
const excludedSessions = new Set<string>();
if (!process.argv.includes("--include-all")) {
	const state = JSON.parse(readFileSync(join(project, ".omp/ratchet/progress-time/_state.json"), "utf8"));
	if (!state.test_ids?.length) throw new Error("Cannot train without the held-out session split");
	for (const id of state.test_ids as string[]) excludedSessions.add(id.replace(/-p\d+$/, ""));
}

type TrainSample = Sample | "allDense";
type Row = { x: number[]; e: number; y: number; todoAnchor: number; bucket: number; promptWt: number };
const samples: Record<TrainSample, Row[]> = { sparse: [], dense: [], clock: [], allDense: [] };
const counts = [0, 0, 0, 0];
let sessionsUsed = 0, clockPrompts = 0;
const header = Buffer.alloc(4096);
const excludedProjects = ["omp-session-progress", "omp-progress-smoke", "omp-progress-bench", "omp-progress-eval"];
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
		} finally { closeSync(fd); }
		const trace = extractSession(path);
		if (excludedSessions.has(trace.id)) continue;
		const facts = emptyFacts();
		let cursor = 0, used = false;
		for (const prompt of trace.prompts) {
			if (prompt.outcome !== "completed") continue;
			const segments = activeSegments(trace.events, prompt);
			const totalS = segments.reduce((sum, s) => sum + s.ms, 0) / 1000;
			const n = prompt.toolStartIndices.length;
			const bucket = totalS < 60 ? 0 : totalS < 180 ? 1 : totalS <= 900 ? 2 : 3;
			const structured = bucket === 0 ? totalS >= 15 && n >= 2 : n >= 5;
			if (structured) counts[bucket]!++;
			if (totalS <= 5) continue;
			clockPrompts++;
			used = true;
			const ad = allDenseLandmarks(segments);
			const allPts: Array<{ sample: TrainSample; at: number; cut: number; elapsedS: number; remainingS: number; promptWt: number }> = [
				...landmarks(segments, structured).map(p => ({ ...p, promptWt: 1 })),
				...ad.map(p => ({ ...p, sample: "allDense" as const, promptWt: 1 / ad.length })),
			];
			allPts.sort((a, b) => a.elapsedS - b.elapsedS);
			for (const p of allPts) {
				while (cursor < p.cut) {
					const ev = trace.events[cursor++]!;
					applyEvent(facts, ev.event, ev.t);
				}
				const features = extractFeatures(facts, p.at);
				samples[p.sample].push({
					x: features.x,
					todoAnchor: features.anchorZ,
					e: p.elapsedS,
					y: p.remainingS,
					bucket,
					promptWt: p.promptWt,
				});
			}
		}
		if (used) sessionsUsed++;
	}
}

const main = counts[1]! + counts[2]! + counts[3]!;
function fit(rows: Row[], weightFn: (r: Row) => number, options: TrainHeadOptions) {
	const { anchor } = options;
	return trainGBDTHead(
		rows.map(r => r.x),
		rows.map(r => Math.log(r.y + SMOOTH_S)),
		rows.map(r => anchor === "zero" ? 0 : anchor === "elapsed" ? Math.log(r.e + SMOOTH_S) : r.todoAnchor),
		rows.map(r => r.e),
		rows.map(weightFn),
		options,
	);
}

const structuredWt = (r: Row) => (r.bucket === 0 ? 0.15 : Math.sqrt(main / (3 * counts[r.bucket]!)));
const longClockRows = samples.clock.filter(r => r.e >= 300);

const common = { lr: 0.045, colSample: 0.45, l2: 12, maxBins: 64 };
const quantiles = { anchor: "elapsed", nTrees: 45, lr: 0.055, maxDepth: 3, minLeaf: 24, l2: 12, colSample: 0.5, maxBins: 64 } as const;
const model: EnsembleModel = {
	version: 3,
	featureCount: samples.sparse[0]!.x.length,
	heads: [
		fit(samples.sparse, structuredWt, { ...common, anchor: "zero", role: "core", nTrees: 110, maxDepth: 4, minLeaf: 18, seed: 7 }),
		fit(samples.dense, structuredWt, { ...common, anchor: "elapsed", role: "core", nTrees: 130, maxDepth: 5, minLeaf: 38, seed: 36 }),
		fit(samples.allDense, r => r.promptWt, { ...common, anchor: "zero", role: "early", nTrees: 130, maxDepth: 5, minLeaf: 22, seed: 101 }),
		fit(samples.clock, () => 1, { ...common, anchor: "elapsed", role: "core", nTrees: 130, maxDepth: 5, minLeaf: 24, seed: 173 }),
		fit(longClockRows, () => 1, { ...common, anchor: "zero", role: "long", nTrees: 110, maxDepth: 4, minLeaf: 18, seed: 293 }),
		fit(longClockRows, () => 1, { ...common, anchor: "elapsed", role: "long", nTrees: 110, maxDepth: 4, minLeaf: 18, seed: 257 }),
	],
	qLow: fit(samples.allDense, r => r.promptWt, { ...quantiles, seed: 401, quantile: 0.25 }),
	qHigh: fit(samples.clock, () => 1, { ...quantiles, seed: 409, quantile: 0.75 }),
};

const json = JSON.stringify(model);
const outPath = join(project, "src", "model-data.ts");
writeFileSync(
	outPath,
	`// Local survival-regime GBDT ensemble generated by \`bun eval/train.ts\`.\n` +
		`// ${sessionsUsed} sessions; ${clockPrompts} completed prompts; ${excludedSessions.size} held-out sessions excluded.\n` +
		`import type { EnsembleModel } from "./gbdt";\n\n` +
		`export const MODEL_DATA: EnsembleModel = ${json};\n`,
);

console.log(
	`Trained on ${sessionsUsed} sessions, ${clockPrompts} completed prompts (${samples.sparse[0]!.x.length} features); ${excludedSessions.size} held-out sessions excluded.`,
);
console.log(`Wrote ${outPath} (${(json.length / 1024).toFixed(1)} KB).`);
