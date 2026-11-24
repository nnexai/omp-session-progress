#!/usr/bin/env bun
// Evaluates the local progress estimator against historic OMP sessions in
// `eval/cases.jsonl` (using the exact same `eval/grade.ts` grader as the
// earlier `@widget` LLM ratchet runs), PLUS a chronological live-replay audit
// every 15s of active time measuring:
//   - score, progress accuracy, remaining accuracy, within-2x
//   - calibrated [P25, P75] range coverage
//   - premature "nearly done" rate (>= 85% predicted when > 35% time remains)
//   - latency per inference
//   - paired head-to-head comparison vs the recorded `@widget` LLM baseline
//
//   bun eval/run.ts [--only id,id]

import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { estimateProgress } from "../src/estimator";
import { applyEvent, emptyFacts } from "../src/facts";
import { activeSegments, extractSession, type Trace } from "./extract";
import { gradeCase, gradeProbe, halfway, type Prediction, type Probe } from "./grade";
import { EVAL_DIR } from "./paths";

interface Case {
	id: string;
	session: string;
	prompt: number;
	tools: number;
	activeS: number;
	tags: string[];
	probes: Probe[];
}

const project = resolve(import.meta.dir, "..");
const argv = process.argv.slice(2);
const onlyIdx = argv.indexOf("--only");
const only = onlyIdx >= 0 ? (argv[onlyIdx + 1] ?? "") : "";

const cases: Case[] = readFileSync(join(import.meta.dir, "cases.jsonl"), "utf8")
	.split("\n")
	.filter(Boolean)
	.map(l => JSON.parse(l))
	.filter((c: Case) => !only || only.split(",").includes(c.id));

const index: Record<string, string> = JSON.parse(readFileSync(join(EVAL_DIR, "index.json"), "utf8"));
const statePath = join(project, ".omp", "ratchet", "progress-time", "_state.json");
const state = existsSync(statePath) ? JSON.parse(readFileSync(statePath, "utf8")) : { train_ids: [], test_ids: [] };
const trainIds = new Set<string>(state.train_ids ?? []);
const testIds = new Set<string>(state.test_ids ?? []);

const traces = new Map<string, Trace>();
const traceOf = (session: string) => {
	let t = traces.get(session);
	if (!t) {
		const path = index[session];
		if (!path) throw new Error(`no session file for ${session}`);
		t = extractSession(path);
		traces.set(session, t);
	}
	return t;
};

interface CaseEval {
	c: Case;
	grade: ReturnType<typeof gradeCase>;
	halfwayGrade: ReturnType<typeof gradeCase>;
	rangeHits: number;
	rangeTotal: number;
	premature: number;
	probesTotal: number;
	chronoScore: number;
	chronoW2x: number;
	chronoProbes: number;
	chronoPremature: number;
	chronoJitter: number;
}

const tStart = performance.now();
let inferCount = 0;
let inferMs = 0;
const evals: CaseEval[] = [];

for (const c of cases) {
	const trace = traceOf(c.session);
	const facts = emptyFacts();
	let cursor = 0;
	const preds: Prediction[] = [];
	const hwPreds: Prediction[] = [];
	let rangeHits = 0;
	let rangeTotal = 0;
	let premature = 0;

	for (const probe of c.probes) {
		while (cursor < probe.cut && cursor < trace.events.length) {
			const ev = trace.events[cursor]!;
			applyEvent(facts, ev.event, ev.t);
			cursor++;
		}
		const t0 = performance.now();
		const r = estimateProgress(facts, probe.at);
		inferMs += performance.now() - t0;
		inferCount++;

		preds.push({ progress: r.estimate.progress, remainingMinutes: r.estimate.remainingMinutes });
		hwPreds.push(halfway(probe));

		if (r.estimate.lowMinutes !== null && r.estimate.highMinutes !== null) {
			rangeTotal++;
			const remMin = probe.remainingS / 60;
			if (remMin >= r.estimate.lowMinutes && remMin <= r.estimate.highMinutes) rangeHits++;
		}
		const trueProg = probe.elapsedS / (probe.elapsedS + probe.remainingS);
		if (r.estimate.progress !== null && r.estimate.progress >= 0.85 && trueProg <= 0.65) premature++;
	}

	// Chronological every-15s live replay on this prompt
	const p = trace.prompts[c.prompt]!;
	const segs = activeSegments(trace.events, p);
	const totalMs = segs.reduce((a, s) => a + s.ms, 0);
	const cFacts = emptyFacts();
	let cCursor = 0;
	let cScore = 0;
	let cW2x = 0;
	let cProbes = 0;
	let cPremature = 0;
	let cJitter = 0;
	let prevProg: number | null = null;

	for (let targetMs = 15_000; targetMs < totalMs - 5_000; targetMs += 15_000) {
		let acc = 0;
		for (const s of segs) {
			if (s.ms > 0 && acc + s.ms >= targetMs) {
				const at = Math.round(s.t + (targetMs - acc));
				const cut = s.index + 1;
				while (cCursor < cut && cCursor < trace.events.length) {
					const ev = trace.events[cCursor]!;
					applyEvent(cFacts, ev.event, ev.t);
					cCursor++;
				}
				const r = estimateProgress(cFacts, at);
				const probe: Probe = { at, cut, elapsedS: targetMs / 1000, remainingS: (totalMs - targetMs) / 1000 };
				const g = gradeProbe(probe, { progress: r.estimate.progress, remainingMinutes: r.estimate.remainingMinutes });
				cScore += g.score;
				cW2x += g.within2x;
				cProbes++;
				const trueProg = targetMs / totalMs;
				if (r.estimate.progress !== null) {
					if (r.estimate.progress >= 0.85 && trueProg <= 0.65) cPremature++;
					if (prevProg !== null) cJitter += Math.abs(r.estimate.progress - prevProg);
					prevProg = r.estimate.progress;
				}
				break;
			}
			acc += s.ms;
		}
	}

	evals.push({
		c,
		grade: gradeCase(c.probes, preds),
		halfwayGrade: gradeCase(c.probes, hwPreds),
		rangeHits,
		rangeTotal,
		premature,
		probesTotal: c.probes.length,
		chronoScore: cProbes ? cScore / cProbes : 0,
		chronoW2x: cProbes ? cW2x / cProbes : 0,
		chronoProbes: cProbes,
		chronoPremature: cPremature,
		chronoJitter: cProbes > 1 ? cJitter / (cProbes - 1) : 0,
	});
}

function summarize(label: string, list: CaseEval[]) {
	if (!list.length) return;
	const n = list.length;
	const mean = (fn: (e: CaseEval) => number) => list.reduce((a, e) => a + fn(e), 0) / n;
	const rHits = list.reduce((a, e) => a + e.rangeHits, 0);
	const rTot = list.reduce((a, e) => a + e.rangeTotal, 0);
	const prem = list.reduce((a, e) => a + e.premature, 0);
	const pTot = list.reduce((a, e) => a + e.probesTotal, 0);
	console.log(
		`${label.padEnd(20)} n=${String(n).padStart(3)} | score=${mean(e => e.grade.score).toFixed(4)} (prog=${mean(e => e.grade.progress).toFixed(4)} rem=${mean(e => e.grade.remaining).toFixed(4)} w2x=${(mean(e => e.grade.within2x) * 100).toFixed(1)}%) | P25-75 cov=${((100 * rHits) / Math.max(1, rTot)).toFixed(0)}% | prem>=85%=${((100 * prem) / Math.max(1, pTot)).toFixed(1)}%`,
	);
}

console.log(`Local GBDT eval (${evals.length} cases, ${inferCount} probes, ${(inferMs / Math.max(1, inferCount)).toFixed(3)} ms/estimate, total ${((performance.now() - tStart) / 1000).toFixed(2)}s)\n`);
summarize("ALL", evals);
summarize("TRAIN (60)", evals.filter(e => trainIds.has(e.c.id)));
const testEvals = evals.filter(e => testIds.has(e.c.id));
summarize("HELD-OUT TEST (40)", testEvals);
console.log("");
for (const tag of ["short", "medium", "long", "todo", "no-todo", "asked", "no-ask"]) {
	summarize(`TEST [${tag}]`, testEvals.filter(e => e.c.tags.includes(tag)));
}

// Chronological every-15s live-replay summary on held-out test cases
if (testEvals.length) {
	const totP = testEvals.reduce((a, e) => a + e.chronoProbes, 0);
	const totPrem = testEvals.reduce((a, e) => a + e.chronoPremature, 0);
	const meanCScore = testEvals.reduce((a, e) => a + e.chronoScore, 0) / testEvals.length;
	const meanCW2x = testEvals.reduce((a, e) => a + e.chronoW2x, 0) / testEvals.length;
	const meanJitter = testEvals.reduce((a, e) => a + e.chronoJitter, 0) / testEvals.length;
	const hwTest = testEvals.reduce((a, e) => a + e.halfwayGrade.score, 0) / testEvals.length;
	console.log("\nChronological live-replay (every 15s on 40 held-out sessions):");
	console.log(
		`  probes=${totP} | score=${meanCScore.toFixed(4)} | within2x=${(meanCW2x * 100).toFixed(1)}% | premature>=85%=${((100 * totPrem) / Math.max(1, totP)).toFixed(1)}% | avg 15s step=${(meanJitter * 100).toFixed(1)} pts`,
	);
	console.log(`  halfway baseline on 40 held-out (4-probe): ${hwTest.toFixed(4)}`);
}

// Head-to-head vs recorded shipped LLM (@widget recorded role) on the exact same 40 held-out cases
const llmPath = join(project, ".omp", "ratchet", "progress-cost", "v6", "results.jsonl");
if (existsSync(llmPath) && testEvals.length) {
	const rows = readFileSync(llmPath, "utf8")
		.trim()
		.split("\n")
		.filter(Boolean)
		.map(l => JSON.parse(l))
		.filter(r => testIds.has(r.prompt_id));
	const byId = new Map<string, number[]>();
	for (const r of rows) {
		const a = byId.get(r.prompt_id) ?? [];
		a.push(r.grade.score);
		byId.set(r.prompt_id, a);
	}
	let llmSum = 0;
	let localSum = 0;
	let better = 0;
	let worse = 0;
	for (const e of testEvals) {
		const arr = byId.get(e.c.id);
		if (!arr?.length) continue;
		const llm = arr.reduce((a, b) => a + b, 0) / arr.length;
		llmSum += llm;
		localSum += e.grade.score;
		if (e.grade.score > llm) better++;
		else if (e.grade.score < llm) worse++;
	}
	const n = testEvals.length;
	console.log(
		`\nPaired vs recorded @widget LLM (progress-cost/v6) on ${n} held-out cases:\n  local=${(localSum / n).toFixed(4)} vs LLM=${(llmSum / n).toFixed(4)} (delta=${localSum >= llmSum ? "+" : ""}${((localSum - llmSum) / n).toFixed(4)}; better/worse=${better}/${worse})`,
	);
}
