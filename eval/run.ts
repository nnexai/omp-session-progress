#!/usr/bin/env bun
// Evaluate the four official probes and replay the actual Observer display at
// 15-second active-time checkpoints for the official held-out cases.
// Usage: bun eval/run.ts [--only id,id] [--horizons]

import { existsSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { estimateProgress } from "../src/estimator";
import { applyEvent, emptyFacts } from "../src/facts";
import { activeSegments, extractSession, GAP_CAP_MS, type Trace } from "./extract";
import { gradeCase, gradeProbe, halfway, type Prediction, type Probe } from "./grade";
import { EVAL_DIR } from "./paths";
import { Observer, type Clock } from "../src/observer";

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

const index: Record<string, string> = JSON.parse(readFileSync(join(EVAL_DIR, "index.json"), "utf8"));
const statePath = join(project, ".omp", "ratchet", "progress-time", "_state.json");
const state = existsSync(statePath) ? JSON.parse(readFileSync(statePath, "utf8")) : { train_ids: [], test_ids: [] };
const trainIds = new Set<string>(state.train_ids ?? []);
const testIds = new Set<string>(state.test_ids ?? []);
const cases: Case[] = readFileSync(join(import.meta.dir, "cases.jsonl"), "utf8")
	.split("\n")
	.filter(Boolean)
	.map(l => JSON.parse(l))
	.filter((c: Case) => !only || only.split(",").includes(c.id));

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
	chronoAbstain: number;
	chronoRangeHits: number;
	chronoRangeTotal: number;
	chronoHalfway: number;
}

// The Observer owns the timer and smoothing; this clock only schedules its
// callbacks. At an inactive/saturated gap, wait for the EMA to settle before
// skipping identical ticks, retaining the last tick and its timer alignment.
class ReplayClock implements Clock {
	time = 0;
	private timer: { at: number; fn: () => void } | null = null;
	now() { return this.time; }
	setTimeout(fn: () => void, ms: number) {
		const timer = { at: this.time + ms, fn };
		this.timer = timer;
		return timer;
	}
	clearTimeout(handle: unknown) {
		if (this.timer === handle) this.timer = null;
	}
	advance(to: number, observer: Observer) {
		let stable = 0;
		let previous: number | null | undefined;
		while (this.timer && this.timer.at <= to) {
			const timer = this.timer;
			this.timer = null;
			this.time = timer.at;
			timer.fn();
			const { facts, reading } = observer.report();
			const stationary = facts.asking.size > 0 || (facts.lastAt !== null && this.time >= facts.lastAt + GAP_CAP_MS);
			const remaining = reading?.estimate.remainingMinutes;
			stable = stationary && remaining === previous ? stable + 1 : 0;
			previous = remaining;
			const next = this.timer as { at: number; fn: () => void } | null;
			// Wait for the hidden log-space EMA even if minute conversion clamps.
			if (stable >= 100 && next) {
				const ticks = Math.max(0, Math.floor((to - next.at) / 1_000));
				next.at += ticks * 1_000;
			}
		}
		this.time = to;
	}
}

function replay(trace: Trace, prompt: number) {
	const p = trace.prompts[prompt]!;
	const segs = activeSegments(trace.events, p);
	const totalMs = segs.reduce((a, s) => a + s.ms, 0);
	const clock = new ReplayClock();
	const observer = new Observer(clock);
	observer.enable("replay", () => {});
	let cursor = 0, elapsed = 0, segment = 0;
	let score = 0, w2x = 0, probes = 0, premature = 0, abstain = 0, rangeHits = 0, rangeTotal = 0, halfwayScore = 0;
	for (let target = 15_000; target < totalMs - 5_000; target += 15_000) {
		while (segment < segs.length && elapsed + segs[segment]!.ms < target) elapsed += segs[segment++]!.ms;
		const s = segs[segment]!;
		const at = s.t + target - elapsed;
		// Ticks at the same time precede events; all events at the checkpoint
		// precede sampling. No future event or label enters the Observer.
		while (cursor < trace.events.length && trace.events[cursor]!.t <= at) {
			const event = trace.events[cursor++]!;
			// Earlier ticks affect only a previous prompt's smoothing, which the
			// target prompt's before_agent_start resets. Events still update facts.
			if (cursor - 1 <= p.startIndex) clock.time = event.t;
			else clock.advance(event.t, observer);
			observer.observe("replay", event.event);
		}
		clock.advance(at, observer);
		const estimate = observer.report().reading?.estimate;
		const probe: Probe = { at, cut: cursor, elapsedS: target / 1000, remainingS: (totalMs - target) / 1000 };
		const prediction = { progress: estimate?.progress ?? null, remainingMinutes: estimate?.remainingMinutes ?? null };
		const grade = gradeProbe(probe, prediction);
		score += grade.score;
		w2x += grade.within2x;
		halfwayScore += gradeProbe(probe, halfway(probe)).score;
		probes++;
		if (prediction.progress === null || prediction.remainingMinutes === null) abstain++;
		if (prediction.progress !== null && prediction.progress >= 0.85 && target / totalMs <= 0.65) premature++;
		if (estimate?.lowMinutes != null && estimate.highMinutes != null) {
			rangeTotal++;
			if (probe.remainingS / 60 >= estimate.lowMinutes && probe.remainingS / 60 <= estimate.highMinutes) rangeHits++;
		}
	}
	observer.disable();
	return { score, w2x, probes, premature, abstain, rangeHits, rangeTotal, halfwayScore };
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

	const chronological = replay(trace, c.prompt);

	evals.push({
		c,
		grade: gradeCase(c.probes, preds),
		halfwayGrade: gradeCase(c.probes, hwPreds),
		rangeHits,
		rangeTotal,
		premature,
		probesTotal: c.probes.length,
		chronoScore: chronological.probes ? chronological.score / chronological.probes : 0,
		chronoW2x: chronological.probes ? chronological.w2x / chronological.probes : 0,
		chronoProbes: chronological.probes,
		chronoPremature: chronological.premature,
		chronoAbstain: chronological.abstain,
		chronoRangeHits: chronological.rangeHits,
		chronoRangeTotal: chronological.rangeTotal,
		chronoHalfway: chronological.probes ? chronological.halfwayScore / chronological.probes : 0,
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
// Explicit subsets can be replayed for development; the default is held-out.
const replayEvals = only ? evals : testEvals;
const eligible = replayEvals.filter(e => e.chronoProbes > 0);
if (eligible.length) {
	const mean = (fn: (e: CaseEval) => number) => eligible.reduce((a, e) => a + fn(e), 0) / eligible.length;
	const sum = (fn: (e: CaseEval) => number) => eligible.reduce((a, e) => a + fn(e), 0);
	const probes = sum(e => e.chronoProbes);
	const ranges = sum(e => e.chronoRangeTotal);
	console.log("\nChronological displayed reading (15s active checkpoints, prompt-weighted):");
	console.log(`  prompts=${eligible.length} excluded_without_checkpoints=${replayEvals.length - eligible.length} checkpoints=${probes}`);
	console.log(`  score=${mean(e => e.chronoScore).toFixed(4)} | within2x=${(100 * mean(e => e.chronoW2x)).toFixed(1)}% | halfway score=${mean(e => e.chronoHalfway).toFixed(4)}`);
	console.log(`  premature>=85%=${(100 * mean(e => e.chronoPremature / e.chronoProbes)).toFixed(1)}% | abstentions=${(100 * mean(e => e.chronoAbstain / e.chronoProbes)).toFixed(1)}%`);
	console.log(`  range available=${ranges}/${probes} | coverage=${ranges ? (100 * sum(e => e.chronoRangeHits) / ranges).toFixed(1) + "%" : "n/a"} (checkpoint-weighted among available ranges)`);
}

// Expanded audit: replay each selected session once, in event order. Labels
// determine only where to score and the truth, never what the Observer sees.
if (argv.includes("--horizons")) {
	interface HorizonPrompt {
		checkpoints: number;
		predictions: number;
		errorSum: number;
		misses: number;
		over15: number;
		over30: number;
	}
	const groups = new Map<string, HorizonPrompt[]>();
	const groupNames = ["all completed", "3–15 min", "15+ min", "30+ min", "short <1 min", "running tool >60s"];
	for (const name of groupNames) groups.set(name, []);
	const sessions = new Set((only ? cases : cases.filter(c => testIds.has(c.id))).map(c => c.session));
	if (!only && (testIds.size === 0 || cases.filter(c => testIds.has(c.id)).length !== testIds.size || sessions.size !== testIds.size)) {
		throw new Error("Official test cases must resolve to distinct indexed sessions");
	}
	if (!sessions.size) throw new Error("No sessions selected for horizons audit");
	let completed = 0, tooShort = 0, withoutDisplay = 0;
	for (const session of sessions) {
		const trace = traceOf(session);
		const clock = new ReplayClock();
		const observer = new Observer(clock);
		observer.enable("replay", () => {});
		let cursor = 0;
		const advance = (at: number) => {
			while (cursor < trace.events.length && trace.events[cursor]!.t <= at) {
				const event = trace.events[cursor++]!;
				clock.advance(event.t, observer);
				observer.observe("replay", event.event);
			}
			clock.advance(at, observer);
		};
		for (const prompt of trace.prompts) {
			if (prompt.outcome !== "completed") continue;
			completed++;
			const segments = activeSegments(trace.events, prompt);
			const totalMs = segments.reduce((sum, s) => sum + s.ms, 0);
			const row = (): HorizonPrompt => ({ checkpoints: 0, predictions: 0, errorSum: 0, misses: 0, over15: 0, over30: 0 });
			const overall = row(), longTool = row();
			let elapsed = 0, segment = 0;
			// Strictly before the end; a five-second prompt has no live checkpoint.
			for (let target = 5_000; target < totalMs; target = target === 5_000 ? 15_000 : target + 15_000) {
				while (segment < segments.length && elapsed + segments[segment]!.ms < target) elapsed += segments[segment++]!.ms;
				const s = segments[segment]!;
				const at = s.t + target - elapsed;
				advance(at);
				const { facts, reading } = observer.report();
				const prediction = reading?.estimate.remainingMinutes;
				const runningLongTool = [...facts.underway.values()].some(tool => tool.name !== "ask" && target / 1000 - tool.startedAtS > 60);
				for (const r of runningLongTool ? [overall, longTool] : [overall]) {
					r.checkpoints++;
					if (prediction == null || !Number.isFinite(prediction)) continue;
					const error = Math.abs(prediction - (totalMs - target) / 60_000);
					r.predictions++;
					r.errorSum += error;
					if (error >= 15) r.misses++;
					if (prediction >= 15) r.over15++;
					if (prediction >= 30) r.over30++;
				}
			}
			if (!overall.checkpoints) tooShort++;
			else if (!overall.predictions) withoutDisplay++;
			groups.get("all completed")!.push(overall);
			if (totalMs >= 180_000 && totalMs < 900_000) groups.get("3–15 min")!.push(overall);
			if (totalMs >= 900_000) groups.get("15+ min")!.push(overall);
			if (totalMs >= 1_800_000) groups.get("30+ min")!.push(overall);
			if (totalMs < 60_000) groups.get("short <1 min")!.push(overall);
			if (longTool.checkpoints) groups.get("running tool >60s")!.push(longTool);
		}
		observer.disable();
	}
	console.log(`\nExpanded horizons (${only ? "selected sessions" : "official held-out sessions"}): sessions=${sessions.size} completed_prompts=${completed}`);
	console.log(`  Active checkpoints: 5s, 15s, 30s, ... strictly before completion; too_short_for_checkpoint=${tooShort}; checkpoints_without_any_displayed_ETA_prompts=${withoutDisplay}.`);
	console.log("  ETA error is minutes remaining vs true active minutes remaining. Each prompt is weighted equally after averaging its displayed-ETA checkpoints; missing ETAs are excluded and counted.");
	console.log("  Rare-event prompt-weighted rates average per-prompt checkpoint fractions. Raw counts are event checkpoints / displayed-ETA checkpoints, not distinct prompts; forecast thresholds apply to remaining minutes.");
	console.log("  Running-tool stratum includes only checkpoints with a non-ask tool open for >60 active seconds; prompts are weighted equally within that subset.");
	for (const name of groupNames) {
		const rows = groups.get(name)!;
		const available = rows.filter(r => r.predictions > 0);
		const sum = (fn: (r: HorizonPrompt) => number) => rows.reduce((a, r) => a + fn(r), 0);
		const mean = (fn: (r: HorizonPrompt) => number) => available.length ? available.reduce((a, r) => a + fn(r), 0) / available.length : null;
		const fmt = (n: number | null, suffix = "") => n === null ? "n/a" : n.toFixed(2) + suffix;
		const fraction = (fn: (r: HorizonPrompt) => number) => mean(r => 100 * fn(r) / r.predictions);
		const predictions = sum(r => r.predictions);
		console.log(`  ${name}: prompts=${rows.length} scored_prompts=${available.length} checkpoints=${sum(r => r.checkpoints)} displayed_eta=${predictions} missing_eta=${sum(r => r.checkpoints) - predictions}`);
		console.log(`    prompt-weighted MAE=${fmt(mean(r => r.errorSum / r.predictions), " min")} | >=15min miss=${fmt(fraction(r => r.misses), "%")} (${sum(r => r.misses)}/${predictions}) | forecast >=15min=${fmt(fraction(r => r.over15), "%")} (${sum(r => r.over15)}/${predictions}) | forecast >=30min=${fmt(fraction(r => r.over30), "%")} (${sum(r => r.over30)}/${predictions})`);
	}
}

// Keep the recorded four-probe comparison on the same official held-out set.
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
