// Scores replayed readings against the offline labels, and the no-model
// baseline. Labels use the whole session; readings only ever saw a prefix.
//
// Rubric (reproducible; see README "Replay benchmark"):
// - progress p at time t of a prompt: with todo use in the prompt and items
//   finished in it, the share of the prompt's finished items done by t;
//   otherwise the share of the prompt's tool calls started by t.
// - truth stage while the turn runs: p < .10 just started, < .35 early,
//   < .65 middle, < .90 late, else verifying / wrapping up (one rank).
//   After the turn ends: "done" when it ended normally, "stopped" when it was
//   aborted or failed.
// - structured prompt: ≥2 listed items in the request or todo use; otherwise
//   open-ended, where abstaining (stage null) is the right answer.

import { type Reading, STAGES, type Stage } from "../src/observer";
import type { Action, PromptLabel, Trace } from "./extract";
import type { ReplayResult } from "./replay";

const RANK: Record<Stage, number> = {
	"just started": 0,
	early: 1,
	middle: 2,
	late: 3,
	verifying: 4,
	"wrapping up": 4,
	"appears done": 5,
};

export const stageRank = (stage: Stage) => RANK[stage];

/** Progress from todo items when the prompt finished some, else from tool calls; `toolsOnly` always uses tool calls. */
export function progressAt(p: PromptLabel, t: number, toolsOnly = false): number | null {
	const doneAt = (time: number) => {
		let done = p.todoDone[0]?.done ?? 0;
		for (const x of p.todoDone) if (x.t <= time) done = x.done;
		return done;
	};
	const start = p.todoDone[0]?.done ?? 0;
	const gained = doneAt(p.end) - start;
	if (!toolsOnly && p.todoActive && gained > 0) return Math.max(0, Math.min(1, (doneAt(t) - start) / gained));
	if (!p.toolStarts.length) return null;
	return p.toolStarts.filter(x => x <= t).length / p.toolStarts.length;
}
export const rankOfProgress = (p: number) => (p < 0.1 ? 0 : p < 0.35 ? 1 : p < 0.65 ? 2 : p < 0.9 ? 3 : 4);

/** Coarse action class of a free-text next-step guess; null when it names none. */
export function actionOfText(text: string): Action | null {
	const s = text.toLowerCase();
	const rules: Array<[Action, RegExp]> = [
		["ask", /\b(ask|clarif|wait(s|ing)? for (the )?user|confirm with)/],
		["delegate", /\b(delegat|subagent|spawn)/],
		["plan", /\b(todo|plan|mark .* done|update the (task|todo))/],
		["research", /\b(web|search online|look up docs|fetch)/],
		["run", /\b(run|test|build|execut|lint|compil|verif|smoke|check|commit|push|install|deploy|launch|restart)/],
		["edit", /\b(edit|implement|writ|fix|add|updat|refactor|chang|modif|creat|patch|apply|remov|renam)/],
		["inspect", /\b(read|inspect|look|review|search|find|explor|examin|grep|investigat|analy|open|scan)/],
		["reply", /\b(report|summar|repl|respond|final|wrap|answer|hand ?off|deliver|present)/],
	];
	for (const [action, pattern] of rules) if (pattern.test(s)) return action;
	return null;
}

export interface Checkpoint {
	trace: string;
	prompt: number;
	at: number;
	running: boolean;
	structured: boolean;
	outcome: PromptLabel["outcome"];
	truth: number | null; // rank 0..4 while running, 5 = done, null = stopped/unknown
	truthTools: number | null; // the same with tool-call progress only (robustness: no todo-derived truth)
	progress: number | null;
	predicted: number | null; // stage rank, null = abstained
	premature: boolean;
	nextGuess: Action | "unknown" | "unclassified";
	nextActual: Action | null;
}

export function checkpoints(trace: Trace, result: ReplayResult): Checkpoint[] {
	const out: Checkpoint[] = [];
	for (const shown of result.readings) {
		const label = trace.prompts[shown.prompt];
		if (!label) continue;
		const reading = shown.reading;
		const t = reading.at;
		const running = t < label.end || label.outcome === "superseded" || label.outcome === "unfinished";
		const progress = progressAt(label, t);
		const truth = running ? (progress === null ? null : rankOfProgress(progress)) : label.outcome === "completed" ? 5 : null;
		const predicted = reading.estimate.stage ? stageRank(reading.estimate.stage) : null;
		const premature =
			predicted !== null &&
			((running && progress !== null && ((predicted >= 4 && progress < 0.65) || (predicted === 5 && progress < 0.9))) ||
				(!running && predicted === 5 && (label.outcome === "aborted" || label.outcome === "error")));
		const nextActual = running ? (label.actions.find(a => a.t > t)?.action ?? "reply") : null;
		const nextGuess = reading.next === "unknown" ? "unknown" : (actionOfText(reading.next) ?? "unclassified");
		const tools = progressAt(label, t, true);
		const truthTools = running ? (tools === null ? null : rankOfProgress(tools)) : truth;
		out.push({ trace: trace.id, prompt: shown.prompt, at: t, running, structured: label.structured, outcome: label.outcome, truth, truthTools, progress, predicted, premature, nextGuess, nextActual });
	}
	return out;
}

export interface Metrics {
	checkpoints: number;
	structuredRunning: number;
	coverage: number; // stage given, structured & running
	exact: number;
	within1: number;
	within1Tools: number; // within 1 against tool-call progress only
	mae: number;
	openRunning: number;
	openAbstain: number; // abstained on open-ended running prompts
	abstentionPrecision: number; // abstentions that were on open-ended prompts
	prematureRate: number; // premature completion claims per stage-giving checkpoint
	prematureCount: number;
	endedCompleted: number;
	doneRecall: number; // "appears done" after a normally ended structured prompt
	nextCoverage: number;
	nextAccuracy: number;
	displayCoverage: number; // time-weighted, structured running prompts
	displayWithin1: number;
}

const ratio = (a: number, b: number) => (b ? a / b : Number.NaN);

/** Time-weighted view: what the widget showed every 5 s of a running structured prompt. */
function display(trace: Trace, result: ReplayResult): { samples: number; covered: number; within1: number } {
	let samples = 0;
	let covered = 0;
	let within1 = 0;
	trace.prompts.forEach((label, index) => {
		if (!label.structured) return;
		const shown = result.readings.filter(r => r.prompt === index);
		for (let t = label.start + 5_000; t < label.end; t += 5_000) {
			samples += 1;
			let current: Reading | null = null;
			for (const r of shown) if (r.reading.at <= t) current = r.reading;
			const stage = current?.estimate.stage;
			const progress = progressAt(label, t);
			if (!stage || progress === null) continue;
			covered += 1;
			if (Math.abs(stageRank(stage) - rankOfProgress(progress)) <= 1) within1 += 1;
		}
	});
	return { samples, covered, within1 };
}

export function metrics(pairs: Array<{ trace: Trace; result: ReplayResult }>): Metrics {
	const cps = pairs.flatMap(({ trace, result }) => checkpoints(trace, result));
	const sr = cps.filter(c => c.structured && c.running && c.truth !== null);
	const covered = sr.filter(c => c.predicted !== null);
	const err = covered.map(c => Math.abs((c.predicted as number) - (c.truth as number)));
	const open = cps.filter(c => !c.structured && c.running);
	const abstained = cps.filter(c => c.predicted === null);
	const giving = cps.filter(c => c.predicted !== null);
	const ended = cps.filter(c => !c.running && c.structured && c.outcome === "completed");
	const nextable = cps.filter(c => c.running && c.nextActual);
	const guessed = nextable.filter(c => c.nextGuess !== "unknown");
	const disp = pairs.map(({ trace, result }) => display(trace, result)).reduce((a, b) => ({ samples: a.samples + b.samples, covered: a.covered + b.covered, within1: a.within1 + b.within1 }), { samples: 0, covered: 0, within1: 0 });
	return {
		checkpoints: cps.length,
		structuredRunning: sr.length,
		coverage: ratio(covered.length, sr.length),
		exact: ratio(err.filter(e => e === 0).length, covered.length),
		within1: ratio(err.filter(e => e <= 1).length, covered.length),
		within1Tools: ratio(covered.filter(c => c.truthTools !== null && Math.abs((c.predicted as number) - c.truthTools) <= 1).length, covered.filter(c => c.truthTools !== null).length),
		mae: ratio(err.reduce((a, b) => a + b, 0), err.length),
		openRunning: open.length,
		openAbstain: ratio(open.filter(c => c.predicted === null).length, open.length),
		abstentionPrecision: ratio(abstained.filter(c => !c.structured).length, abstained.length),
		prematureRate: ratio(giving.filter(c => c.premature).length, giving.length),
		prematureCount: giving.filter(c => c.premature).length,
		endedCompleted: ended.length,
		doneRecall: ratio(ended.filter(c => c.predicted === 5).length, ended.length),
		nextCoverage: ratio(guessed.length, nextable.length),
		nextAccuracy: ratio(guessed.filter(c => c.nextGuess === c.nextActual).length, guessed.length),
		displayCoverage: ratio(disp.covered, disp.samples),
		displayWithin1: ratio(disp.within1, disp.covered),
	};
}

export interface Cost {
	calls: number;
	modelCalls: number;
	failures: number;
	skippedUnchanged: number;
	cappedSessions: number;
	inputTokens: number;
	cacheRead: number;
	cacheWrite: number;
	outputTokens: number;
	costUsd: number;
	meanPayloadChars: number;
	maxPayloadChars: number;
	p50Ms: number;
	p90Ms: number;
}

export function cost(results: ReplayResult[]): Cost {
	const calls = results.flatMap(r => r.calls);
	const model = calls.filter(c => c.model);
	const ms = model.map(c => c.ms ?? 0).sort((a, b) => a - b);
	const sum = (f: (c: (typeof calls)[number]) => number | undefined) => model.reduce((a, c) => a + (f(c) ?? 0), 0);
	const pct = (q: number) => (ms.length ? (ms[Math.min(ms.length - 1, Math.floor(q * ms.length))] ?? 0) : 0);
	return {
		calls: calls.length,
		modelCalls: model.length,
		failures: calls.filter(c => c.error).length,
		skippedUnchanged: results.reduce((a, r) => a + r.skippedUnchanged, 0),
		cappedSessions: results.filter(r => r.capped).length,
		inputTokens: sum(c => c.inputTokens),
		cacheRead: sum(c => c.cacheRead),
		cacheWrite: sum(c => c.cacheWrite),
		outputTokens: sum(c => c.outputTokens),
		costUsd: sum(c => c.costUsd),
		meanPayloadChars: ratio(calls.reduce((a, c) => a + c.payloadChars, 0), calls.length),
		maxPayloadChars: calls.reduce((a, c) => Math.max(a, c.payloadChars), 0),
		p50Ms: pct(0.5),
		p90Ms: pct(0.9),
	};
}

/** No model: stage from the todo ratio alone, otherwise abstain. */
export function baselineReply(payload: string): string {
	const doc = JSON.parse(payload) as { current_prompt?: { turn?: string }; todo_list?: { done: number; total: number } | null };
	const todo = doc.todo_list;
	const running = doc.current_prompt?.turn === "agent working";
	let stage: Stage | null = null;
	if (todo && todo.total > 0) {
		const r = todo.done / todo.total;
		if (!running) stage = todo.done === todo.total ? "appears done" : null;
		else stage = STAGES[rankOfProgress(r) === 4 ? 5 : rankOfProgress(r)] ?? null;
	}
	return JSON.stringify({ summary: "local baseline", estimate: { scope: "current prompt", stage, basis: stage ? "todo ratio" : "no todo list" }, blocker: null, next: "unknown" });
}
