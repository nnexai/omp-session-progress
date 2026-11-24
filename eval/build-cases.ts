#!/usr/bin/env bun
// Picks the eval cases from local OMP sessions (read-only) and writes
// eval/cases.jsonl: pseudonymous ids, probe times and truth durations only.
// The pseudonym → session file index stays outside the repository in
// $OMP_PROGRESS_EVAL_DIR (default ~/.cache/omp-session-progress-eval).
//
// A case is one completed historic prompt with at least 5 tool calls and at
// least 1 minute of active time (extract.ts `activeSegments`: time waiting on
// the user in `ask` calls left out, silent gaps capped), probed at about 15%,
// 40%, 65% and 90% of its active time. Strata (tags[0]) by active time: short
// < 3 min, medium 3–15 min, long > 15 min; tags[1] is todo / no-todo; tags[2]
// is asked / no-ask (the prompt waited on the user). At most 2 prompts per
// session; order is a hash of the ids, so the pick is reproducible.
//
//   bun eval/build-cases.ts [--n 100]

import { createHash } from "node:crypto";
import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { activeSegments, extractSession, type PromptLabel, type TraceEvent } from "./extract";
import { EVAL_DIR, SESSIONS_DIR } from "./paths";
import type { Probe } from "./grade";

const n = Number(process.argv[process.argv.indexOf("--n") + 1] || 100);
const EXCLUDE_CWD = ["omp-session-progress", "omp-progress-smoke", "omp-progress-bench", "omp-progress-eval"];
const FRACTIONS = [0.15, 0.4, 0.65, 0.9];

type Bucket = "short" | "medium" | "long";
const bucket = (activeS: number): Bucket => (activeS < 180 ? "short" : activeS <= 900 ? "medium" : "long");
const order = (key: string) => createHash("sha256").update(`omp-progress-eval-order:${key}`).digest("hex");

interface Candidate {
	id: string;
	session: string;
	prompt: number;
	tools: number;
	activeS: number;
	tags: string[];
	probes: Probe[];
	path: string;
}

/** Probes at the given shares of the prompt's active time. */
function probes(events: TraceEvent[], p: PromptLabel): Probe[] {
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

const now = Date.now();
const candidates: Candidate[] = [];
for (const dir of readdirSync(SESSIONS_DIR)) {
	const full = join(SESSIONS_DIR, dir);
	if (!statSync(full).isDirectory()) continue;
	for (const file of readdirSync(full)) {
		if (!file.endsWith(".jsonl")) continue;
		const path = join(full, file);
		const st = statSync(path);
		if (now - st.mtimeMs < 6 * 3_600_000 || st.size > 64 * 1024 * 1024) continue;
		const head = readFileSync(path, "utf8").slice(0, 4096);
		if (EXCLUDE_CWD.some(s => head.includes(s))) continue;
		const trace = extractSession(path);
		const picked = trace.prompts
			.map((p, prompt) => ({ p, prompt }))
			.filter(({ p }) => p.outcome === "completed" && p.toolStartIndices.length >= 5)
			.map(({ p, prompt }) => ({ p, prompt, activeS: activeSegments(trace.events, p).reduce((a, s) => a + s.ms, 0) / 1000 }))
			.filter(({ activeS }) => activeS >= 60)
			.sort((a, b) => order(`${trace.id}:${a.prompt}`).localeCompare(order(`${trace.id}:${b.prompt}`)))
			.slice(0, 2);
		for (const { p, prompt, activeS } of picked) {
			const asked = trace.events.slice(p.startIndex, p.endIndex).some(e => e.event.type === "tool_execution_start" && e.event.toolName === "ask");
			candidates.push({
				id: `${trace.id}-p${prompt}`,
				session: trace.id,
				prompt,
				tools: p.toolStartIndices.length,
				activeS: Math.round(activeS),
				tags: [bucket(activeS), p.todoActive ? "todo" : "no-todo", asked ? "asked" : "no-ask"],
				probes: probes(trace.events, p),
				path,
			});
		}
	}
}

const per: Record<Bucket, Candidate[]> = { short: [], medium: [], long: [] };
for (const c of candidates.sort((a, b) => order(a.id).localeCompare(order(b.id)))) per[c.tags[0] as Bucket].push(c);
const quota = Math.floor(n / 3);
const chosen = [...per.short.slice(0, n - 2 * quota), ...per.medium.slice(0, quota), ...per.long.slice(0, quota)];

mkdirSync(EVAL_DIR, { recursive: true });
const index: Record<string, string> = {};
for (const c of chosen) index[c.session] = c.path;
writeFileSync(join(EVAL_DIR, "index.json"), JSON.stringify(index, null, 2));
const out = resolve(import.meta.dir, "cases.jsonl");
writeFileSync(out, `${chosen.map(({ path: _, ...c }) => JSON.stringify(c)).join("\n")}\n`);
console.log(
	`${candidates.length} candidate prompts (short ${per.short.length}, medium ${per.medium.length}, long ${per.long.length}); wrote ${chosen.length} cases from ${Object.keys(index).length} sessions to ${out}`,
);
