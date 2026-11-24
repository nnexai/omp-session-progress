#!/usr/bin/env bun
// Picks the eval cases from local OMP sessions (read-only) and writes
// eval/cases.jsonl: pseudonymous ids, probe cut points and truth counts only.
// The pseudonym → session file index stays outside the repository in
// $OMP_PROGRESS_EVAL_DIR (default ~/.cache/omp-session-progress-eval).
//
// A case is one completed historic prompt with at least 5 tool calls, probed
// at about 15%, 40%, 65% and 90% of its tool calls. Strata (tags[0]) by tool
// calls: short 5–15, medium 16–50, long > 50; tags[1] is todo / no-todo.
// At most 2 prompts per session; order is a hash of the ids, so the pick is
// reproducible for the same session files.
//
//   bun eval/build-cases.ts [--n 100]

import { createHash } from "node:crypto";
import { mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { extractSession } from "./extract";
import { EVAL_DIR, SESSIONS_DIR } from "./paths";
import type { Probe } from "./grade";

const n = Number(process.argv[process.argv.indexOf("--n") + 1] || 100);
const EXCLUDE_CWD = ["omp-session-progress", "omp-progress-smoke", "omp-progress-bench", "omp-progress-eval"];
const FRACTIONS = [0.15, 0.4, 0.65, 0.9];

type Bucket = "short" | "medium" | "long";
const bucket = (tools: number): Bucket => (tools <= 15 ? "short" : tools <= 50 ? "medium" : "long");
const order = (key: string) => createHash("sha256").update(`omp-progress-eval-order:${key}`).digest("hex");

interface Candidate {
	id: string;
	session: string;
	prompt: number;
	tools: number;
	tags: string[];
	probes: Probe[];
	path: string;
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
			.sort((a, b) => order(`${trace.id}:${a.prompt}`).localeCompare(order(`${trace.id}:${b.prompt}`)))
			.slice(0, 2);
		for (const { p, prompt } of picked) {
			const total = p.toolStartIndices.length;
			const ks = [...new Set(FRACTIONS.map(f => Math.min(total - 1, Math.max(1, Math.round(f * total)))))];
			candidates.push({
				id: `${trace.id}-p${prompt}`,
				session: trace.id,
				prompt,
				tools: total,
				tags: [bucket(total), p.todoActive ? "todo" : "no-todo"],
				probes: ks.map(k => ({ cut: p.toolStartIndices[k] as number, done: k, remaining: total - k })),
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
