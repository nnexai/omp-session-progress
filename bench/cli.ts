#!/usr/bin/env bun
// Replay benchmark of the progress estimator on historic OMP sessions.
//
//   bun bench/cli.ts select [--per 6]                 pick sessions, split dev/held-out, extract traces
//   bun bench/cli.ts run --variant v1 --split dev --run <name>
//                                                     replay (local variants in-process; model variants
//                                                     through @widget in a disposable omp RPC process)
//   bun bench/cli.ts score --run <name> --split dev --variants baseline,v1 [--out report.md]
//   bun bench/cli.ts review --variant v1 --run <name> [--n 4]   print sample sent payloads for manual review
//   bun bench/cli.ts diag --variant v2 --run <name> --split dev  stage/next confusion, premature claims
//   bun bench/cli.ts check --run <name> [--tuned cadence,evidence]
//                                                     split disjoint, tuning runs dev-only, prefix-only calls,
//                                                     every trace event inside the sanitized vocabulary
//
// Session files are read-only. Traces, runs and the pseudonym index live in
// $OMP_PROGRESS_BENCH_DIR (default ~/.cache/omp-session-progress-bench); a
// score report holds aggregates and pseudonymous ids only.

import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { applyEvent, buildPayload, emptyFacts } from "../src/facts";
import { DATA_DIR, ensureDir, loadManifest, loadTrace, type ManifestEntry, select, type Split } from "./data";
import type { Job } from "./extension";
import { type ReplayResult, replay } from "./replay";
import { unexpectedWords } from "./sanitize";
import { checkpoints, cost, metrics } from "./score";
import { variant } from "./variants";

const argv = process.argv.slice(2);
const flag = (name: string, fallback?: string) => {
	const i = argv.indexOf(`--${name}`);
	const value = i >= 0 ? argv[i + 1] : undefined;
	if (value === undefined && fallback === undefined) throw new Error(`--${name} is required`);
	return value ?? (fallback as string);
};
const project = resolve(import.meta.dir, "..");
const runDir = (run: string, v: string) => ensureDir(join(DATA_DIR, "runs", run, v));
const splitEntries = (split: string): ManifestEntry[] => loadManifest().entries.filter(e => split === "all" || e.split === (split as Split));

async function runModel(v: string, ids: string[], out: string): Promise<void> {
	const scratch = mkdtempSync(join(tmpdir(), "omp-progress-bench-"));
	const overlay = join(scratch, "overlay.yml");
	writeFileSync(overlay, "memory:\n  backend: off\nautolearn:\n  enabled: false\nadvisor:\n  enabled: false\n");
	const jobPath = join(scratch, "job.json");
	const job: Job = { variant: v, traces: ids, out, concurrency: Number(flag("concurrency", "6")) };
	writeFileSync(jobPath, JSON.stringify(job));
	rmSync(join(out, "done.json"), { force: true });
	// The production extension is loaded too, so the run also proves it still loads.
	const proc = Bun.spawn(
		[
			"omp", "--mode", "rpc", "--no-extensions",
			"-e", join(project, "src", "index.ts"),
			"-e", join(project, "bench", "extension.ts"),
			"--session-dir", ensureDir(join(scratch, "sessions")),
			"--cwd", ensureDir(join(scratch, "work")),
			"--config", overlay,
			"--no-skills", "--no-rules", "--no-context-files", "--no-title", "--no-lsp",
		],
		{ cwd: join(scratch, "work"), stdin: "pipe", stdout: "pipe", stderr: Bun.file(join(scratch, "stderr.log")) },
	);
	proc.stdin.write(`${JSON.stringify({ type: "prompt", id: "bench", message: `/progress-bench ${jobPath}` })}\n`);
	proc.stdin.flush();
	const decoder = new TextDecoder();
	let buffer = "";
	for await (const chunk of proc.stdout) {
		buffer += decoder.decode(chunk, { stream: true });
		let nl: number;
		while ((nl = buffer.indexOf("\n")) >= 0) {
			const line = buffer.slice(0, nl);
			buffer = buffer.slice(nl + 1);
			if (!line.includes("progress-bench:")) continue;
			const frame = JSON.parse(line);
			if (frame.type === "extension_ui_request" && frame.method === "notify") {
				console.log(frame.message);
				proc.kill();
			}
		}
	}
	await proc.exited;
	if (!existsSync(join(out, "done.json"))) throw new Error(`bench run ended without results; see ${join(scratch, "stderr.log")}`);
	const done = JSON.parse(readFileSync(join(out, "done.json"), "utf8"));
	if (done.errors.length) console.error(done.errors.join("\n"));
	rmSync(scratch, { recursive: true, force: true });
}

const loadResults = (run: string, v: string): Map<string, ReplayResult> => {
	const dir = join(DATA_DIR, "runs", run, v);
	const out = new Map<string, ReplayResult>();
	if (!existsSync(dir)) return out;
	for (const f of readdirSync(dir)) if (f.startsWith("s-")) out.set(f.replace(/\.json$/, ""), JSON.parse(readFileSync(join(dir, f), "utf8")));
	return out;
};

const pct = (x: number) => (Number.isFinite(x) ? `${(100 * x).toFixed(1)}%` : "–");
const num = (x: number, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : "–");

function scoreReport(run: string, split: string, names: string[]): string {
	const entries = splitEntries(split);
	const traces = new Map(entries.map(e => [e.id, loadTrace(e.id)]));
	const rows: string[] = [];
	const costRows: string[] = [];
	const growthRows: string[] = [];
	const strataRows: string[] = [];
	for (const name of names) {
		const results = loadResults(run, name);
		const missing = entries.filter(e => !results.has(e.id)).length;
		const pairs = entries.flatMap(e => {
			const trace = traces.get(e.id);
			const result = results.get(e.id);
			return trace && result ? [{ trace, result }] : [];
		});
		const m = metrics(pairs);
		const c = cost(pairs.map(p => p.result));
		rows.push(
			`| ${name} | ${pairs.length}${missing ? ` (${missing} missing)` : ""} | ${m.checkpoints} | ${pct(m.coverage)} | ${pct(m.exact)} | ${pct(m.within1)} | ${pct(m.within1Tools)} | ${num(m.mae)} | ${pct(m.openAbstain)} | ${pct(m.abstentionPrecision)} | ${pct(m.prematureRate)} (${m.prematureCount}) | ${pct(m.doneRecall)} | ${pct(m.nextCoverage)} / ${pct(m.nextAccuracy)} | ${pct(m.displayCoverage)} / ${pct(m.displayWithin1)} |`,
		);
		costRows.push(
			`| ${name} | ${c.modelCalls} | ${c.failures} | ${c.skippedUnchanged} | ${c.cappedSessions} | ${c.inputTokens} | ${c.cacheRead} | ${c.cacheWrite} | ${c.outputTokens} | ${num(c.modelCalls ? c.inputTokens / c.modelCalls : Number.NaN, 0)} | ${num(c.modelCalls ? c.outputTokens / c.modelCalls : Number.NaN, 0)} | $${num(c.costUsd, 4)} | ${num(c.meanPayloadChars, 0)} / ${c.maxPayloadChars} | ${c.p50Ms} / ${c.p90Ms} |`,
		);
		// Context growth: payload and prompt tokens by position of the call in its session.
		const buckets = [
			[1, 5],
			[6, 15],
			[16, 30],
			[31, Number.MAX_SAFE_INTEGER],
		];
		const cells = buckets.map(([lo, hi]) => {
			const calls = pairs.flatMap(p => p.result.calls.filter((_, i) => i + 1 >= (lo as number) && i + 1 <= (hi as number)));
			const model = calls.filter(x => x.model && x.inputTokens);
			const chars = calls.reduce((a, x) => a + x.payloadChars, 0) / (calls.length || 1);
			const tokens = model.reduce((a, x) => a + (x.inputTokens ?? 0), 0) / (model.length || 1);
			return calls.length ? `${num(chars, 0)} ch / ${model.length ? num(tokens, 0) : "–"} tok (n=${calls.length})` : "–";
		});
		growthRows.push(`| ${name} | ${cells.join(" | ")} |`);
		const strata = [...new Set(entries.map(e => e.stratum))].sort();
		strataRows.push(
			`| ${name} | ${strata
				.map(s => {
					const sm = metrics(pairs.filter(p => entries.find(e => e.id === p.trace.id)?.stratum === s));
					return `${pct(sm.within1)} / ${pct(sm.coverage)} / ${pct(sm.openAbstain)}`;
				})
				.join(" | ")} |`,
		);
	}
	const strata = [...new Set(entries.map(e => e.stratum))].sort();
	return [
		`### ${split} (${entries.length} sessions, ${entries.reduce((a, e) => a + e.prompts, 0)} prompts)`,
		"",
		"| variant | sessions | checkpoints | stage coverage (structured, running) | exact | within 1 | within 1 (tool-call truth) | MAE (ranks) | abstain on open-ended | abstention precision | premature completion | done after completed | next: given / correct | shown over time: coverage / within 1 |",
		"|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
		...rows,
		"",
		"| variant | model calls | failures | skipped unchanged | capped sessions | prompt tokens | of which cache read | cache write | output tokens | prompt tok/call | output tok/call | cost | payload chars mean / max | latency ms p50 / p90 |",
		"|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
		...costRows,
		"",
		"Context growth (mean payload chars / prompt tokens per call, by call number within a session):",
		"",
		"| variant | calls 1–5 | 6–15 | 16–30 | 31+ |",
		"|---|---|---|---|---|",
		...growthRows,
		"",
		"By stratum (within 1 / coverage / abstain on open-ended):",
		"",
		`| variant | ${strata.join(" | ")} |`,
		`|---|${strata.map(() => "---").join("|")}|`,
		...strataRows,
		"",
	].join("\n");
}

const command = argv[0];
if (command === "select") {
	const manifest = select(Number(flag("per", "6")), {
		modifiedWithinMs: 6 * 3_600_000,
		cwdIncludes: ["omp-session-progress", "omp-progress-smoke", "omp-progress-bench"],
	});
	const by = (split: Split) => manifest.entries.filter(e => e.split === split);
	console.log(`scanned ${manifest.scanned} top-level sessions, ${manifest.eligible} eligible; dev ${by("dev").length}, held-out ${by("heldout").length}`);
	for (const e of manifest.entries) console.log(`${e.split}\t${e.stratum}\t${e.id}\t${e.project}\tprompts ${e.prompts} (structured ${e.structured})\ttools ${e.toolCalls}\t${e.hours.toFixed(1)} h\t${JSON.stringify(e.outcomes)}`);
} else if (command === "run") {
	const name = flag("variant");
	const v = variant(name);
	const ids = splitEntries(flag("split")).map(e => e.id);
	const out = runDir(flag("run"), name);
	if (v.local) {
		for (const id of ids) {
			const trace = loadTrace(id);
			const result = await replay(id, trace.events, v, () => Promise.reject(new Error("local variant called the model")));
			writeFileSync(join(out, `${id}.json`), JSON.stringify(result));
		}
		console.log(`${name}: ${ids.length} traces replayed locally`);
	} else {
		await runModel(name, ids, out);
	}
} else if (command === "score") {
	const report = scoreReport(flag("run"), flag("split"), flag("variants").split(","));
	console.log(report);
	const out = flag("out", "");
	if (out) writeFileSync(out, report);
} else if (command === "review") {
	// Re-derive a few sent payloads from the traces for manual review.
	const results = loadResults(flag("run"), flag("variant"));
	const n = Number(flag("n", "4"));
	for (const [id, result] of [...results].slice(0, n)) {
		const call = result.calls[Math.floor(result.calls.length / 2)];
		if (!call) continue;
		const facts = emptyFacts();
		for (const { event } of loadTrace(id).events.slice(0, call.prefixEvents)) applyEvent(facts, event);
		console.log(`--- ${id} call at prefix ${call.prefixEvents}\n${buildPayload(facts)}\n`);
	}
} else if (command === "diag") {
	// Aggregate diagnostics of one variant: stage confusion, next-step confusion, premature claims.
	const name = flag("variant");
	const results = loadResults(flag("run"), name);
	const entries = splitEntries(flag("split"));
	const cps = entries.flatMap(e => {
		const r = results.get(e.id);
		return r ? checkpoints(loadTrace(e.id), r) : [];
	});
	const sr = cps.filter(c => c.structured && c.running && c.truth !== null);
	const labels = ["0 just started", "1 early", "2 middle", "3 late", "4 verify/wrap", "5 appears done", "abstain"];
	const grid = labels.map(() => labels.slice(0, 5).map(() => 0));
	for (const c of sr) (grid[c.predicted ?? 6] as number[])[c.truth as number] += 1;
	console.log(`stage: rows predicted, columns truth rank 0..4 (${sr.length} structured running checkpoints)`);
	labels.forEach((l, i) => console.log(`${l.padEnd(16)} ${(grid[i] as number[]).map(n => String(n).padStart(5)).join("")}`));
	const given = sr.filter(c => c.predicted !== null);
	console.log(`signed mean error (predicted − truth): ${num(given.reduce((a, c) => a + (c.predicted as number) - (c.truth as number), 0) / given.length)}`);
	const acts = [...new Set(cps.flatMap(c => [c.nextGuess, c.nextActual ?? "none"]))].sort();
	console.log(`\nnext: rows guess, columns actual (${acts.join(", ")})`);
	for (const g of acts) {
		const row = acts.map(a => cps.filter(c => c.running && c.nextGuess === g && (c.nextActual ?? "none") === a).length);
		if (row.some(Boolean)) console.log(`${g.padEnd(13)} ${row.map(n => String(n).padStart(5)).join("")}`);
	}
	console.log("\npremature claims (stage, progress, running, outcome):");
	for (const c of cps.filter(x => x.premature)) console.log(`  ${c.predicted} ${c.progress === null ? "–" : c.progress.toFixed(2)} ${c.running} ${c.outcome} structured=${c.structured}`);
	const open = cps.filter(c => !c.structured && c.running && c.predicted !== null);
	console.log(`\nstage given on open-ended running prompts: ${open.length} (ranks ${open.map(c => c.predicted).join(",")})`);
} else if (command === "check") {
	// Invariants over the stored data: disjoint split, tuning only on dev, prefix-only calls, sanitized traces.
	const manifest = loadManifest();
	const split = new Map(manifest.entries.map(e => [e.id, e.split]));
	const problems: string[] = [];
	if (split.size !== manifest.entries.length) problems.push("duplicate trace ids in the manifest");
	const run = flag("run");
	const tuned = flag("tuned", "cadence,evidence").split(",");
	let calls = 0;
	for (const v of existsSync(join(DATA_DIR, "runs", run)) ? readdirSync(join(DATA_DIR, "runs", run)) : []) {
		for (const [id, result] of loadResults(run, v)) {
			if (!split.has(id)) problems.push(`${v}/${id}: not in the manifest`);
			if (tuned.includes(v) && split.get(id) !== "dev") problems.push(`${v}/${id}: tuning candidate run on ${split.get(id)}`);
			for (const c of result.calls) {
				calls += 1;
				if (c.prefixEnd > c.at) problems.push(`${v}/${id}: call at ${c.at} saw an event at ${c.prefixEnd}`);
			}
		}
	}
	let fields = 0;
	for (const e of manifest.entries) {
		for (const { event } of loadTrace(e.id).events) {
			// The fields that carry historic text in a real session; ids, op names and roles are fixed by the extractor.
			const message = event.message && typeof event.message === "object" ? event.message : {};
			const args = event.args && typeof event.args === "object" ? event.args : {};
			const texts = [event.prompt, event.intent, event.toolName, "content" in message ? message.content : null, "errorMessage" in message ? message.errorMessage : null, "items" in args ? args.items : null, "task" in args ? args.task : null, "list" in args ? args.list : null];
			fields += 1;
			const bad = unexpectedWords(JSON.stringify(texts).replace(/"type":"text",/g, ""));
			if (bad.length) problems.push(`${e.id}: event field outside the sanitized vocabulary (${bad.join(" ").length} chars)`);
		}
	}
	console.log(`${manifest.entries.length} traces (${manifest.entries.filter(e => e.split === "dev").length} dev / ${manifest.entries.filter(e => e.split === "heldout").length} held-out), ${fields} events, ${calls} stored calls checked`);
	console.log(problems.length ? problems.slice(0, 20).join("\n") : "ok: split disjoint, tuning runs dev-only, every call prefix-only, every trace event sanitized");
	if (problems.length) process.exit(1);
} else {
	console.log("usage: bun bench/cli.ts select | run | score | review | diag | check (see header)");
}
