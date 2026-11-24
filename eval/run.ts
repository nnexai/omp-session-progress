#!/usr/bin/env bun
// Runs the progress-estimate eval: every case in eval/cases.jsonl × reps,
// through @widget in a disposable `omp --mode rpc` process (eval/host.ts).
// Resumable: (case, rep) pairs already in <out>/results.jsonl are skipped.
//
//   bun eval/run.ts --out <dir> [--reps 2] [--only id,id] [--concurrency 8]
//
// Transcripts are written for train cases only; the split comes from the
// ratchet flow state, read-only (--flow, default progress-estimate). Without a
// split (pilot), no transcript is written unless --pilot-traces is given.

import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import type { Case, Job } from "./host";

const argv = process.argv.slice(2);
const flag = (name: string, fallback?: string): string => {
	const i = argv.indexOf(`--${name}`);
	const value = i >= 0 ? argv[i + 1] : undefined;
	if (value === undefined && fallback === undefined) throw new Error(`--${name} is required`);
	return value ?? (fallback as string);
};
const project = resolve(import.meta.dir, "..");
const out = resolve(flag("out"));
const only = flag("only", "");
const cases: Case[] = readFileSync(join(import.meta.dir, "cases.jsonl"), "utf8")
	.split("\n")
	.filter(Boolean)
	.map(l => JSON.parse(l))
	.filter((c: Case) => !only || only.split(",").includes(c.id));

/** Train ids from the ratchet state; read-only. */
function trainIds(): string[] {
	if (argv.includes("--pilot-traces")) return cases.map(c => c.id);
	const state = join(project, ".omp", "ratchet", flag("flow", "progress-estimate"), "_state.json");
	return existsSync(state) ? (JSON.parse(readFileSync(state, "utf8")).train_ids ?? []) : [];
}

mkdirSync(out, { recursive: true });
const scratch = mkdtempSync(join(tmpdir(), "omp-progress-eval-"));
const overlay = join(scratch, "overlay.yml");
writeFileSync(overlay, "memory:\n  backend: off\nautolearn:\n  enabled: false\nadvisor:\n  enabled: false\n");
const jobPath = join(scratch, "job.json");
const job: Job = { cases, train: trainIds(), reps: Number(flag("reps", "2")), out, concurrency: Number(flag("concurrency", "8")) };
writeFileSync(jobPath, JSON.stringify(job));
mkdirSync(join(scratch, "sessions"));
mkdirSync(join(scratch, "work"));
const proc = Bun.spawn(
	[
		"omp", "--mode", "rpc", "--no-extensions",
		"-e", join(project, "eval", "host.ts"),
		"--session-dir", join(scratch, "sessions"),
		"--cwd", join(scratch, "work"),
		"--config", overlay,
		"--no-skills", "--no-rules", "--no-context-files", "--no-title", "--no-lsp",
	],
	{ cwd: join(scratch, "work"), stdin: "pipe", stdout: "pipe", stderr: Bun.file(join(scratch, "stderr.log")) },
);
proc.stdin.write(`${JSON.stringify({ type: "prompt", id: "eval", message: `/progress-eval ${jobPath}` })}\n`);
proc.stdin.flush();
const decoder = new TextDecoder();
let buffer = "";
let finished = false;
for await (const chunk of proc.stdout) {
	buffer += decoder.decode(chunk, { stream: true });
	for (let nl = buffer.indexOf("\n"); nl >= 0; nl = buffer.indexOf("\n")) {
		const line = buffer.slice(0, nl);
		buffer = buffer.slice(nl + 1);
		if (!line.includes("progress-eval:")) continue;
		const frame = JSON.parse(line);
		if (frame.type === "extension_ui_request" && frame.method === "notify") {
			console.log(frame.message);
			finished = frame.message.startsWith("progress-eval: done");
			proc.kill();
		}
	}
}
await proc.exited;
if (!finished) {
	console.error(`eval host ended without finishing; see ${join(scratch, "stderr.log")}`);
	process.exit(1);
}
rmSync(scratch, { recursive: true, force: true });
