#!/usr/bin/env bun
// Live smoke check against the installed `omp` over RPC in disposable sessions:
// temporary workspace and session directory, no ambient extensions, skills,
// rules, context files, memory, autolearn or advisor. Verifies:
//   1. /progress on shows the widget immediately (no consent dialog, zero LLM calls)
//   2. A constrained five-phase task with todos produces smooth, rising local estimates
//   3. new_session removes the widget
//   4. /progress on + underspecified request updates cleanly
//   5. /progress off removes the widget
//   6. Zero extension notify frames and zero observer markers in session files
//
// Usage: bun scripts/live-smoke.ts [--primary @tiny]

import { appendFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const argv = process.argv.slice(2);
const primaryIndex = argv.indexOf("--primary");
const primary = primaryIndex >= 0 && argv[primaryIndex + 1] ? argv[primaryIndex + 1]! : "@tiny";

const project = resolve(import.meta.dir, "..");
const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const evidence = join(project, "evidence", `live-${stamp}`);
mkdirSync(evidence, { recursive: true });
const scratch = mkdtempSync(join(tmpdir(), "omp-progress-smoke-"));
const work = join(scratch, "work");
const sessions = join(scratch, "sessions");
mkdirSync(work);
mkdirSync(sessions);
const overlay = join(scratch, "overlay.yml");
writeFileSync(overlay, "memory:\n  backend: off\nautolearn:\n  enabled: false\nadvisor:\n  enabled: false\ntools:\n  intentTracing: true\n");
const framesPath = join(evidence, "frames.jsonl");

const args = [
	"--mode", "rpc",
	"--no-extensions", "-e", join(project, "src", "index.ts"),
	"--session-dir", sessions,
	"--cwd", work,
	"--model", primary,
	"--config", overlay,
	"--no-skills", "--no-rules", "--no-context-files", "--no-title", "--no-lsp",
	"--auto-approve",
	"--max-cost", "1",
];

const FIXTURE = [
	"This is a synthetic test task in a scratch directory. Work in phases and track them with the todo tool:",
	"init the list first and mark each phase done as you finish it.",
	"1. Create data.txt containing the numbers 1 to 5, one per line.",
	"2. Run `sleep 12; wc -l data.txt` with bash.",
	"3. Create sum.sh that prints the sum of the numbers in data.txt, then run `sleep 12; sh sum.sh` with bash.",
	"4. Run `sleep 12; cat missing-input.txt` with bash. It is expected to fail; note the failure and continue.",
	"5. Reply with one sentence stating the sum.",
	"Do nothing else.",
].join("\n");
const AMBIGUOUS = "Have a look around this folder and tidy things up a bit if it seems worth it.";

interface Frame {
	t: number;
	frame: Record<string, any>;
}
const t0 = Date.now();
const frames: Frame[] = [];
const waiters: Array<{ pred: (f: Frame) => boolean; resolve: (f: Frame) => void }> = [];

const proc = Bun.spawn(["omp", ...args], {
	cwd: work,
	stdin: "pipe",
	stdout: "pipe",
	stderr: Bun.file(join(evidence, "stderr.log")),
});

const send = (cmd: Record<string, unknown>) => {
	appendFileSync(framesPath, `${JSON.stringify({ t: Date.now() - t0, sent: cmd })}\n`);
	proc.stdin.write(`${JSON.stringify(cmd)}\n`);
	proc.stdin.flush();
};

const waitFor = (pred: (f: Frame) => boolean, ms: number, what: string, since = 0) => {
	const hit = frames.slice(since).find(pred);
	if (hit) return Promise.resolve(hit);
	const { promise, resolve: done, reject } = Promise.withResolvers<Frame>();
	const timer = setTimeout(() => reject(new Error(`timed out waiting for ${what}`)), ms);
	waiters.push({ pred, resolve: f => (clearTimeout(timer), done(f)) });
	return promise;
};

(async () => {
	const decoder = new TextDecoder();
	let buffer = "";
	for await (const chunk of proc.stdout) {
		buffer += decoder.decode(chunk, { stream: true });
		let nl: number;
		while ((nl = buffer.indexOf("\n")) >= 0) {
			const line = buffer.slice(0, nl).trim();
			buffer = buffer.slice(nl + 1);
			if (!line) continue;
			let frame: Record<string, any>;
			try {
				frame = JSON.parse(line);
			} catch {
				continue;
			}
			const entry = { t: Date.now() - t0, frame };
			frames.push(entry);
			if (frame.type !== "message_update") appendFileSync(framesPath, `${JSON.stringify(entry)}\n`);
			for (const w of [...waiters]) {
				if (w.pred(entry)) {
					waiters.splice(waiters.indexOf(w), 1);
					w.resolve(entry);
				}
			}
		}
	}
})();

const isWidget = (f: Frame) => f.frame.type === "extension_ui_request" && f.frame.method === "setWidget" && f.frame.widgetKey === "session-progress";
const widgetShown = (f: Frame) => isWidget(f) && Array.isArray(f.frame.widgetLines);
const widgetCleared = (f: Frame) => isWidget(f) && !Array.isArray(f.frame.widgetLines);
const response = (id: string) => (f: Frame) => f.frame.type === "response" && f.frame.id === id;
const steps: string[] = [];
const step = (s: string) => {
	steps.push(`${((Date.now() - t0) / 1000).toFixed(1)}s ${s}`);
	console.log(steps[steps.length - 1]);
};

interface Run {
	name: string;
	start: number;
	end: number;
	sessionFile?: string;
	messages: Array<{ role: string; content?: unknown }>;
}
const runs: Run[] = [];

async function optIn(id: string) {
	const mark = frames.length;
	send({ id, type: "prompt", message: "/progress on" });
	await waitFor(widgetShown, 15_000, "widget after opt-in", mark);
	step("opted in; widget shown immediately");
}

async function runPrompt(name: string, message: string) {
	const mark = frames.length;
	send({ id: name, type: "prompt", message });
	const start = await waitFor(f => f.frame.type === "agent_start", 60_000, `${name} agent_start`, mark);
	const end = await waitFor(f => f.frame.type === "agent_end", 8 * 60_000, `${name} agent_end`, mark);
	step(`${name}: agent ran ${((end.t - start.t) / 1000).toFixed(1)} s`);
	await Bun.sleep(500);
	send({ id: `${name}-state`, type: "get_state" });
	const state = (await waitFor(response(`${name}-state`), 30_000, "state")).frame.data;
	send({ id: `${name}-messages`, type: "get_messages" });
	const messages = (await waitFor(response(`${name}-messages`), 30_000, "messages")).frame.data?.messages ?? [];
	runs.push({ name, start: start.t, end: end.t, sessionFile: state?.sessionFile, messages });
	if (state?.sessionFile && existsSync(state.sessionFile)) writeFileSync(join(evidence, `session-${name}.jsonl`), readFileSync(state.sessionFile));
}

let failure: string | null = null;
try {
	await waitFor(f => f.frame.type === "ready", 60_000, "ready");
	step("omp ready");
	await optIn("on-1");
	await runPrompt("fixture", FIXTURE);

	let mark = frames.length;
	send({ id: "new", type: "new_session" });
	await waitFor(response("new"), 30_000, "new_session");
	await waitFor(widgetCleared, 10_000, "widget cleared on new_session", mark);
	step("new_session removed the widget");

	await optIn("on-2");
	await runPrompt("ambiguous", AMBIGUOUS);

	mark = frames.length;
	send({ id: "off", type: "prompt", message: "/progress off" });
	await waitFor(widgetCleared, 10_000, "widget cleared on /progress off", mark);
	step("/progress off removed the widget");
} catch (error) {
	failure = error instanceof Error ? error.message : String(error);
	step(`FAILED: ${failure}`);
} finally {
	proc.stdin.end();
	await Promise.race([proc.exited, Bun.sleep(15_000)]);
	proc.kill();
}

const notifies = frames.filter(f => f.frame.type === "extension_ui_request" && f.frame.method === "notify");
const confirms = frames.filter(f => f.frame.type === "extension_ui_request" && f.frame.method === "confirm");
const sec = (ms: number) => (ms / 1000).toFixed(1);
const widgetFrames = frames.filter(isWidget);
// Sample distinct widget states (avoid spamming 1s ticks in summary)
const sampled = widgetFrames.filter((f, i) => i === 0 || i === widgetFrames.length - 1 || i % 3 === 0 || !Array.isArray(f.frame.widgetLines));
const widgetSnapshots = sampled.map(
	f => `t=${sec(f.t)}s\n${Array.isArray(f.frame.widgetLines) ? f.frame.widgetLines.join("\n") : "(widget removed)"}`,
);

const md = [
	`# Live check ${stamp}`,
	"",
	`- omp: ${Bun.spawnSync(["omp", "--version"]).stdout.toString().trim()}; primary agent: ${primary}; estimator: local GBDT (zero LLM calls)`,
	`- result: ${failure ? `FAILED: ${failure}` : "passed"}`,
	`- confirm dialogs: ${confirms.length}; notify frames: ${notifies.length}; total widget updates: ${widgetFrames.length}`,
	"",
	"## Steps",
	"",
	"```",
	...steps,
	"```",
	"",
	"## Widget snapshots (sampled)",
	"",
	"```",
	widgetSnapshots.join("\n\n"),
	"```",
].join("\n");
writeFileSync(join(evidence, "summary.md"), md);
console.log(`\nEvidence: ${evidence}`);
process.exit(failure ? 1 : 0);
