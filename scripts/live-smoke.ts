#!/usr/bin/env bun
// Live check against the installed `omp`, in disposable sessions: a temporary
// workspace and session directory, no ambient extensions, skills, rules,
// context files, memory, autolearn or advisor. It drives `omp --mode rpc`,
// whose extension UI carries the widget as plain lines, so every widget update
// is recorded with its time. Two prompts:
//   1. a constrained five-phase task with a todo list (the share done should
//      rise and the calls expected to remain should fall during the run), and
//   2. after /new and a fresh opt-in, an underspecified request (the observer
//      still guesses; nothing is withheld once a request is seen).
// Every observer call (what was sent to @widget and what it answered) is logged.
//
// Usage: bun scripts/live-smoke.ts [--primary @tiny]
// Evidence: evidence/live-<timestamp>/{summary.md,observer-calls.jsonl,frames.jsonl,...}

import { appendFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const argv = process.argv.slice(2);
const primaryIndex = argv.indexOf("--primary");
// The primary agent runs on a different role than the observer, so the two never share a model.
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
// Isolation from personal state, and intent tracing on as in the interactive TUI (RPC turns it off by default).
writeFileSync(overlay, "memory:\n  backend: off\nautolearn:\n  enabled: false\nadvisor:\n  enabled: false\ntools:\n  intentTracing: true\n");
const framesPath = join(evidence, "frames.jsonl");
const callsPath = join(evidence, "observer-calls.jsonl");

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
	env: { ...process.env, OMP_SESSION_PROGRESS_LOG: callsPath },
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
			// The consent dialog of /progress on: this check opts in explicitly.
			if (frame.type === "extension_ui_request" && frame.method === "confirm") {
				send({ type: "extension_ui_response", id: frame.id, confirmed: true });
			}
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
const readCalls = () =>
	existsSync(callsPath) ? readFileSync(callsPath, "utf8").trim().split("\n").filter(Boolean).map(l => JSON.parse(l)) : [];

interface Run {
	name: string;
	optIn: number;
	start: number;
	end: number;
	sessionFile?: string;
	messages: Array<{ role: string; content?: unknown }>;
}
const runs: Run[] = [];

async function optIn(id: string) {
	const mark = frames.length;
	const at = Date.now() - t0;
	send({ id, type: "prompt", message: "/progress on" });
	await waitFor(widgetShown, 30_000, "widget after opt-in", mark);
	step("opted in; widget shown");
	return at;
}

async function runPrompt(name: string, message: string, optInAt: number) {
	const mark = frames.length;
	send({ id: name, type: "prompt", message });
	const start = await waitFor(f => f.frame.type === "agent_start", 60_000, `${name} agent_start`, mark);
	const end = await waitFor(f => f.frame.type === "agent_end", 8 * 60_000, `${name} agent_end`, mark);
	step(`${name}: agent ran ${((end.t - start.t) / 1000).toFixed(1)} s`);
	// The observer refreshes once more after agent_end (subject to its 15 s spacing); wait for that reply.
	for (let i = 0; i < 80 && !readCalls().some(c => Date.parse(c.at) - t0 > end.t); i++) await Bun.sleep(500);
	await Bun.sleep(300);
	send({ id: `${name}-state`, type: "get_state" });
	const state = (await waitFor(response(`${name}-state`), 30_000, "state")).frame.data;
	send({ id: `${name}-messages`, type: "get_messages" });
	const messages = (await waitFor(response(`${name}-messages`), 30_000, "messages")).frame.data?.messages ?? [];
	runs.push({ name, optIn: optInAt, start: start.t, end: end.t, sessionFile: state?.sessionFile, messages });
	if (state?.sessionFile && existsSync(state.sessionFile)) writeFileSync(join(evidence, `session-${name}.jsonl`), readFileSync(state.sessionFile));
}

let failure: string | null = null;
try {
	await waitFor(f => f.frame.type === "ready", 60_000, "ready");
	step("omp ready");
	await runPrompt("fixture", FIXTURE, await optIn("on-1"));

	// Session switch while on: the widget must be removed and the new session starts off.
	let mark = frames.length;
	send({ id: "new", type: "new_session" });
	await waitFor(response("new"), 30_000, "new_session");
	await waitFor(widgetCleared, 10_000, "widget cleared on new_session", mark);
	step("new_session removed the widget");

	await runPrompt("ambiguous", AMBIGUOUS, await optIn("on-2"));

	mark = frames.length;
	send({ id: "off", type: "prompt", message: "/progress off" });
	await waitFor(widgetCleared, 10_000, "widget cleared on /progress off", mark);
	step("/progress off removed the widget");
} catch (error) {
	failure = error instanceof Error ? error.message : String(error);
	step(`FAILED: ${failure}`);
} finally {
	proc.stdin.end();
	await Promise.race([proc.exited, Bun.sleep(30_000)]);
	proc.kill();
}

// ---- Summary ----------------------------------------------------------------
const calls = readCalls();
const notifies = frames.filter(f => f.frame.type === "extension_ui_request" && f.frame.method === "notify");
const sec = (ms: number) => (ms / 1000).toFixed(1);
const assistantText = (m: { role: string; content?: unknown }) =>
	m.role === "assistant" && Array.isArray(m.content)
		? m.content.filter((b: any) => b?.type === "text").map((b: any) => b.text).join(" ").trim()
		: "";

function callRows(run: Run) {
	// Each opt-in starts a fresh observer: its calls are the ones made before the next opt-in.
	const until = runs[runs.indexOf(run) + 1]?.optIn ?? Number.POSITIVE_INFINITY;
	const mine = calls.filter(c => {
		const at = Date.parse(c.at) - t0;
		return at >= run.optIn && at < until;
	});
	return mine.map((c, i) => {
		const sent = JSON.parse(c.sent);
		let reply: any = {};
		try {
			reply = JSON.parse(c.reply.slice(c.reply.indexOf("{"), c.reply.lastIndexOf("}") + 1));
		} catch {}
		const todo = sent.todo_list ? `${sent.todo_list.done}/${sent.todo_list.total}` : "none";
		const guess = reply.progress_pct === undefined ? "**unparsable**" : `${reply.progress_pct}% · ${reply.remaining_tool_calls} left`;
		return `| ${i + 1} | ${sec(Date.parse(c.at) - t0 - run.start)} | ${sent.current_prompt.turn} | ${todo} | ${sent.current_prompt.tool_calls.finished} (${sent.current_prompt.tool_calls.failed} failed) | ${guess} | ${reply.basis ?? ""} | ${reply.next ?? c.error ?? ""} |`;
	});
}

const widgetSnapshots = frames
	.filter(isWidget)
	.map(f => `t=${sec(f.t)}s\n${Array.isArray(f.frame.widgetLines) ? f.frame.widgetLines.join("\n") : "(widget removed)"}`);

const md = [
	`# Live check ${stamp}`,
	"",
	`- omp: ${Bun.spawnSync(["omp", "--version"]).stdout.toString().trim()}; primary agent: ${primary}; observer: @widget`,
	`- result: ${failure ? `FAILED: ${failure}` : "passed"}`,
	`- observer models used: ${[...new Set(calls.map(c => c.model))].join(", ") || "none"}; ${calls.length} calls; ${calls.reduce((a, c) => a + (c.usage?.input ?? 0) + (c.usage?.cacheRead ?? 0) + (c.usage?.cacheWrite ?? 0), 0)} input (incl. cached) / ${calls.reduce((a, c) => a + (c.usage?.output ?? 0), 0)} output tokens`,
	`- extension notify frames (chat/toast output): ${notifies.length}`,
	`- scratch dir (disposable): ${scratch}`,
	"",
	"## Steps",
	"",
	"```",
	...steps,
	"```",
	"",
	...runs.flatMap(run => {
		const sessionText = existsSync(join(evidence, `session-${run.name}.jsonl`)) ? readFileSync(join(evidence, `session-${run.name}.jsonl`), "utf8") : "";
		const markers = ["observed session data", "how far along", "session-progress", '"progress_pct"'].filter(m => sessionText.includes(m));
		const last = [...run.messages].reverse().map(assistantText).find(Boolean) ?? "";
		return [
			`## Prompt "${run.name}": agent ran ${sec(run.end - run.start)} s`,
			"",
			"Observer readings (t = seconds from agent_start; observed columns are what was sent):",
			"",
			"| # | t | turn | todos done | tool calls | model guess (before calibration) | basis | likely next |",
			"|---|---|---|---|---|---|---|---|",
			...callRows(run),
			"",
			`What the agent finally did: ${last ? `"${last.slice(0, 300)}"` : "(no reply text)"}`,
			"",
			`Transcript: ${run.messages.length} messages, ${run.messages.filter(m => m.role === "user").length} user message(s); observer markers in the session file: ${markers.length ? markers.join(", ") : "none"}`,
			"",
		];
	}),
	"## Widget updates (as sent to the RPC host)",
	"",
	"```",
	widgetSnapshots.join("\n\n"),
	"```",
].join("\n");
writeFileSync(join(evidence, "summary.md"), md);
console.log(`\nEvidence: ${evidence}`);
process.exit(failure ? 1 : 0);
