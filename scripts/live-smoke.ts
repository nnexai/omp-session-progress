#!/usr/bin/env bun
// Live smoke test against the installed `omp`, in a disposable session:
// a temporary workspace and session directory, no ambient extensions, skills,
// rules, context files, memory or autolearn. It loads this extension with -e,
// opts in with /progress on, runs a synthetic multi-phase task on the primary
// model (@smol by default), and records every RPC frame plus every observer
// call (what was sent to the observer model and what it answered).
//
// Usage: bun scripts/live-smoke.ts [--primary @smol] [--role tiny|widget]
// Evidence: evidence/live-<timestamp>/{frames.jsonl,observer-calls.jsonl,summary.md}

import { mkdirSync, mkdtempSync, readFileSync, writeFileSync, appendFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const argv = process.argv.slice(2);
const opt = (name: string, fallback: string) => {
	const i = argv.indexOf(`--${name}`);
	return i >= 0 && argv[i + 1] ? argv[i + 1]! : fallback;
};
const primary = opt("primary", "@smol");
const role = opt("role", "tiny");

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

const TASK = [
	"This is a synthetic test task in a scratch directory. Work in phases and track them with the todo tool:",
	"init the list first and mark each phase done as you finish it.",
	"1. Create data.txt containing the numbers 1 to 5, one per line.",
	"2. Run `sleep 12; wc -l data.txt` with bash.",
	"3. Create sum.sh that prints the sum of the numbers in data.txt, then run `sleep 12; sh sum.sh` with bash.",
	"4. Run `sleep 12; cat missing-input.txt` with bash. It is expected to fail; note the failure and continue.",
	"5. Reply with one sentence stating the sum.",
	"Do nothing else.",
].join("\n");

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
			// Consent dialog of /progress on: the smoke test opts in explicitly.
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

const ui = (method: string, text?: string) => (f: Frame) =>
	f.frame.type === "extension_ui_request" && f.frame.method === method && (text === undefined || String(f.frame.message ?? f.frame.statusText ?? "").includes(text));
const response = (id: string) => (f: Frame) => f.frame.type === "response" && f.frame.id === id;
const steps: string[] = [];
const step = (s: string) => {
	steps.push(`${((Date.now() - t0) / 1000).toFixed(1)}s ${s}`);
	console.log(steps[steps.length - 1]);
};

let failure: string | null = null;
try {
	await waitFor(f => f.frame.type === "ready", 60_000, "ready");
	step("omp ready");

	send({ id: "roles", type: "prompt", message: "/progress roles" });
	const roles = await waitFor(ui("notify", "@tiny"), 30_000, "roles notify");
	step(`roles:\n${roles.frame.message}`);

	let mark = frames.length;
	send({ id: "on", type: "prompt", message: `/progress on ${role}` });
	const on = await waitFor(ui("notify", "observer on"), 30_000, "opt-in notify", mark);
	step(on.frame.message);

	mark = frames.length;
	send({ id: "task", type: "prompt", message: TASK });
	await waitFor(f => f.frame.type === "agent_start", 60_000, "agent_start", mark);
	step("agent started");
	const end = await waitFor(f => f.frame.type === "agent_end", 8 * 60_000, "agent_end", mark);
	step("agent ended");
	// The observer refreshes once more after agent_end (subject to its 15 s spacing); wait for that reply.
	const endAt = t0 + end.t;
	const finalCall = async () => {
		for (let i = 0; i < 80; i++) {
			const lines = existsSync(callsPath) ? readFileSync(callsPath, "utf8").trim().split("\n").filter(Boolean) : [];
			if (lines.some(l => Date.parse(JSON.parse(l).at) > endAt)) return true;
			await Bun.sleep(500);
		}
		return false;
	};
	step((await finalCall()) ? "final post-run observer call completed" : "no post-run observer call within 40 s");
	await Bun.sleep(300); // let its setStatus frame arrive

	mark = frames.length;
	send({ id: "show", type: "prompt", message: "/progress" });
	const report = await waitFor(ui("notify", "Session progress ("), 30_000, "report", mark);
	step(`report:\n${report.frame.message}`);

	send({ id: "state", type: "get_state" });
	const state = (await waitFor(response("state"), 30_000, "state")).frame.data;
	send({ id: "messages", type: "get_messages" });
	const messages = (await waitFor(response("messages"), 30_000, "messages")).frame.data?.messages ?? [];
	writeFileSync(join(evidence, "messages.json"), JSON.stringify(messages, null, 2));
	step(`session ${state?.sessionId} file ${state?.sessionFile}; ${messages.length} messages`);

	// Session switch while on: the status must be cleared and the new session must start off.
	mark = frames.length;
	send({ id: "on2", type: "prompt", message: `/progress on ${role}` });
	await waitFor(ui("notify", "observer on"), 30_000, "second opt-in", mark);
	mark = frames.length;
	send({ id: "new", type: "new_session" });
	await waitFor(response("new"), 30_000, "new_session");
	await waitFor(f => ui("setStatus")(f) && (f.frame.statusText === undefined || f.frame.statusText === null), 10_000, "status cleared", mark);
	step("new_session cleared the status");
	mark = frames.length;
	send({ id: "show2", type: "prompt", message: "/progress" });
	const after = await waitFor(ui("notify", "observer is off"), 30_000, "off after switch", mark);
	step(`after switch: ${after.frame.message}`);

	writeFileSync(join(evidence, "state.json"), JSON.stringify(state, null, 2));
	if (state?.sessionFile && existsSync(state.sessionFile)) {
		writeFileSync(join(evidence, "session-file-copy.jsonl"), readFileSync(state.sessionFile));
	}
} catch (error) {
	failure = error instanceof Error ? error.message : String(error);
	step(`FAILED: ${failure}`);
} finally {
	proc.stdin.end();
	await Promise.race([proc.exited, Bun.sleep(30_000)]);
	proc.kill();
}

// Summary: timeline of tool events and status updates, observer calls, transcript check.
const calls = existsSync(callsPath)
	? readFileSync(callsPath, "utf8").trim().split("\n").filter(Boolean).map(l => JSON.parse(l))
	: [];
const agentStart = frames.find(f => f.frame.type === "agent_start")?.t ?? 0;
const agentEnd = frames.find(f => f.frame.type === "agent_end")?.t ?? 0;
const timeline = frames
	.filter(
		f =>
			["tool_execution_start", "tool_execution_end", "agent_start", "agent_end"].includes(f.frame.type) ||
			(f.frame.type === "extension_ui_request" && f.frame.method === "setStatus"),
	)
	.map(f => {
		const s = ((f.t - agentStart) / 1000).toFixed(1).padStart(6);
		const x = f.frame;
		if (x.type === "tool_execution_start") return `| ${s} | tool start | ${x.toolName}: ${x.intent ?? ""} |`;
		if (x.type === "tool_execution_end") return `| ${s} | tool end | ${x.toolName}${x.isError ? " (error)" : ""} |`;
		if (x.type === "extension_ui_request") return `| ${s} | **status** | ${x.statusText ?? "(cleared)"} |`;
		return `| ${s} | ${x.type} | |`;
	});
const sessionCopy = join(evidence, "session-file-copy.jsonl");
const sessionText = existsSync(sessionCopy) ? readFileSync(sessionCopy, "utf8") : "";
const leaks = ["observed session data", "You watch another coding agent", "session-progress", "\"summary\":"].filter(m => sessionText.includes(m));
const messagesJson = existsSync(join(evidence, "messages.json")) ? JSON.parse(readFileSync(join(evidence, "messages.json"), "utf8")) : [];
const userMessages = messagesJson.filter((m: { role: string }) => m.role === "user").length;
const midRun = calls.filter(c => {
	const at = Date.parse(c.at) - t0;
	return at > agentStart && at < agentEnd;
}).length;

const md = [
	`# Live smoke ${stamp}`,
	"",
	`- omp: ${Bun.spawnSync(["omp", "--version"]).stdout.toString().trim()}; primary model: ${primary}; observer role: @${role}`,
	`- result: ${failure ? `FAILED: ${failure}` : "passed"}`,
	`- observer calls: ${calls.length} (${midRun} while the agent was running, agent ran ${((agentEnd - agentStart) / 1000).toFixed(1)} s)`,
	`- observer models used: ${[...new Set(calls.map(c => c.model))].join(", ") || "none"}`,
	`- tokens: ${calls.reduce((a, c) => a + (c.usage?.input ?? 0), 0)} input / ${calls.reduce((a, c) => a + (c.usage?.output ?? 0), 0)} output`,
	`- primary transcript: ${messagesJson.length} messages, ${userMessages} user message(s); observer markers in session file: ${leaks.length ? leaks.join(", ") : "none"}`,
	`- scratch dir (disposable): ${scratch}`,
	"",
	"## Steps",
	"",
	"```",
	...steps,
	"```",
	"",
	"## Timeline (seconds from agent_start)",
	"",
	"| t | event | detail |",
	"|---|---|---|",
	...timeline,
	"",
	"## Observer calls: observed (sent) vs predicted (reply)",
	"",
	...calls.flatMap((c, i) => [
		`### Call ${i + 1} at ${((Date.parse(c.at) - t0 - agentStart) / 1000).toFixed(1)} s, ${c.ms} ms, ${c.model}, stop ${c.stopReason}`,
		"",
		"Sent:",
		"```json",
		JSON.stringify(JSON.parse(c.sent), null, 1),
		"```",
		"Reply:",
		"```",
		c.reply || c.error,
		"```",
		"",
	]),
].join("\n");
writeFileSync(join(evidence, "summary.md"), md);
console.log(`\nEvidence: ${evidence}`);
process.exit(failure ? 1 : 0);
