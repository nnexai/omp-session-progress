// omp-session-progress: an opt-in, read-only progress observer for the current
// OMP session. A cheap role model (@tiny by default) reads a bounded digest of
// the session's events and its answer is shown in the status line and by
// `/progress`. Nothing is sent into the observed session.

import { appendFileSync } from "node:fs";
import { completeSimple } from "@oh-my-pi/pi-ai";
import type { ObservedEvent } from "./facts";
import { Observer, type Report, reportText, statusText, type Transport } from "./observer";

const STATUS_KEY = "session-progress";
const ROLES = ["tiny", "widget"] as const;
type Role = (typeof ROLES)[number];
const DEFAULT_ROLE: Role = "tiny";
const MAX_OUTPUT_TOKENS = 200;

// The slice of the OMP extension API this file uses (19.1.x).
interface Model {
	provider: string;
	id: string;
	name?: string;
	reasoning?: boolean;
	contextWindow?: number;
	cost?: { input: number; output: number };
}
interface Ui {
	notify(message: string, type?: "info" | "warning" | "error"): void;
	setStatus(key: string, text: string | undefined): void;
	confirm(title: string, message: string): Promise<boolean>;
}
interface Ctx {
	mode: string;
	hasUI: boolean;
	ui: Ui;
	agent?: { kind: "main" | "sub" };
	sessionManager: { getSessionId(): string; getBranch(): Array<{ type: string; message?: { role?: string; content?: unknown } }> };
	modelRegistry: { getApiKey(model: Model, sessionId?: string, options?: { signal?: AbortSignal }): Promise<string | undefined> };
	models: { resolve(spec: string): Model | undefined; current(): Model | undefined };
	isIdle(): boolean;
	setTimeout(fn: () => void, ms: number): unknown;
	clearTimer(handle: unknown): void;
}
interface Api {
	on(event: string, handler: (event: ObservedEvent, ctx: Ctx) => unknown): void;
	registerCommand(name: string, options: { description: string; handler: (args: string, ctx: Ctx) => Promise<void> | void }): void;
}

const label = (role: Role, model: Model) => `@${role} → ${model.provider}/${model.id}`;

function describeModel(role: Role, model: Model | undefined): string {
	if (!model) return `@${role}: does not resolve to an available model`;
	const cost = model.cost ? `$${model.cost.input}/$${model.cost.output} per Mtok in/out` : "cost unknown";
	return `${label(role, model)} (${model.name ?? model.id}; reasoning ${model.reasoning ? "yes" : "no"}; context ${model.contextWindow ?? "?"}; ${cost})`;
}

function makeTransport(ctx: Ctx, role: Role, model: Model, sessionId: string): Transport {
	// A key of its own, so the observer never shares a provider cache lineage with the session.
	const key = `${sessionId}:session-progress`;
	const logPath = process.env.OMP_SESSION_PROGRESS_LOG;
	return async ({ system, user, signal }) => {
		const started = Date.now();
		const apiKey = await ctx.modelRegistry.getApiKey(model, key, { signal });
		if (!apiKey) throw new Error(`no credentials for ${model.provider}`);
		const reply = await completeSimple(
			model,
			{ systemPrompt: [system], messages: [{ role: "user", content: user, timestamp: started }] },
			{ apiKey, sessionId: key, maxTokens: MAX_OUTPUT_TOKENS, disableReasoning: true, signal },
		);
		const text = reply.content
			.filter((block: { type: string }) => block.type === "text")
			.map((block: { text: string }) => block.text)
			.join("");
		if (logPath) {
			appendFileSync(
				logPath,
				`${JSON.stringify({ at: new Date(started).toISOString(), ms: Date.now() - started, role: `@${role}`, model: `${model.provider}/${model.id}`, sent: user, stopReason: reply.stopReason, error: reply.errorMessage, reply: text, usage: reply.usage })}\n`,
			);
		}
		if (reply.stopReason === "error" || reply.stopReason === "aborted") throw new Error(reply.errorMessage || reply.stopReason);
		return { text, inputTokens: reply.usage?.input, outputTokens: reply.usage?.output };
	};
}

// The request the session is currently working on, so an opt-in mid-task has context.
function latestRequest(ctx: Ctx): string | undefined {
	const branch = ctx.sessionManager.getBranch();
	for (let i = branch.length - 1; i >= 0; i--) {
		const message = branch[i]?.message;
		if (branch[i]?.type !== "message" || message?.role !== "user") continue;
		const content = message.content;
		if (typeof content === "string") return content;
		if (Array.isArray(content)) {
			return content
				.filter((block): block is { type: "text"; text: string } => block?.type === "text" && typeof block.text === "string")
				.map(block => block.text)
				.join(" ");
		}
		return undefined;
	}
	return undefined;
}

const PRIVACY = [
	"Sent to the observer model on each refresh (about 1-3 KB, at most every 15 s while work happens, at most 30 calls):",
	"• your latest request (first 280 chars), todo titles, tool names with the agent's one-line intent (or just the target file name), success/failure counts,",
	"• provider retry/error notes and the first sentence of the agent's latest reply.",
	"Never sent: commands, other tool arguments, tool output, file contents, thinking, other sessions. Code spans and credential-like tokens are redacted.",
	"The observer only displays its reading; it sends nothing into this session. Off again with /progress off; it never persists.",
].join("\n");

export default function sessionProgress(pi: Api) {
	const isMain = (ctx: Ctx) => ctx.agent?.kind !== "sub";
	let ui: Ui | null = null;
	let timerCtx: Ctx | null = null;
	const observer = new Observer({
		now: () => Date.now(),
		setTimeout: (fn, ms) => timerCtx?.setTimeout(fn, ms),
		clearTimeout: handle => {
			if (handle !== undefined) timerCtx?.clearTimer(handle);
		},
	});

	const turnOff = () => {
		observer.disable();
		ui?.setStatus(STATUS_KEY, undefined);
		ui = null;
		timerCtx = null;
	};

	const forward = (event: ObservedEvent, ctx: Ctx) => {
		if (observer.phase === "off" || !isMain(ctx)) return;
		try {
			observer.observe(ctx.sessionManager.getSessionId(), event);
		} catch {
			// Observing must never disturb the session.
		}
	};

	for (const type of [
		"before_agent_start",
		"agent_start",
		"agent_end",
		"tool_execution_start",
		"tool_execution_end",
		"message_end",
		"auto_retry_start",
		"auto_retry_end",
	]) {
		pi.on(type, (event, ctx) => {
			forward({ ...event, type }, ctx);
		});
	}

	// Opt-in is per session: any switch (/new, /resume, /fork) turns it off.
	pi.on("session_switch", (_event, ctx) => {
		if (isMain(ctx)) turnOff();
	});
	pi.on("session_shutdown", (_event, ctx) => {
		if (isMain(ctx)) turnOff();
	});
	for (const type of ["session_branch", "session_tree", "session_clear"]) {
		pi.on(type, (_event, ctx) => {
			if (isMain(ctx) && observer.sessionId === ctx.sessionManager.getSessionId()) observer.reset();
		});
	}

	pi.registerCommand("progress", {
		description: "Opt-in progress observer for this session: /progress [on [tiny|widget] | off | roles]",
		handler: async (args, ctx) => {
			const [verb = "", roleArg] = args.trim().split(/\s+/);
			if (!isMain(ctx)) return;
			if (verb === "off") {
				const wasOn = observer.phase !== "off";
				turnOff();
				ctx.ui.notify(wasOn ? "Session progress observer off." : "Session progress observer was already off.", "info");
				return;
			}
			if (verb === "roles") {
				ctx.ui.notify(ROLES.map(role => describeModel(role, ctx.models.resolve(`@${role}`))).join("\n"), "info");
				return;
			}
			if (verb === "on") {
				const role = (roleArg ?? DEFAULT_ROLE).replace(/^@/, "") as Role;
				if (!ROLES.includes(role)) {
					ctx.ui.notify(`Unknown observer role "${roleArg}". Use tiny or widget.`, "warning");
					return;
				}
				if (!ctx.hasUI) {
					ctx.ui.notify("The progress observer needs a UI to show its status.", "warning");
					return;
				}
				const model = ctx.models.resolve(`@${role}`);
				if (!model) {
					// No fallback to another role or the session's model: the user picks the role.
					ctx.ui.notify(`@${role} does not resolve to an available model; set modelRoles.${role} or pick the other role.`, "error");
					return;
				}
				const consent = await ctx.ui.confirm(`Watch this session with ${label(role, model)}?`, PRIVACY);
				if (!consent) {
					ctx.ui.notify("Session progress observer stays off.", "info");
					return;
				}
				const sessionId = ctx.sessionManager.getSessionId();
				ui = ctx.ui;
				timerCtx = ctx;
				const sink = (report: Report) => {
					if (report.sessionId === sessionId) ctx.ui.setStatus(STATUS_KEY, statusText(report));
				};
				observer.enable(sessionId, label(role, model), makeTransport(ctx, role, model, sessionId), sink);
				// Context from before the opt-in is recorded but triggers no call; the next event does.
				const request = latestRequest(ctx);
				if (request) observer.seed(sessionId, { type: "before_agent_start", prompt: request });
				observer.seed(sessionId, { type: ctx.isIdle() ? "agent_end" : "agent_start" });
				ctx.ui.notify(`Session progress observer on (${label(role, model)}). /progress shows the report, /progress off stops it.`, "info");
				return;
			}
			if (verb !== "" && verb !== "show") {
				ctx.ui.notify("Usage: /progress [on [tiny|widget] | off | roles | show]", "warning");
				return;
			}
			ctx.ui.notify(reportText(observer.report(), Date.now()), "info");
		},
	});
}
