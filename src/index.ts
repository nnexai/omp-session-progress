// omp-session-progress: an opt-in, read-only progress widget for the current
// OMP session. The @widget role model reads a bounded digest of the session's
// events and estimates how far along the current prompt is; the reading is
// drawn in a native TUI widget above the editor. Nothing is sent into the
// observed session.

import { appendFileSync } from "node:fs";
import { truncateToWidth } from "@oh-my-pi/pi-tui";
import type { ObservedEvent } from "./facts";
import { Observer, type Report } from "./observer";
import { type CallRecord, type Credentials, makeTransport, type Model, ROLE } from "./transport";
import { type Tone, widgetLines } from "./widget";

const WIDGET_KEY = "session-progress";
// Width used for the plain string-array widget of RPC hosts, which have no TUI width.
const RPC_WIDTH = 78;

// The slice of the OMP extension API this file uses (19.1.x).
interface Component {
	render(width: number): string[];
	invalidate(): void;
}
type WidgetFactory = (tui: { requestRender?: () => void }, theme: { fg(tone: Tone, text: string): string }) => Component;
interface Ui {
	notify(message: string, type?: "info" | "warning" | "error"): void;
	setWidget(key: string, content: WidgetFactory | string[] | undefined, options?: { placement?: "aboveEditor" | "belowEditor" }): void;
	confirm(title: string, message: string): Promise<boolean>;
}
interface Ctx {
	mode: string;
	hasUI: boolean;
	ui: Ui;
	agent?: { kind: "main" | "sub" };
	sessionManager: { getSessionId(): string; getBranch(): Array<{ type: string; message?: { role?: string; content?: unknown } }> };
	modelRegistry: Credentials;
	models: { resolve(spec: string): Model | undefined };
	isIdle(): boolean;
	setTimeout(fn: () => void, ms: number): unknown;
	clearTimer(handle: unknown): void;
}
interface Api {
	on(event: string, handler: (event: ObservedEvent, ctx: Ctx) => unknown): void;
	registerCommand(name: string, options: { description: string; handler: (args: string, ctx: Ctx) => Promise<void> | void }): void;
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
	"Sent to the observer model on each refresh (at most 3.5 KB, at most every 15 s while work happens, at most 30 calls):",
	"• your current request (first 500 chars) and the previous one (160), todo titles, tool names with the agent's one-line intent",
	"  (or just the target file name), per-prompt success/failure counts, provider retry/error notes, and the first sentence of the agent's latest reply.",
	"Never sent: commands, other tool arguments, tool output, file contents, thinking, other sessions. Code spans and credential-like tokens are redacted.",
	"The widget only displays the observer's reading; nothing is sent into this session. /progress off removes it; it never persists.",
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
		ui?.setWidget(WIDGET_KEY, undefined);
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

	const turnOn = async (ctx: Ctx) => {
		if (!ctx.hasUI) {
			ctx.ui.notify("The progress widget needs the interactive TUI.", "warning");
			return;
		}
		// OMP's own role resolution; no other role and not the session's model.
		const model = ctx.models.resolve(ROLE);
		if (!model) {
			ctx.ui.notify(`${ROLE} does not resolve to an available model; set modelRoles.widget (or smol, which it inherits).`, "error");
			return;
		}
		const modelLabel = `${ROLE} → ${model.provider}/${model.id}`;
		if (!(await ctx.ui.confirm(`Show a progress widget estimated by ${modelLabel}?`, PRIVACY))) return;
		const sessionId = ctx.sessionManager.getSessionId();
		ui = ctx.ui;
		timerCtx = ctx;
		const target = ctx.ui;
		let latest: Report | null = null;
		let requestRender: (() => void) | undefined;
		if (ctx.mode === "tui") {
			// A width-aware component: one physical row per line at any terminal width. Not focusable.
			target.setWidget(
				WIDGET_KEY,
				(tui, theme) => {
					requestRender = tui.requestRender?.bind(tui);
					return {
						render: width =>
							latest ? widgetLines(latest, Date.now(), width, (tone, text) => theme.fg(tone, text)).map(line => truncateToWidth(line, width)) : [],
						invalidate() {},
					};
				},
				{ placement: "aboveEditor" },
			);
		}
		const sink = (report: Report) => {
			if (report.sessionId !== sessionId) return;
			latest = report;
			if (ctx.mode === "tui") requestRender?.();
			// RPC hosts take plain string-array widgets.
			else target.setWidget(WIDGET_KEY, widgetLines(report, Date.now(), RPC_WIDTH, (_tone, text) => text), { placement: "aboveEditor" });
		};
		const logPath = process.env.OMP_SESSION_PROGRESS_LOG;
		const log = logPath
			? (call: CallRecord) =>
					appendFileSync(
						logPath,
						`${JSON.stringify({ at: new Date(call.startedAt).toISOString(), ms: call.ms, role: ROLE, model: call.model, sent: call.sent, stopReason: call.stopReason, error: call.error, reply: call.reply, usage: call.usage })}\n`,
					)
			: undefined;
		observer.enable(sessionId, makeTransport(ctx.modelRegistry, model, sessionId, log), sink);
		// Context from before the opt-in is recorded but triggers no call; the next event does.
		const request = latestRequest(ctx);
		if (request) observer.seed(sessionId, { type: "before_agent_start", prompt: request });
		observer.seed(sessionId, { type: ctx.isIdle() ? "agent_end" : "agent_start" });
	};

	pi.registerCommand("progress", {
		description: "Progress widget for this session, estimated by the @widget model: /progress [on | off]",
		handler: async (args, ctx) => {
			if (!isMain(ctx)) return;
			const verb = args.trim();
			if (verb === "off" || (verb === "" && observer.phase !== "off")) turnOff();
			else if (verb === "on" || verb === "") {
				// From off, or to restart after a pause or the call cap.
				if (observer.phase !== "on") await turnOn(ctx);
			} else ctx.ui.notify("Usage: /progress [on | off]", "warning");
		},
	});
}
