// omp-session-progress: an opt-in, read-only, zero-LLM progress widget for the
// current OMP session. A compact local GBDT model estimates progress and
// remaining active time from live session events in < 0.1 ms, with zero API
// cost and nothing ever leaving the machine.

import { truncateToWidth } from "@oh-my-pi/pi-tui";
import type { BranchEntry, ObservedEvent } from "./facts";
import { Observer, type Report } from "./observer";
import { type Tone, type WidgetStyle, widgetLines } from "./widget";

const WIDGET_KEY = "session-progress";
const RPC_WIDTH = 78;

interface Component {
	render(width: number): string[];
	invalidate(): void;
}

interface OmpTheme {
	fg(tone: Tone, text: string): string;
	bold?(text: string): string;
	progress?: { filled?: string; empty?: string };
	tree?: { branch?: string; last?: string };
	status?: { done?: string; warning?: string };
	icon?: { time?: string; warning?: string };
}

type WidgetFactory = (tui: { requestRender?: () => void }, theme: OmpTheme) => Component;

interface Ui {
	notify(message: string, type?: "info" | "warning" | "error"): void;
	setWidget(key: string, content: WidgetFactory | string[] | undefined, options?: { placement?: "aboveEditor" | "belowEditor" }): void;
}

interface Ctx {
	mode: string;
	hasUI: boolean;
	ui: Ui;
	agent?: { kind: "main" | "sub" };
	sessionManager: { getSessionId(): string; getBranch(): BranchEntry[] };
	isIdle(): boolean;
	setTimeout(fn: () => void, ms: number): unknown;
	clearTimer(handle: unknown): void;
}

interface Api {
	on(event: string, handler: (event: ObservedEvent, ctx: Ctx) => unknown): void;
	registerCommand(name: string, options: { description: string; handler: (args: string, ctx: Ctx) => Promise<void> | void }): void;
}

function makeStyle(theme: OmpTheme): WidgetStyle {
	return {
		fg: (tone, text) => theme.fg(tone, text),
		bold: theme.bold ? text => theme.bold!(text) : undefined,
		filled: theme.progress?.filled,
		empty: theme.progress?.empty,
		branch: theme.tree?.branch,
		last: theme.tree?.last,
		check: theme.status?.done,
		warn: theme.icon?.warning ?? theme.status?.warning,
		clock: theme.icon?.time,
	};
}

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

	const turnOn = (ctx: Ctx) => {
		if (!ctx.hasUI) {
			ctx.ui.notify("The progress widget needs an interactive UI.", "warning");
			return;
		}
		const sessionId = ctx.sessionManager.getSessionId();
		ui = ctx.ui;
		timerCtx = ctx;
		const target = ctx.ui;
		let latest: Report | null = null;
		let requestRender: (() => void) | undefined;

		if (ctx.mode === "tui") {
			target.setWidget(
				WIDGET_KEY,
				(tui, theme) => {
					requestRender = tui.requestRender?.bind(tui);
					const style = makeStyle(theme);
					return {
						render: width =>
							latest ? widgetLines(latest, Date.now(), width, style).map(line => truncateToWidth(line, width)) : [],
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
			else target.setWidget(WIDGET_KEY, widgetLines(report, Date.now(), RPC_WIDTH, (_tone, text) => text), { placement: "aboveEditor" });
		};

		observer.enable(sessionId, sink);
		// Recover full current-session history (elapsed time, todos, tools, prior prompts) on mid-session opt-in.
		try {
			const branch = ctx.sessionManager.getBranch();
			if (Array.isArray(branch) && branch.length > 0) {
				observer.seedBranch(sessionId, branch, ctx.isIdle());
			}
		} catch {
			// Best-effort branch replay; live events still work if branch is unavailable.
		}
	};

	pi.registerCommand("progress", {
		description: "Local progress widget for this session (zero LLM calls): /progress [on | off]",
		handler: (args, ctx) => {
			if (!isMain(ctx)) return;
			const verb = args.trim();
			if (verb === "off" || (verb === "" && observer.phase !== "off")) turnOff();
			else if (verb === "on" || verb === "") {
				if (observer.phase !== "on") turnOn(ctx);
			} else ctx.ui.notify("Usage: /progress [on | off]", "warning");
		},
	});
}
