// Turns one historic OMP session file into a replay trace: the sanitized,
// timestamped extension events the widget would have observed (see
// sanitize.ts), plus offline labels computed from the whole session.
// Events and labels are kept apart: the replay reads only `events`, and only
// the scorer reads `prompts`. The session file is opened read-only.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { applyEvent, emptyFacts, type ObservedEvent } from "../src/facts";
import { describeAgentMessage, describeIntent, describeRequest, listedItems, sanitizeToolName, TodoTitles } from "./sanitize";

export interface TraceEvent {
	t: number;
	event: ObservedEvent;
}

/** Coarse kinds of the agent's next action, for next-step scoring. */
export type Action = "inspect" | "edit" | "run" | "plan" | "delegate" | "ask" | "research" | "reply" | "other";

export type Outcome = "completed" | "aborted" | "error" | "superseded" | "unfinished";

export interface PromptLabel {
	start: number;
	/** Turn end, the next prompt (superseded), or the last event (unfinished). */
	end: number;
	outcome: Outcome;
	/** List items in the request ("1.", "-", ...). */
	listed: number;
	/** The agent called the todo tool during this prompt. */
	todoActive: boolean;
	/** A finite plan exists: ≥2 listed items or an active todo list. Otherwise the prompt is open-ended. */
	structured: boolean;
	/** Start times of the prompt's tool calls. */
	toolStarts: number[];
	/** Every next action: tool starts, then the final reply. */
	actions: Array<{ t: number; action: Action }>;
	/** Todo items done (non-dropped list) after each todo call in the prompt, with the count at start. */
	todoDone: Array<{ t: number; done: number }>;
}

export interface Trace {
	id: string;
	project: string;
	events: TraceEvent[];
	prompts: PromptLabel[];
}

const ACTION_BY_TOOL: Record<string, Action> = {
	read: "inspect", grep: "inspect", glob: "inspect", find: "inspect", lsp: "inspect", ast_grep: "inspect", inspect_image: "inspect",
	edit: "edit", write: "edit", ast_edit: "edit", notebook: "edit",
	bash: "run", eval: "run", python: "run", browser: "run", debug: "run", ssh: "run",
	todo: "plan", todo_write: "plan",
	task: "delegate",
	ask: "ask",
	web_search: "research", fetch: "research",
};

export const actionOfTool = (name: string): Action => ACTION_BY_TOOL[name] ?? "other";

export const pseudonym = (kind: string, value: string) => `${kind}-${createHash("sha256").update(`omp-progress-bench:${value}`).digest("hex").slice(0, 8)}`;

interface Entry {
	type: string;
	id?: string;
	parentId?: string | null;
	timestamp?: string;
	customType?: string;
	data?: Record<string, unknown>;
	cwd?: string;
	message?: {
		role?: string;
		content?: unknown;
		stopReason?: string;
		toolCallId?: string;
		toolName?: string;
		isError?: boolean;
	};
}

const textOf = (content: unknown): string =>
	typeof content === "string"
		? content
		: Array.isArray(content)
			? content
					.filter((b): b is { type: "text"; text: string } => b?.type === "text" && typeof b.text === "string")
					.map(b => b.text)
					.join("\n")
			: "";

const time = (value: unknown): number => (typeof value === "number" ? value : typeof value === "string" ? Date.parse(value) : Number.NaN);

/** The session's current branch, root first. */
function mainPath(entries: Entry[]): Entry[] {
	const byId = new Map(entries.filter(e => e.id).map(e => [e.id as string, e]));
	const path: Entry[] = [];
	let cur = [...entries].reverse().find(e => e.id);
	while (cur) {
		path.push(cur);
		cur = cur.parentId ? byId.get(cur.parentId) : undefined;
	}
	return path.reverse();
}

function sanitizeTodoArgs(args: Record<string, unknown>, titles: TodoTitles): Record<string, unknown> {
	const map = (v: unknown) => (Array.isArray(v) ? v.map(x => titles.title(x)) : typeof v === "string" ? titles.title(v) : undefined);
	const out: Record<string, unknown> = { op: args.op };
	const itemsOf = (phase: unknown) => (phase && typeof phase === "object" && "items" in phase ? phase.items : undefined);
	if (Array.isArray(args.list)) out.list = args.list.map(phase => ({ items: map(itemsOf(phase)) ?? [] }));
	if (args.items !== undefined) out.items = map(args.items);
	if (args.task !== undefined) out.task = map(args.task);
	return out;
}

export function extractSession(path: string): Trace {
	const entries: Entry[] = [];
	for (const line of readFileSync(path, "utf8").split("\n")) {
		if (!line.trim()) continue;
		try {
			entries.push(JSON.parse(line));
		} catch {
			// a torn final line
		}
	}
	const header = entries.find(e => e.type === "session");
	const path_ = mainPath(entries);
	const toolStartAt = new Map<string, number>();
	for (const e of path_) {
		if (e.type === "custom" && e.customType === "tool_execution_start" && e.data) {
			const t = time(e.data.startedAt);
			if (Number.isFinite(t)) toolStartAt.set(String(e.data.toolCallId), t);
		}
	}
	const messages = path_.filter(e => e.type === "message" && e.message && Number.isFinite(time(e.timestamp)));
	const titles = new TodoTitles();
	const events: TraceEvent[] = [];
	const push = (t: number, event: ObservedEvent) => events.push({ t, event });

	for (let i = 0; i < messages.length; i++) {
		const e = messages[i] as Entry & { message: NonNullable<Entry["message"]> };
		const t = time(e.timestamp);
		const m = e.message;
		if (m.role === "user") {
			const text = textOf(m.content);
			if (!text.trim()) continue;
			// The labeler needs the list count; the replay only sees the sanitized description.
			push(t, { type: "before_agent_start", prompt: describeRequest(text), listed: listedItems(text) });
		} else if (m.role === "assistant") {
			const blocks = Array.isArray(m.content) ? (m.content as Array<Record<string, unknown>>) : [];
			const text = textOf(blocks);
			const message: Record<string, unknown> = { role: "assistant", stopReason: m.stopReason, content: text.trim() ? [{ type: "text", text: describeAgentMessage(text) }] : [] };
			if (m.stopReason === "error") message.errorMessage = "provider error";
			push(t, { type: "message_end", message });
			const calls = blocks.filter(b => b.type === "toolCall");
			if (m.stopReason !== "aborted") {
				for (const call of calls) {
					const id = String(call.id);
					const name = sanitizeToolName(call.name);
					const args = call.arguments && typeof call.arguments === "object" ? (call.arguments as Record<string, unknown>) : {};
					push(Math.max(t, toolStartAt.get(id) ?? t), {
						type: "tool_execution_start",
						toolCallId: id,
						toolName: name,
						intent: describeIntent(call.intent),
						args: name === "todo" ? sanitizeTodoArgs(args, titles) : undefined,
					});
				}
			}
			const next = messages[i + 1]?.message?.role;
			const retried = m.stopReason === "error" && next === "assistant";
			if ((calls.length === 0 || m.stopReason === "aborted") && ["stop", "aborted", "error"].includes(m.stopReason ?? "") && !retried) {
				push(t + 1, { type: "agent_end", stopReason: m.stopReason });
			}
		} else if (m.role === "toolResult") {
			push(t, { type: "tool_execution_end", toolCallId: String(m.toolCallId), toolName: sanitizeToolName(m.toolName), isError: m.isError === true });
		}
	}
	events.sort((a, b) => a.t - b.t);
	return { id: pseudonym("s", String(header?.id ?? path)), project: pseudonym("p", String(header?.cwd ?? "")), events, prompts: label(events) };
}

/** Offline labels, from the whole event sequence (future included). */
export function label(events: TraceEvent[]): PromptLabel[] {
	const prompts: PromptLabel[] = [];
	const facts = emptyFacts();
	let cur: PromptLabel | null = null;
	const finish = (p: PromptLabel, end: number, outcome: Outcome) => {
		p.end = end;
		p.outcome = outcome;
		p.structured = p.listed >= 2 || p.todoActive;
	};
	const doneCount = () => facts.todos.filter(t => t.status === "done").length;
	for (const { t, event } of events) {
		if (event.type === "before_agent_start") {
			if (cur) finish(cur, t, "superseded");
			cur = { start: t, end: t, outcome: "unfinished", listed: Number(event.listed) || 0, todoActive: false, structured: false, toolStarts: [], actions: [], todoDone: [{ t, done: doneCount() }] };
			prompts.push(cur);
		}
		applyEvent(facts, event);
		if (!cur) continue;
		cur.end = t;
		if (event.type === "tool_execution_start") {
			const name = String(event.toolName);
			cur.toolStarts.push(t);
			cur.actions.push({ t, action: actionOfTool(name) });
			if (name === "todo") {
				cur.todoActive = true;
				cur.todoDone.push({ t, done: doneCount() });
			}
		} else if (event.type === "agent_end") {
			cur.actions.push({ t, action: "reply" });
			const reason = String(event.stopReason);
			finish(cur, t, reason === "stop" ? "completed" : reason === "aborted" ? "aborted" : "error");
			cur = null;
		}
	}
	if (cur) finish(cur, cur.end, "unfinished");
	return prompts;
}
