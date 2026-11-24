// Turns one historic OMP session file into the extension events the widget
// would have seen live (real text; the production `applyEvent` cleans and
// redacts it exactly as in a live session), plus per-prompt labels computed
// from the whole session. The session file is opened read-only. Events are
// only ever kept in memory: nothing from them is written into the repository.

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import type { ObservedEvent } from "../src/facts";

export interface TraceEvent {
	t: number;
	event: ObservedEvent;
}

export type Outcome = "completed" | "aborted" | "error" | "superseded" | "unfinished";

export interface PromptLabel {
	/** Index of the prompt's `before_agent_start` event. */
	startIndex: number;
	outcome: Outcome;
	/** Event indices of the prompt's tool starts, in order. */
	toolStartIndices: number[];
	/** The agent called the todo tool during this prompt. */
	todoActive: boolean;
}

export interface Trace {
	id: string;
	project: string;
	events: TraceEvent[];
	prompts: PromptLabel[];
}

export const pseudonym = (kind: string, value: string) => `${kind}-${createHash("sha256").update(`omp-progress-eval:${value}`).digest("hex").slice(0, 8)}`;

interface Entry {
	type: string;
	id?: string;
	parentId?: string | null;
	timestamp?: string;
	customType?: string;
	data?: Record<string, unknown>;
	cwd?: string;
	message?: { role?: string; content?: unknown; stopReason?: string; toolCallId?: string; toolName?: string; isError?: boolean };
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

/** The tool arguments the extension's event handler reads: a target path and the todo tool's own arguments. */
function eventArgs(name: string, args: Record<string, unknown>): Record<string, unknown> | undefined {
	if (name === "todo") return { op: args.op, list: args.list, items: args.items, task: args.task };
	return typeof args.path === "string" ? { path: args.path } : undefined;
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
	const branch = mainPath(entries);
	const toolStartAt = new Map<string, number>();
	for (const e of branch) {
		if (e.type === "custom" && e.customType === "tool_execution_start" && e.data) {
			const t = time(e.data.startedAt);
			if (Number.isFinite(t)) toolStartAt.set(String(e.data.toolCallId), t);
		}
	}
	const messages = branch.filter(e => e.type === "message" && e.message && Number.isFinite(time(e.timestamp)));
	const events: TraceEvent[] = [];
	const push = (t: number, event: ObservedEvent) => events.push({ t, event });

	for (let i = 0; i < messages.length; i++) {
		const e = messages[i] as Entry & { message: NonNullable<Entry["message"]> };
		const t = time(e.timestamp);
		const m = e.message;
		if (m.role === "user") {
			const text = textOf(m.content);
			if (text.trim()) push(t, { type: "before_agent_start", prompt: text });
		} else if (m.role === "assistant") {
			const blocks = Array.isArray(m.content) ? (m.content as Array<Record<string, unknown>>) : [];
			const text = textOf(blocks);
			const message: Record<string, unknown> = { role: "assistant", stopReason: m.stopReason, content: text.trim() ? [{ type: "text", text }] : [] };
			if (m.stopReason === "error") message.errorMessage = "provider error";
			push(t, { type: "message_end", message });
			const calls = blocks.filter(b => b.type === "toolCall");
			if (m.stopReason !== "aborted") {
				for (const call of calls) {
					const id = String(call.id);
					const name = typeof call.name === "string" ? call.name : "tool";
					const args = call.arguments && typeof call.arguments === "object" ? (call.arguments as Record<string, unknown>) : {};
					push(Math.max(t, toolStartAt.get(id) ?? t), { type: "tool_execution_start", toolCallId: id, toolName: name, intent: call.intent, args: eventArgs(name, args) });
				}
			}
			const next = messages[i + 1]?.message?.role;
			const retried = m.stopReason === "error" && next === "assistant";
			if ((calls.length === 0 || m.stopReason === "aborted") && ["stop", "aborted", "error"].includes(m.stopReason ?? "") && !retried) {
				push(t + 1, { type: "agent_end", stopReason: m.stopReason });
			}
		} else if (m.role === "toolResult") {
			push(t, { type: "tool_execution_end", toolCallId: String(m.toolCallId), toolName: m.toolName, isError: m.isError === true });
		}
	}
	// Stable: events of one message keep their order.
	events.sort((a, b) => a.t - b.t);
	return { id: pseudonym("s", String(header?.id ?? path)), project: pseudonym("p", String(header?.cwd ?? "")), events, prompts: label(events) };
}

/** Offline labels, from the whole event sequence (future included). */
export function label(events: TraceEvent[]): PromptLabel[] {
	const prompts: PromptLabel[] = [];
	let cur: PromptLabel | null = null;
	events.forEach(({ event }, index) => {
		if (event.type === "before_agent_start") {
			if (cur) cur.outcome = "superseded";
			cur = { startIndex: index, outcome: "unfinished", toolStartIndices: [], todoActive: false };
			prompts.push(cur);
		}
		if (!cur) return;
		if (event.type === "tool_execution_start") {
			cur.toolStartIndices.push(index);
			if (event.toolName === "todo") cur.todoActive = true;
		} else if (event.type === "agent_end") {
			const reason = String(event.stopReason);
			cur.outcome = reason === "stop" ? "completed" : reason === "aborted" ? "aborted" : "error";
			cur = null;
		}
	});
	return prompts;
}
