// Observed facts about one session, built only from OMP extension events.
//
// Privacy boundary: this module decides what can ever reach the auxiliary
// model. It keeps tool names, the model-written one-line tool intent (or, when
// the session has no intents, the file name a tool targets), todo task titles,
// success/failure flags, the user's latest request (trimmed), and the first
// sentence of the agent's latest reply. It never keeps other tool arguments
// (commands, content, patterns), tool results, file contents, or thinking.
// Every kept string is stripped of control characters and code, scrubbed of
// credential-looking tokens and length-capped.

export const LIMITS = {
	goal: 500,
	previousGoal: 160,
	note: 160,
	intent: 90,
	todoText: 80,
	todoItems: 12,
	recent: 8,
	failures: 3,
	payloadChars: 3500,
} as const;

export interface TodoItem {
	text: string;
	status: "pending" | "in_progress" | "done" | "dropped";
}

export interface Facts {
	running: boolean;
	/** Requests seen since opt-in; the current prompt is number `requests`. */
	requests: number;
	goal: string | null;
	previousGoal: string | null;
	/** The agent's todo list; it can outlive one prompt. */
	todos: TodoItem[];
	/** Counts and labels below are for the current prompt only. */
	toolCalls: number;
	failedCalls: number;
	underway: Map<string, string>; // toolCallId -> label, tools started and not ended
	recent: string[]; // labels of finished tool calls, newest last
	failures: string[]; // labels of failed tool calls, newest last
	retry: string | null;
	agentNote: string | null;
}

export function emptyFacts(): Facts {
	return {
		running: false,
		requests: 0,
		goal: null,
		previousGoal: null,
		todos: [],
		toolCalls: 0,
		failedCalls: 0,
		underway: new Map(),
		recent: [],
		failures: [],
		retry: null,
		agentNote: null,
	};
}

const SECRET_PATTERNS: RegExp[] = [
	/-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(-----END [A-Z ]*PRIVATE KEY-----|$)/g,
	/\b(sk|rk|pk)-[A-Za-z0-9_-]{12,}/g,
	/\b(ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}/g,
	/\bgithub_pat_[A-Za-z0-9_]{20,}/g,
	/\bglpat-[A-Za-z0-9_-]{16,}/g,
	/\bxox[abprs]-[A-Za-z0-9-]{10,}/g,
	/\bAKIA[0-9A-Z]{16}\b/g,
	/\bAIza[0-9A-Za-z_-]{30,}/g,
	/\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/g,
	/\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}/gi,
	/\b([A-Za-z_]*(api[_-]?key|token|secret|password|passwd|pwd|credential)[A-Za-z_]*)\s*[:=]\s*("[^"]*"|'[^']*'|\S+)/gi,
	/[a-z][a-z0-9+.-]*:\/\/[^\s/:@]+:[^\s/@]+@/gi, // user:password@ in URLs
	/\b[0-9a-f]{32,}\b/gi,
	/\b[A-Za-z0-9+/_-]{40,}={0,2}/g,
];

/** Strip control/ANSI, code, and credential-looking tokens; collapse whitespace; cap length. */
export function clean(input: unknown, max: number): string {
	if (typeof input !== "string") return "";
	let s = input
		.replace(/```[\s\S]*?(```|$)/g, " [code] ")
		.replace(/`[^`\n]*`/g, "[code]")
		.replace(/\x1b\[[0-9;?]*[ -/]*[@-~]/g, "")
		.replace(/[\u0000-\u0008\u000b-\u001f\u007f-\u009f]/g, " ");
	for (const pattern of SECRET_PATTERNS) s = s.replace(pattern, "[redacted]");
	s = s.replace(/\s+/g, " ").trim();
	return s.length > max ? `${s.slice(0, max - 1).trimEnd()}…` : s;
}

function pushBounded(list: string[], value: string, max: number): void {
	list.push(value);
	if (list.length > max) list.splice(0, list.length - max);
}

function toolLabel(toolName: unknown, intent: unknown, args: Record<string, unknown> | undefined): string {
	const name = clean(toolName, 40) || "tool";
	// Intents exist when tools.intentTracing is on (the TUI default); otherwise name the target file only.
	// "[a/b.ts#TAG]" and "b.ts:10-20" selectors name the file b.ts.
	const path = typeof args?.path === "string" ? args.path.replace(/^\[|[#\]].*$|:[\d+-].*$/g, "").split(/[\\/]/).pop() : undefined;
	const why = clean(intent, LIMITS.intent) || clean(path, 60);
	return why ? `${name}: ${why}` : name;
}

// The todo tool's own arguments are the agent's plan; mirror them by title.
function applyTodo(facts: Facts, args: Record<string, unknown>): void {
	const op = args.op;
	const titles = (value: unknown): string[] =>
		(Array.isArray(value) ? value : typeof value === "string" ? [value] : [])
			.map(v => clean(v, LIMITS.todoText))
			.filter(Boolean);
	const setStatus = (value: unknown, status: TodoItem["status"]) => {
		for (const title of titles(value)) {
			const item = facts.todos.find(t => t.text === title);
			if (item) item.status = status;
		}
	};
	if (op === "init" && Array.isArray(args.list)) {
		facts.todos = [];
		for (const phase of args.list as Array<{ items?: unknown }>) {
			for (const title of titles(phase?.items)) facts.todos.push({ text: title, status: "pending" });
		}
		if (facts.todos[0]) facts.todos[0].status = "in_progress";
	} else if (op === "init" && Array.isArray(args.items)) {
		facts.todos = titles(args.items).map(text => ({ text, status: "pending" as const }));
		if (facts.todos[0]) facts.todos[0].status = "in_progress";
	} else if (op === "append") {
		for (const text of titles(args.items)) facts.todos.push({ text, status: "pending" });
	} else if (op === "start") {
		for (const t of facts.todos) if (t.status === "in_progress") t.status = "pending";
		setStatus(args.task, "in_progress");
	} else if (op === "done") {
		setStatus(args.task, "done");
		if (!facts.todos.some(t => t.status === "in_progress")) {
			const next = facts.todos.find(t => t.status === "pending");
			if (next) next.status = "in_progress";
		}
	} else if (op === "drop" || op === "rm") {
		setStatus(args.task, "dropped");
	}
	if (facts.todos.length > 40) facts.todos.splice(0, facts.todos.length - 40);
}

export type ObservedEvent = { type: string; [key: string]: unknown };

const record = (value: unknown): Record<string, unknown> | undefined =>
	value && typeof value === "object" ? (value as Record<string, unknown>) : undefined;

/**
 * Fold one OMP extension event into the facts. Returns true when the event is
 * meaningful progress (worth a refresh), false when it was ignored.
 */
export function applyEvent(facts: Facts, event: ObservedEvent): boolean {
	switch (event.type) {
		case "before_agent_start": {
			const goal = clean(event.prompt, LIMITS.goal);
			if (!goal) return false;
			// A new prompt: the per-prompt record starts over, the todo list carries on.
			facts.previousGoal = facts.goal ? clean(facts.goal, LIMITS.previousGoal) : null;
			facts.goal = goal;
			facts.requests += 1;
			facts.toolCalls = 0;
			facts.failedCalls = 0;
			facts.recent = [];
			facts.failures = [];
			facts.agentNote = null;
			facts.running = true;
			facts.retry = null;
			return true;
		}
		case "agent_start":
			facts.running = true;
			return false;
		case "agent_end":
			facts.running = false;
			facts.underway.clear();
			return true;
		case "tool_execution_start": {
			const args = record(event.args);
			facts.underway.set(String(event.toolCallId), toolLabel(event.toolName, event.intent, args));
			if (event.toolName === "todo" && args) applyTodo(facts, args);
			return true;
		}
		case "tool_execution_end": {
			const id = String(event.toolCallId);
			const label = facts.underway.get(id) ?? toolLabel(event.toolName, undefined, undefined);
			facts.underway.delete(id);
			facts.toolCalls += 1;
			if (event.isError) {
				facts.failedCalls += 1;
				pushBounded(facts.failures, label, LIMITS.failures);
			}
			pushBounded(facts.recent, event.isError ? `${label} (failed)` : label, LIMITS.recent);
			return true;
		}
		case "message_end": {
			const message = record(event.message);
			if (!message || message.role !== "assistant") return false;
			if (message.stopReason === "error") {
				facts.retry = `model error: ${clean(message.errorMessage, 100) || "unknown"}`;
				return true;
			}
			const text = Array.isArray(message.content)
				? message.content
						.map(block => record(block))
						.filter(block => block?.type === "text" && typeof block.text === "string")
						.map(block => block?.text as string)
						.join(" ")
				: "";
			const flat = clean(text, 2000);
			const note = clean(/^(.+?[.!?])(\s|$)/.exec(flat)?.[1] ?? flat, LIMITS.note);
			if (!note) return false;
			facts.agentNote = note;
			return true;
		}
		case "auto_retry_start":
			facts.retry = `provider retry ${Number(event.attempt) || "?"}/${Number(event.maxAttempts) || "?"}: ${clean(event.errorMessage, 100)}`;
			return true;
		case "auto_retry_end":
			if (event.success) facts.retry = null;
			return true;
		default:
			return false;
	}
}

/** The bounded JSON document sent to the auxiliary model. */
export function buildPayload(facts: Facts): string {
	const live = facts.todos.filter(t => t.status !== "dropped");
	const doc = {
		current_prompt: {
			number_since_opt_in: facts.requests,
			request: facts.goal,
			turn: facts.running ? "agent working" : "turn ended (agent idle)",
			tool_calls: { finished: facts.toolCalls, failed: facts.failedCalls },
			running_now: [...facts.underway.values()].slice(-3),
			recent_actions: [...facts.recent],
			recent_failures: [...facts.failures],
			retry_or_error: facts.retry,
			latest_agent_message: facts.agentNote,
		},
		todo_list: live.length
			? {
					note: "the agent's own plan; it may span several prompts",
					done: live.filter(t => t.status === "done").length,
					total: live.length,
					items: facts.todos.slice(-LIMITS.todoItems).map(t => `[${t.status}] ${t.text}`),
				}
			: null,
		earlier_request: facts.previousGoal,
	};
	let json = JSON.stringify(doc);
	const recent = doc.current_prompt.recent_actions;
	const items = doc.todo_list?.items ?? [];
	while (json.length > LIMITS.payloadChars && (recent.length > 2 || items.length > 4)) {
		if (recent.length > 2) recent.shift();
		else items.shift();
		json = JSON.stringify(doc);
	}
	return json.length > LIMITS.payloadChars ? json.slice(0, LIMITS.payloadChars) : json;
}
