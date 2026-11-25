// Observed facts about one OMP session, built exclusively from local OMP
// extension events (and, on mid-session opt-in, the current branch history).
// Everything stays strictly local — nothing is sent to any external model.

export const LIMITS = {
	goal: 500,
	note: 160,
	intent: 90,
	todoText: 90,
	todoItems: 40,
	recent: 12,
	failures: 5,
	toolHistory: 200,
} as const;

/** A silent stretch longer than this (a sleeping laptop, an unattended prompt) counts only this long. */
export const IDLE_GAP_MS = 10 * 60_000;

export type TodoStatus = "pending" | "in_progress" | "done" | "blocked" | "dropped";

export interface TodoItem {
	text: string;
	status: TodoStatus;
	phase?: string;
}

export interface UnderwayTool {
	name: string;
	label: string;
	startedAtS: number;
	subtasks: number;
	todoArgs?: Record<string, unknown>;
}

export interface ToolRecord {
	name: string;
	isError: boolean;
	atS: number;
	durS: number;
}

export interface ToolCategoryCounts {
	explore: number;
	write: number;
	exec: number;
	delegate: number;
	todo: number;
	other: number;
}

export type WorkPhase =
	| "idle"
	| "watching"
	| "planning"
	| "exploring"
	| "editing"
	| "verifying"
	| "delegating"
	| "waiting"
	| "retrying"
	| "wrapping up"
	| "done";

export interface BackgroundJob {
	startedAtS: number;
	kind: "task" | "bash" | "other";
	isSlowModel: boolean;
}

export interface Facts {
	running: boolean;
	/** Completed or active user prompts seen in this session (0-based index; -1 before any prompt). */
	promptIndex: number;
	/** Requests seen since opt-in/branch start. */
	requests: number;
	/** Cleaned user request for display/inspection. */
	goal: string | null;
	/** Raw prompt length & structure (computed locally, never transmitted). */
	promptChars: number;
	promptListItems: number;
	promptKwHeavy: boolean;
	promptKwLight: boolean;
	promptKwContinue: boolean;
	promptIsQuestion: boolean;
	promptIsTinyCmd: boolean;
	promptHasCodeOrLog: boolean;
	promptFilePaths: number;
	promptKwE2eBench: boolean;
	promptKwCommitShip: boolean;

	/** Authoritative todo list (survives across prompts in the session). */
	todos: TodoItem[];
	/** The agent touched the todo list during this prompt. */
	todoUpdated: boolean;
	/** The agent initialized a fresh todo list during this prompt. */
	todoInitThisPrompt: boolean;
	/** Open (non-done, non-dropped) todos at the start of this prompt or right after init. */
	todoStartRem: number;
	/** Live todo count at initial observation in this prompt (tracks scope expansion). */
	todoInitTotal: number;
	/** Open todos carried over from earlier prompts before this prompt started. */
	carryoverOpenTodos: number;
	/** Active seconds when todos were initialized in this prompt (null if carried over). */
	todoInitAtS: number | null;
	/** Active-second timestamps of each todo completion during this prompt. */
	todoDoneEventsS: number[];
	/** Tool-call counts at each todo completion during this prompt. */
	todoDoneToolsAt: number[];
	todoPhasesTotal: number;
	todoPhasesDone: number;
	todoAppendCount: number;

	/** Per-prompt tool metrics. */
	toolCalls: number;
	failedCalls: number;
	errorStreak: number;
	underway: Map<string, UnderwayTool>;
	recent: string[];
	failures: string[];
	toolHistory: ToolRecord[];
	toolTimesS: number[];
	catCounts: ToolCategoryCounts;
	toolTimeSumS: number;
	taskTimeSumS: number;
	taskDoneCount: number;
	maxToolDurS: number;
	lastToolEndAtS: number;
	firstWriteAtS: number | null;
	lastWriteAtS: number | null;
	postWriteExecCount: number;
	postWriteCleanExecCount: number;
	subsequentToolsAfterLastWrite: number;
	lastExecClean: boolean;
	taskSubagentsLaunched: number;
	taskCompleted: number;
	taskDispatches: number;

	/** Async background jobs (async task/bash launches reconciled with wait snapshots). */
	bgJobs: Map<string, BackgroundJob>;
	bgDoneCount: number;
	bgDoneDurSumS: number;
	bgDoneMaxS: number;
	waitCount: number;
	waitTimeSumS: number;

	/** Tool intent & workflow stage progression (strictly causal). */
	intentTestCount: number;
	intentE2eCount: number;
	intentBuildCount: number;
	intentGitCount: number;
	intentReleaseCount: number;
	lastIntentTestAtS: number;
	lastIntentGitAtS: number;
	underwayIntentTest: boolean;
	underwayIntentE2e: boolean;
	underwayIntentGit: boolean;
	curStage: number;
	stageAtS: number;
	stageTransitions: number;
	editVerifyCycles: number;
	verificationsSeen: number;
	sawEditInCycle: boolean;
	lastErrorAtS: number;

	/** Working set & file dynamics (local paths only, never transmitted). */
	filesRead: Set<string>;
	filesEdited: Set<string>;
	editPathCounts: Map<string, number>;
	recentReadNovel: number[];
	editedTestFile: boolean;
	editedDocFile: boolean;

	/** Per-prompt LLM turn & token metrics (from message_end). */
	assistantTurns: number;
	turnTimesS: number[];
	totalOutputTokens: number;
	recentOutputTokens: number[];
	latestContextTokens: number;
	promptStartContextTokens: number;
	prevTurnContextTokens: number;
	compactions: number;
	lastActivityEndAtS: number;
	thinkTimeSumS: number;
	thinkCount: number;
	recentThinkS: number[];
	curTurnTools: number;
	turnFanoutHist: number[];
	model: string;
	agentNote: string | null;
	agentWrapUp: boolean;
	proseHasRemaining: boolean;
	proseHasDebug: boolean;
	retry: string | null;

	/** Active time of the current prompt up to `lastAt` (ms). */
	activeMs: number;
	/** Last completed prompt's active duration (ms), shown when turn ended. */
	lastCompletedMs: number | null;
	lastAt: number | null;
	prevPromptEndAtMs: number | null;
	gapSincePrevPromptS: number;
	/** Open `ask` calls: the agent is waiting on the user. */
	asking: Set<string>;
	askCount: number;

	/** Strictly causal history of earlier prompts completed in this session (seconds). */
	priorPromptDurationsS: number[];
	priorMaxDurS: number;
	sessionActiveS: number;
}

export const EXPLORE_TOOLS = new Set(["read", "grep", "glob", "find", "web_search", "fetch", "lsp", "search"]);
export const WRITE_TOOLS = new Set(["edit", "write", "notebook_edit", "multi_edit"]);
export const EXEC_TOOLS = new Set(["bash", "eval", "python", "run"]);
export const DELEGATE_TOOLS = new Set(["task", "wait", "hub"]);

export function emptyFacts(): Facts {
	return {
		running: false,
		promptIndex: -1,
		requests: 0,
		goal: null,
		promptChars: 0,
		promptListItems: 0,
		promptKwHeavy: false,
		promptKwLight: false,
		promptKwContinue: false,
		promptIsQuestion: false,
		promptIsTinyCmd: false,
		promptHasCodeOrLog: false,
		promptFilePaths: 0,
		promptKwE2eBench: false,
		promptKwCommitShip: false,
		todos: [],
		todoUpdated: false,
		todoInitThisPrompt: false,
		todoStartRem: 0,
		todoInitTotal: 0,
		carryoverOpenTodos: 0,
		todoInitAtS: null,
		todoDoneEventsS: [],
		todoDoneToolsAt: [],
		todoPhasesTotal: 0,
		todoPhasesDone: 0,
		todoAppendCount: 0,
		toolCalls: 0,
		failedCalls: 0,
		errorStreak: 0,
		underway: new Map(),
		recent: [],
		failures: [],
		toolHistory: [],
		toolTimesS: [],
		catCounts: { explore: 0, write: 0, exec: 0, delegate: 0, todo: 0, other: 0 },
		toolTimeSumS: 0,
		taskTimeSumS: 0,
		taskDoneCount: 0,
		maxToolDurS: 0,
		lastToolEndAtS: 0,
		firstWriteAtS: null,
		lastWriteAtS: null,
		postWriteExecCount: 0,
		postWriteCleanExecCount: 0,
		subsequentToolsAfterLastWrite: 0,
		lastExecClean: false,
		taskSubagentsLaunched: 0,
		taskCompleted: 0,
		taskDispatches: 0,
		bgJobs: new Map(),
		bgDoneCount: 0,
		bgDoneDurSumS: 0,
		bgDoneMaxS: 0,
		waitCount: 0,
		waitTimeSumS: 0,
		intentTestCount: 0,
		intentE2eCount: 0,
		intentBuildCount: 0,
		intentGitCount: 0,
		intentReleaseCount: 0,
		lastIntentTestAtS: -1,
		lastIntentGitAtS: -1,
		underwayIntentTest: false,
		underwayIntentE2e: false,
		underwayIntentGit: false,
		curStage: 0,
		stageAtS: 0,
		stageTransitions: 0,
		editVerifyCycles: 0,
		verificationsSeen: 0,
		sawEditInCycle: false,
		lastErrorAtS: -1,
		filesRead: new Set(),
		filesEdited: new Set(),
		editPathCounts: new Map(),
		recentReadNovel: [],
		editedTestFile: false,
		editedDocFile: false,
		assistantTurns: 0,
		turnTimesS: [],
		totalOutputTokens: 0,
		recentOutputTokens: [],
		latestContextTokens: 0,
		promptStartContextTokens: 0,
		prevTurnContextTokens: 0,
		compactions: 0,
		lastActivityEndAtS: 0,
		thinkTimeSumS: 0,
		thinkCount: 0,
		recentThinkS: [],
		curTurnTools: 0,
		turnFanoutHist: [],
		model: "",
		agentNote: null,
		agentWrapUp: false,
		proseHasRemaining: false,
		proseHasDebug: false,
		retry: null,
		activeMs: 0,
		lastCompletedMs: null,
		lastAt: null,
		prevPromptEndAtMs: null,
		gapSincePrevPromptS: 0,
		asking: new Set(),
		askCount: 0,
		priorPromptDurationsS: [],
		priorMaxDurS: 0,
		sessionActiveS: 0,
	};
}

// Linear-time secret patterns (no nested/quadratic backtracking!).
const SECRET_PATTERNS: RegExp[] = [
	/-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)/g,
	/\b(?:sk|rk|pk)-[A-Za-z0-9_-]{12,}/g,
	/\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}/g,
	/\bgithub_pat_[A-Za-z0-9_]{20,}/g,
	/\bglpat-[A-Za-z0-9_-]{16,}/g,
	/\bxox[abprs]-[A-Za-z0-9-]{10,}/g,
	/\bAKIA[0-9A-Z]{16}\b/g,
	/\bAIza[0-9A-Za-z_-]{30,}/g,
	/\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}/g,
	/\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}/gi,
	/\b(?:api[_-]?key|token|secret|password|passwd|pwd|credential)\s*[:=]\s*(?:"[^"]*"|'[^']*'|\S+)/gi,
	/[a-z][a-z0-9+.-]*:\/\/[^\s/:@]+:[^\s/@]+@/gi,
	/\b[0-9a-f]{32,}\b/gi,
	/\b[A-Za-z0-9+/_-]{40,}={0,2}\b/g,
];

/** Strip control/ANSI, code spans, and credential-looking tokens; collapse whitespace; cap length. Always O(n) bounded. */
export function clean(input: unknown, max: number): string {
	if (typeof input !== "string" || !input) return "";
	// Pre-slice before regex work so arbitrarily huge inputs (50 KB+) stay sub-millisecond.
	const preMax = Math.max(max * 4, 2048);
	let s = input.length > preMax ? input.slice(0, preMax) : input;
	s = s
		.replace(/```[\s\S]*?(?:```|$)/g, " [code] ")
		.replace(/`[^`\n]*`/g, "[code]")
		.replace(/\x1b\[[0-9;?]*[ -/]*[@-~]/g, "")
		.replace(/[\u0000-\u0008\u000b-\u001f\u007f-\u009f]/g, " ");
	for (const pattern of SECRET_PATTERNS) s = s.replace(pattern, "[redacted]");
	s = s.replace(/\s+/g, " ").trim();
	return s.length > max ? `${s.slice(0, Math.max(1, max - 1)).trimEnd()}…` : s;
}

/** Sanitize text for local terminal display: strip ANSI/control chars, collapse whitespace, cap length (keeps inline code readable). */
export function cleanDisplay(input: unknown, max: number): string {
	if (typeof input !== "string" || !input) return "";
	const preMax = Math.max(max * 4, 1024);
	let s = input.length > preMax ? input.slice(0, preMax) : input;
	s = s
		.replace(/\x1b\[[0-9;?]*[ -/]*[@-~]/g, "")
		.replace(/[\u0000-\u0008\u000b-\u001f\u007f-\u009f]/g, " ")
		.replace(/\s+/g, " ")
		.trim();
	return s.length > max ? `${s.slice(0, Math.max(1, max - 1)).trimEnd()}…` : s;
}

function pushBounded<T>(list: T[], value: T, max: number): void {
	list.push(value);
	if (list.length > max) list.splice(0, list.length - max);
}

function toolLabel(toolName: unknown, intent: unknown, args: Record<string, unknown> | undefined): string {
	const name = clean(toolName, 40) || "tool";
	const path =
		typeof args?.path === "string"
			? args.path
					.replace(/^\[|[#\]].*$|:[\d+-].*$/g, "")
					.split(/[\\/]/)
					.pop()
			: undefined;
	const why = clean(intent, LIMITS.intent) || clean(path, 60);
	return why ? `${name}: ${why}` : name;
}

const record = (value: unknown): Record<string, unknown> | undefined =>
	value && typeof value === "object" ? (value as Record<string, unknown>) : undefined;

function mapTodoStatus(raw: unknown): TodoStatus {
	if (raw === "completed" || raw === "done") return "done";
	if (raw === "in_progress") return "in_progress";
	if (raw === "blocked") return "blocked";
	if (raw === "abandoned" || raw === "dropped") return "dropped";
	return "pending";
}

function fuzzyFindTodo(todos: TodoItem[], query: string): TodoItem | undefined {
	const exact = todos.find(t => t.text === query);
	if (exact) return exact;
	const q = query.toLowerCase();
	return todos.find(t => t.text.toLowerCase().includes(q) || q.includes(t.text.toLowerCase()));
}

/** Fallback todo updater from tool_execution_start args when authoritative details.phases is unavailable. */
function applyTodoArgs(facts: Facts, args: Record<string, unknown>, curS: number): void {
	const op = args.op;
	const titles = (value: unknown): string[] =>
		(Array.isArray(value) ? value : typeof value === "string" ? [value] : [])
			.map(v => cleanDisplay(v, LIMITS.todoText))
			.filter(Boolean);

	const setStatus = (value: unknown, status: TodoStatus) => {
		const list = titles(value);
		if (list.length === 0 && status === "done") {
			const active = facts.todos.find(t => t.status === "in_progress");
			if (active) active.status = "done";
			return;
		}
		for (const title of list) {
			if (title.toLowerCase() === "all") {
				for (const t of facts.todos) if (t.status !== "dropped") t.status = status;
				continue;
			}
			const item = fuzzyFindTodo(facts.todos, title);
			if (item) item.status = status;
		}
	};

	const prevDone = facts.todos.filter(t => t.status === "done").length;

	if (op === "init" && Array.isArray(args.list)) {
		facts.todos = [];
		let phases = 0;
		for (const phase of args.list as Array<{ phase?: unknown; items?: unknown }>) {
			const phaseName = clean(phase?.phase, 40) || undefined;
			const items = titles(phase?.items);
			if (items.length) phases++;
			for (const title of items) facts.todos.push({ text: title, status: "pending", phase: phaseName });
		}
		if (facts.todos[0]) facts.todos[0].status = "in_progress";
		facts.todoPhasesTotal = phases;
		facts.todoPhasesDone = 0;
		facts.todoStartRem = facts.todos.length;
		facts.todoInitThisPrompt = true;
		facts.todoInitAtS = curS;
		facts.todoDoneEventsS = [];
		facts.todoDoneToolsAt = [];
	} else if (op === "init" && Array.isArray(args.items)) {
		facts.todos = titles(args.items).map(text => ({ text, status: "pending" as const }));
		if (facts.todos[0]) facts.todos[0].status = "in_progress";
		facts.todoPhasesTotal = facts.todos.length ? 1 : 0;
		facts.todoPhasesDone = 0;
		facts.todoStartRem = facts.todos.length;
		facts.todoInitThisPrompt = true;
		facts.todoInitAtS = curS;
		facts.todoDoneEventsS = [];
		facts.todoDoneToolsAt = [];
	} else if (op === "append") {
		const added = titles(args.items);
		for (const text of added) facts.todos.push({ text, status: "pending" });
		facts.todoStartRem += added.length;
		facts.todoAppendCount += 1;
	} else if (op === "start") {
		for (const t of facts.todos) if (t.status === "in_progress") t.status = "pending";
		setStatus(args.task, "in_progress");
	} else if (op === "done") {
		setStatus(args.task, "done");
		if (!facts.todos.some(t => t.status === "in_progress")) {
			const next = facts.todos.find(t => t.status === "pending");
			if (next) next.status = "in_progress";
		}
	} else if (op === "block") {
		setStatus(args.task, "blocked");
	} else if (op === "unblock") {
		setStatus(args.task, "pending");
	} else if (op === "drop" || op === "rm") {
		setStatus(args.task, "dropped");
	}

	if (op !== "init") {
		const newDone = facts.todos.filter(t => t.status === "done").length;
		if (newDone > prevDone) {
			for (let k = prevDone; k < newDone; k++) {
				facts.todoDoneEventsS.push(curS);
				facts.todoDoneToolsAt.push(facts.toolCalls);
			}
		}
	}
	if (facts.todos.length > LIMITS.todoItems) facts.todos.splice(0, facts.todos.length - LIMITS.todoItems);
}

/** Authoritative todo state from OMP's tool_execution_end `result.details.phases`. */
function applyTodoPhases(facts: Facts, details: Record<string, unknown>, curS: number): void {
	const phases = details.phases;
	if (!Array.isArray(phases)) return;
	const prevDone = facts.todos.filter(t => t.status === "done").length;
	const prevLive = facts.todos.filter(t => t.status !== "dropped").length;
	const nextTodos: TodoItem[] = [];
	let phasesTotal = 0;
	let phasesDone = 0;
	for (const ph of phases as Array<{ name?: unknown; tasks?: Array<{ content?: unknown; status?: unknown }> }>) {
		const phaseName = clean(ph?.name, 40) || undefined;
		const tasks = Array.isArray(ph?.tasks) ? ph.tasks : [];
		const livePhTasks = tasks.filter(t => t.status !== "abandoned" && t.status !== "dropped");
		if (livePhTasks.length > 0) {
			phasesTotal++;
			if (livePhTasks.every(t => t.status === "completed" || t.status === "done")) phasesDone++;
		}
		for (const t of tasks) {
			const text = cleanDisplay(t?.content, LIMITS.todoText);
			if (!text) continue;
			nextTodos.push({ text, status: mapTodoStatus(t?.status), phase: phaseName });
		}
	}
	const newDone = nextTodos.filter(t => t.status === "done").length;
	const newLive = nextTodos.filter(t => t.status !== "dropped").length;
	const op = details.op;
	if (op === "init") {
		facts.todoStartRem = Math.max(0, newLive - newDone);
		facts.todoInitThisPrompt = true;
		if (facts.todoInitAtS === null) facts.todoInitAtS = curS;
		facts.todoDoneEventsS = [];
		facts.todoDoneToolsAt = [];
	} else {
		if (op === "append") {
			facts.todoStartRem += Math.max(0, newLive - prevLive);
		}
		if (newDone > prevDone) {
			for (let k = prevDone; k < newDone; k++) {
				facts.todoDoneEventsS.push(curS);
				facts.todoDoneToolsAt.push(facts.toolCalls);
			}
		}
	}
	facts.todos = nextTodos.slice(-LIMITS.todoItems);
	facts.todoPhasesTotal = phasesTotal;
	facts.todoPhasesDone = phasesDone;
	if (op !== "view") facts.todoUpdated = true;
}

export type ObservedEvent = { type: string; [key: string]: unknown };

/** Active time of the current prompt at `now` (ms). */
export function activeMs(facts: Facts, now: number): number {
	if (facts.lastAt === null || !facts.running || facts.asking.size) return facts.activeMs;
	return facts.activeMs + Math.min(Math.max(0, now - facts.lastAt), IDLE_GAP_MS);
}

const KW_HEAVY = /\b(implement|build|refactor|migrate|redesign|audit|overhaul|architecture|comprehensive|full|entire|all phases|plan)\b/i;
const KW_LIGHT = /\b(fix|typo|rename|check|why|what|explain|status|commit|push|diff|quick|small|just)\b/i;
const KW_CONTINUE = /\b(continue|proceed|go ahead|yes|do it|looks good|lgtm|approved|next)\b/i;
const KW_WRAPUP = /\b(all tests pass|verified|completed|everything passes|ready to|summary|done\b|no errors|build succeeds)/i;
const RE_INTENT_TEST = /\b(test|tests|spec|vitest|jest|pytest|verify|verif\w*|check|lint|typecheck|tsc|cargo\s+test|bun\s+test)\b/i;
const RE_INTENT_E2E = /\b(e2e|end[- ]to[- ]end|playwright|browser|smoke|benchmark|eval|train|experiment|emulator|tui)\b/i;
const RE_INTENT_BUILD = /\b(build|compile|bundle|webpack|vite|cargo\s+build|docker|install|migrate)\b/i;
const RE_INTENT_GIT = /\b(git\s+(status|diff|add|commit|push|log|pr)|commit|committing|pushing|pull request|gh\s+pr)\b/i;
const RE_INTENT_RELEASE = /\b(deploy|publish|release|push)\b/i;
const RE_TEST_PATH = /(?:^|\/)(?:tests?|__tests__|specs?)\/|\.(?:test|spec)\.[a-z]+$|_test\.[a-z]+$/i;
const RE_DOC_PATH = /\.(?:md|mdx|rst|txt)$|readme|changelog/i;
const RE_PROSE_REM = /\b(next i|now let|still need|remaining|left to|moving on to|let me also|after that)\b/i;
const RE_PROSE_DEBUG = /\b(failed|failing|error|bug|mismatch|unexpected|investigat|let me fix|let's fix|broken)\b/i;

function recordEditedFile(facts: Facts, path: string): void {
	if (!path) return;
	facts.filesEdited.add(path);
	facts.editPathCounts.set(path, (facts.editPathCounts.get(path) ?? 0) + 1);
	if (RE_TEST_PATH.test(path)) facts.editedTestFile = true;
	if (RE_DOC_PATH.test(path)) facts.editedDocFile = true;
}

function startPrompt(facts: Facts, rawPrompt: string, at: number): boolean {
	const goal = clean(rawPrompt, LIMITS.goal);
	if (!goal) return false;
	// Record superseded prompt duration only if it hadn't already ended via agent_end.
	if (facts.running && facts.promptIndex >= 0 && facts.activeMs > 0) {
		const durS = facts.activeMs / 1000;
		facts.priorPromptDurationsS.push(durS);
		facts.sessionActiveS += durS;
		if (durS > facts.priorMaxDurS) facts.priorMaxDurS = durS;
	} else if (facts.promptIndex >= 0) {
		const lastDur = facts.priorPromptDurationsS[facts.priorPromptDurationsS.length - 1] ?? 0;
		if (lastDur > facts.priorMaxDurS) facts.priorMaxDurS = lastDur;
	}
	facts.carryoverOpenTodos = facts.todos.filter(t => t.status !== "dropped" && t.status !== "done").length;
	facts.gapSincePrevPromptS =
		facts.prevPromptEndAtMs !== null ? Math.min(3600, Math.max(0, (at - facts.prevPromptEndAtMs) / 1000)) : 0;

	facts.promptIndex += 1;
	facts.requests += 1;
	facts.goal = goal;
	facts.promptChars = rawPrompt.length;
	facts.promptListItems = (rawPrompt.slice(0, 4000).match(/^\s*(?:\d+[.)]|[-*•])\s+/gm) ?? []).length;
	const trimmed = rawPrompt.trim();
	const head = trimmed.slice(0, 2500);
	facts.promptKwHeavy = KW_HEAVY.test(head.slice(0, 2000));
	facts.promptKwLight = KW_LIGHT.test(head.slice(0, 2000));
	facts.promptKwContinue = KW_CONTINUE.test(head.slice(0, 2000)) && rawPrompt.length < 180;
	facts.promptIsQuestion =
		/\?\s*$/.test(head.slice(0, 400)) ||
		/^(what|why|how|where|when|does|do|is|are|can|could|should|explain|tell me)\b/i.test(head);
	facts.promptIsTinyCmd =
		trimmed.length < 40 &&
		/^(commit|push|status|diff|yes|no|ok|go|run|test|retry|continue|ship|merge|rebase|clean)\b/i.test(trimmed);
	facts.promptHasCodeOrLog = /```|^\s*at\s+\S+\s*\(|Error:|Traceback|FAIL\b|Exception/m.test(head);
	facts.promptFilePaths = (head.match(/(?:[\w.-]+\/)+[\w.-]+\.[a-z]{1,6}\b/gi) ?? []).length;
	facts.promptKwE2eBench = /\b(e2e|end[- ]to[- ]end|benchmark|experiment|evaluate|train|browser|playwright|visual|audit|ratchet)\b/i.test(head);
	facts.promptKwCommitShip = /\b(commit|push|pr\b|pull request|release|deploy|merge|changelog)\b/i.test(head);

	facts.running = true;
	facts.activeMs = 0;
	facts.lastCompletedMs = null;
	facts.asking.clear();
	facts.askCount = 0;

	facts.todoUpdated = false;
	facts.todoInitThisPrompt = false;
	facts.todoStartRem = facts.todos.filter(t => t.status !== "dropped" && t.status !== "done").length;
	facts.todoInitTotal = 0;
	facts.todoInitAtS = null;
	facts.todoDoneEventsS = [];
	facts.todoDoneToolsAt = [];
	facts.todoAppendCount = 0;

	facts.toolCalls = 0;
	facts.failedCalls = 0;
	facts.errorStreak = 0;
	facts.underway.clear();
	facts.recent = [];
	facts.failures = [];
	facts.toolHistory = [];
	facts.toolTimesS = [];
	facts.catCounts = { explore: 0, write: 0, exec: 0, delegate: 0, todo: 0, other: 0 };
	facts.toolTimeSumS = 0;
	facts.taskTimeSumS = 0;
	facts.taskDoneCount = 0;
	facts.maxToolDurS = 0;
	facts.lastToolEndAtS = 0;
	facts.firstWriteAtS = null;
	facts.lastWriteAtS = null;
	facts.postWriteExecCount = 0;
	facts.postWriteCleanExecCount = 0;
	facts.subsequentToolsAfterLastWrite = 0;
	facts.lastExecClean = false;
	facts.taskSubagentsLaunched = 0;
	facts.taskCompleted = 0;
	facts.taskDispatches = 0;

	facts.bgJobs.clear();
	facts.bgDoneCount = 0;
	facts.bgDoneDurSumS = 0;
	facts.bgDoneMaxS = 0;
	facts.waitCount = 0;
	facts.waitTimeSumS = 0;

	facts.intentTestCount = 0;
	facts.intentE2eCount = 0;
	facts.intentBuildCount = 0;
	facts.intentGitCount = 0;
	facts.intentReleaseCount = 0;
	facts.lastIntentTestAtS = -1;
	facts.lastIntentGitAtS = -1;
	facts.underwayIntentTest = false;
	facts.underwayIntentE2e = false;
	facts.underwayIntentGit = false;
	facts.curStage = 0;
	facts.stageAtS = 0;
	facts.stageTransitions = 0;
	facts.editVerifyCycles = 0;
	facts.verificationsSeen = 0;
	facts.sawEditInCycle = false;
	facts.lastErrorAtS = -1;

	facts.filesRead.clear();
	facts.filesEdited.clear();
	facts.editPathCounts.clear();
	facts.recentReadNovel = [];
	facts.editedTestFile = false;
	facts.editedDocFile = false;

	facts.assistantTurns = 0;
	facts.turnTimesS = [];
	facts.totalOutputTokens = 0;
	facts.recentOutputTokens = [];
	facts.promptStartContextTokens = facts.latestContextTokens;
	facts.prevTurnContextTokens = facts.latestContextTokens;
	facts.compactions = 0;
	facts.lastActivityEndAtS = 0;
	facts.thinkTimeSumS = 0;
	facts.thinkCount = 0;
	facts.recentThinkS = [];
	facts.curTurnTools = 0;
	facts.turnFanoutHist = [];
	facts.agentNote = null;
	facts.agentWrapUp = false;
	facts.proseHasRemaining = false;
	facts.proseHasDebug = false;
	facts.retry = null;
	return true;
}

/**
 * Fold one OMP extension event, observed at `at` (ms), into `facts`.
 * Returns true when the event carries meaningful state change.
 */
export function applyEvent(facts: Facts, event: ObservedEvent, at: number): boolean {
	facts.activeMs = activeMs(facts, at);
	facts.lastAt = at;
	const curS = facts.activeMs / 1000;

	switch (event.type) {
		case "before_agent_start": {
			if (typeof event.prompt !== "string") return false;
			return startPrompt(facts, event.prompt, at);
		}
		case "agent_start":
			facts.running = true;
			return false;
		case "agent_end": {
			if (facts.running && facts.activeMs > 0) {
				facts.lastCompletedMs = facts.activeMs;
				const durS = facts.activeMs / 1000;
				facts.priorPromptDurationsS.push(durS);
				facts.sessionActiveS += durS;
				if (durS > facts.priorMaxDurS) facts.priorMaxDurS = durS;
			}
			facts.prevPromptEndAtMs = at;
			facts.running = false;
			facts.underway.clear();
			facts.asking.clear();
			facts.bgJobs.clear();
			return true;
		}
		case "tool_execution_start": {
			const args = record(event.args);
			const name = clean(event.toolName, 40) || "tool";
			const rawIntent = typeof event.intent === "string" ? event.intent : "";
			const label = toolLabel(event.toolName, event.intent, args);
			const subtasks = name === "task" && Array.isArray(args?.tasks) ? args.tasks.length : name === "task" ? 1 : 0;
			facts.running = true;
			facts.curTurnTools += 1;
			facts.toolTimesS.push(curS);
			// Do not mutate todos on tool_execution_start: wait for confirmed successful tool_execution_end.
			facts.underway.set(String(event.toolCallId), {
				name,
				label,
				startedAtS: curS,
				subtasks,
				todoArgs: name === "todo" ? args : undefined,
			});
			if (name === "ask") {
				facts.asking.add(String(event.toolCallId));
				facts.askCount += 1;
			}
			if (name === "task") {
				facts.taskSubagentsLaunched += subtasks;
				facts.taskDispatches += 1;
			}

			// Intent semantics & workflow stage progression
			const isTest = RE_INTENT_TEST.test(rawIntent);
			const isE2e = RE_INTENT_E2E.test(rawIntent);
			const isBuild = RE_INTENT_BUILD.test(rawIntent);
			const isGit = RE_INTENT_GIT.test(rawIntent);
			const isRelease = RE_INTENT_RELEASE.test(rawIntent);
			if (isTest) {
				facts.intentTestCount += 1;
				facts.lastIntentTestAtS = curS;
				facts.underwayIntentTest = true;
			}
			if (isE2e) {
				facts.intentE2eCount += 1;
				facts.underwayIntentE2e = true;
			}
			if (isBuild) facts.intentBuildCount += 1;
			if (isGit) {
				facts.intentGitCount += 1;
				facts.lastIntentGitAtS = curS;
				facts.underwayIntentGit = true;
			}
			if (isRelease) facts.intentReleaseCount += 1;

			let stage = 0;
			if (WRITE_TOOLS.has(name)) {
				stage = 2;
				facts.sawEditInCycle = true;
				facts.subsequentToolsAfterLastWrite = 0;
			} else if (DELEGATE_TOOLS.has(name)) {
				stage = 4;
			} else if (EXEC_TOOLS.has(name)) {
				if (isGit || isRelease) {
					stage = 5;
				} else if (isTest || isE2e || isBuild || facts.firstWriteAtS !== null) {
					stage = 3;
					facts.verificationsSeen += 1;
					if (facts.sawEditInCycle) {
						facts.editVerifyCycles += 1;
						facts.sawEditInCycle = false;
					}
				} else {
					stage = 1;
				}
			} else if (EXPLORE_TOOLS.has(name)) {
				stage = 1;
			}
			if (!WRITE_TOOLS.has(name) && facts.lastWriteAtS !== null) {
				facts.subsequentToolsAfterLastWrite += 1;
			}
			if (stage > 0 && stage !== facts.curStage) {
				facts.curStage = stage;
				facts.stageAtS = curS;
				facts.stageTransitions += 1;
			}

			// Working set file tracking from args.path
			if (EXPLORE_TOOLS.has(name) && typeof args?.path === "string" && args.path) {
				const isNew = !facts.filesRead.has(args.path);
				facts.filesRead.add(args.path);
				pushBounded(facts.recentReadNovel, isNew ? 1 : 0, 8);
			}
			if (WRITE_TOOLS.has(name) && typeof args?.path === "string" && args.path) {
				recordEditedFile(facts, args.path);
			}
			return true;
		}
		case "tool_execution_end": {
			const id = String(event.toolCallId);
			const started = facts.underway.get(id);
			const name = started?.name ?? (clean(event.toolName, 40) || "tool");
			const label = started?.label ?? toolLabel(event.toolName, undefined, undefined);
			const durS = started ? Math.max(0, curS - started.startedAtS) : 0;
			facts.underway.delete(id);
			facts.asking.delete(id);
			facts.toolCalls += 1;
			facts.toolTimeSumS += durS;
			facts.lastToolEndAtS = curS;
			facts.lastActivityEndAtS = curS;
			if (durS > facts.maxToolDurS) facts.maxToolDurS = durS;
			const isError = event.isError === true;
			if (isError) {
				facts.failedCalls += 1;
				facts.errorStreak += 1;
				facts.lastErrorAtS = curS;
				pushBounded(facts.failures, label, LIMITS.failures);
			} else {
				facts.errorStreak = 0;
			}
			pushBounded(facts.recent, isError ? `${label} (failed)` : label, LIMITS.recent);
			pushBounded(facts.toolHistory, { name, isError, atS: curS, durS }, LIMITS.toolHistory);

			// Refresh underway intent flags after removing completed call
			facts.underwayIntentTest = false;
			facts.underwayIntentE2e = false;
			facts.underwayIntentGit = false;
			for (const u of facts.underway.values()) {
				if (RE_INTENT_TEST.test(u.label)) facts.underwayIntentTest = true;
				if (RE_INTENT_E2E.test(u.label)) facts.underwayIntentE2e = true;
				if (RE_INTENT_GIT.test(u.label)) facts.underwayIntentGit = true;
			}

			if (EXPLORE_TOOLS.has(name)) facts.catCounts.explore += 1;
			else if (WRITE_TOOLS.has(name)) {
				facts.catCounts.write += 1;
				if (facts.firstWriteAtS === null) facts.firstWriteAtS = curS;
				facts.lastWriteAtS = curS;
				facts.postWriteExecCount = 0;
				facts.postWriteCleanExecCount = 0;
			} else if (EXEC_TOOLS.has(name)) {
				facts.catCounts.exec += 1;
				facts.lastExecClean = !isError;
				if (facts.lastWriteAtS !== null) {
					facts.postWriteExecCount += 1;
					if (!isError) facts.postWriteCleanExecCount += 1;
				}
			} else if (DELEGATE_TOOLS.has(name)) {
				facts.catCounts.delegate += 1;
				if (name === "task") {
					facts.taskCompleted += started?.subtasks ?? 1;
					facts.taskTimeSumS += durS;
					facts.taskDoneCount += 1;
				}
			} else if (name === "todo") {
				facts.catCounts.todo += 1;
			} else {
				facts.catCounts.other += 1;
			}

			const result = record(event.result);
			const details = record(result?.details) ?? record(event.details);

			// Edited file paths from authoritative edit/write tool result details
			if (WRITE_TOOLS.has(name) && details) {
				if (typeof details.path === "string") recordEditedFile(facts, details.path);
				if (typeof details.resolvedPath === "string") recordEditedFile(facts, details.resolvedPath);
				const perFile = record(details.perFileResults);
				if (perFile) {
					for (const p of Object.keys(perFile)) recordEditedFile(facts, p);
				}
			}

			// Async background job tracking (task/bash launches reconciled with wait job snapshots)
			if (name === "task" && details && details.async && Array.isArray(details.progress)) {
				for (const item of details.progress as Array<Record<string, unknown>>) {
					const jid = String(item?.id ?? `${id}:${item?.index ?? 0}`);
					const isSlowModel =
						item?.modelRole === "slow" || /opus|sol|pro|reviewer|planner/i.test(String(item?.agent ?? ""));
					facts.bgJobs.set(jid, { startedAtS: curS, kind: "task", isSlowModel });
				}
			} else if (name === "bash" && details) {
				const asyncObj = record(details.async);
				if (asyncObj && asyncObj.state === "running" && asyncObj.jobId) {
					facts.bgJobs.set(String(asyncObj.jobId), { startedAtS: curS, kind: "bash", isSlowModel: false });
				}
			} else if (name === "wait") {
				facts.waitCount += 1;
				facts.waitTimeSumS += durS;
				if (details && Array.isArray(details.jobs)) {
					for (const job of details.jobs as Array<Record<string, unknown>>) {
						const jid = String(job?.id ?? "");
						const st = String(job?.status ?? "");
						if (st === "running" || st === "pending") {
							if (jid && !facts.bgJobs.has(jid)) {
								const jDurS = typeof job?.durationMs === "number" ? job.durationMs / 1000 : 0;
								facts.bgJobs.set(jid, {
									startedAtS: Math.max(0, curS - jDurS),
									kind: job?.type === "task" ? "task" : "bash",
									isSlowModel: false,
								});
							}
						} else if (st === "completed" || st === "failed" || st === "cancelled") {
							if (jid) facts.bgJobs.delete(jid);
							facts.bgDoneCount += 1;
							const jDurS = typeof job?.durationMs === "number" ? Math.min(3600, job.durationMs / 1000) : 0;
							facts.bgDoneDurSumS += jDurS;
							if (jDurS > facts.bgDoneMaxS) facts.bgDoneMaxS = jDurS;
						}
					}
				}
			}

			// Count confirmed todo milestones only when the todo tool succeeded.
			if (name === "todo" && !isError) {
				if (details && Array.isArray(details.phases)) {
					applyTodoPhases(facts, details, curS);
				} else if (started?.todoArgs) {
					if (started.todoArgs.op !== "view") facts.todoUpdated = true;
					applyTodoArgs(facts, started.todoArgs, curS);
				}
				const liveCount = facts.todos.filter(t => t.status !== "dropped").length;
				if (facts.todoInitTotal === 0 && liveCount > 0) facts.todoInitTotal = liveCount;
			}
			return true;
		}
		case "message_end": {
			const message = record(event.message);
			if (!message || message.role !== "assistant") return false;
			facts.assistantTurns += 1;
			facts.turnTimesS.push(curS);
			if (typeof message.model === "string" && message.model) facts.model = message.model;
			const usage = record(message.usage);
			if (usage) {
				if (typeof usage.output === "number") {
					facts.totalOutputTokens += usage.output;
					pushBounded(facts.recentOutputTokens, usage.output, 5);
				}
				const ctx = (Number(usage.input) || 0) + (Number(usage.cacheRead) || 0) + (Number(usage.cacheWrite) || 0);
				if (ctx > 0) {
					if (facts.prevTurnContextTokens > 30_000 && ctx < facts.prevTurnContextTokens * 0.65) {
						facts.compactions += 1;
					}
					facts.latestContextTokens = ctx;
					facts.prevTurnContextTokens = ctx;
				}
			}
			const thinkS = Math.max(0, Math.min(300, curS - facts.lastActivityEndAtS));
			if (thinkS > 0.2) {
				facts.thinkTimeSumS += thinkS;
				facts.thinkCount += 1;
				pushBounded(facts.recentThinkS, thinkS, 5);
			}
			if (facts.curTurnTools > 0) {
				pushBounded(facts.turnFanoutHist, facts.curTurnTools, 20);
				facts.curTurnTools = 0;
			}
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
			if (text) {
				const head = text.slice(0, 1500);
				facts.agentWrapUp = KW_WRAPUP.test(text.slice(0, 2000));
				facts.proseHasRemaining = RE_PROSE_REM.test(head);
				facts.proseHasDebug = RE_PROSE_DEBUG.test(head);
				const flat = clean(text, 600);
				const note = clean(/^(.+?[.!?])(?:\s|$)/.exec(flat)?.[1] ?? flat, LIMITS.note);
				if (note) facts.agentNote = note;
			}
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

export interface BranchEntry {
	type: string;
	timestamp?: string | number;
	customType?: string;
	data?: Record<string, unknown>;
	message?: {
		role?: string;
		content?: unknown;
		stopReason?: string;
		errorMessage?: string;
		toolCallId?: string;
		toolName?: string;
		isError?: boolean;
		model?: string;
		usage?: Record<string, unknown>;
		details?: Record<string, unknown>;
	};
}

const parseTime = (value: unknown): number =>
	typeof value === "number" ? value : typeof value === "string" ? Date.parse(value) : Number.NaN;

const textFromContent = (content: unknown): string =>
	typeof content === "string"
		? content
		: Array.isArray(content)
			? content
					.filter((b): b is { type: "text"; text: string } => Boolean(b && typeof b === "object" && (b as { type?: unknown }).type === "text" && typeof (b as { text?: unknown }).text === "string"))
					.map(b => b.text)
					.join("\n")
			: "";

/**
 * Replay an OMP session branch (`ctx.sessionManager.getBranch()`) into `facts`,
 * so turning `/progress on` mid-session or mid-prompt recovers exact elapsed time,
 * todo list velocity, tool counts, and prior prompt durations immediately.
 */
export function seedFromBranch(facts: Facts, branch: BranchEntry[], now: number, isIdle: boolean): void {
	const toolStartAt = new Map<string, number>();
	for (const e of branch) {
		if (e.type === "custom" && e.customType === "tool_execution_start" && e.data) {
			const t = parseTime(e.data.startedAt);
			if (Number.isFinite(t)) toolStartAt.set(String(e.data.toolCallId), t);
		}
	}
	const messages = branch.filter(e => e.type === "message" && e.message && Number.isFinite(parseTime(e.timestamp)));
	const events: Array<{ t: number; event: ObservedEvent }> = [];
	for (let i = 0; i < messages.length; i++) {
		const e = messages[i]!;
		const t = parseTime(e.timestamp);
		const m = e.message!;
		if (m.role === "user") {
			const text = textFromContent(m.content);
			if (text.trim()) events.push({ t, event: { type: "before_agent_start", prompt: text } });
		} else if (m.role === "assistant") {
			const blocks = Array.isArray(m.content) ? (m.content as Array<Record<string, unknown>>) : [];
			const text = textFromContent(blocks);
			const calls = blocks.filter(b => b.type === "toolCall");
			events.push({
				t,
				event: {
					type: "message_end",
					message: {
						role: "assistant",
						stopReason: m.stopReason,
						errorMessage: m.errorMessage,
						model: m.model,
						usage: m.usage,
						content: text.trim() ? [{ type: "text", text }] : [],
					},
				},
			});
			if (m.stopReason !== "aborted") {
				for (const call of calls) {
					const id = String(call.id);
					const name = typeof call.name === "string" ? call.name : "tool";
					const args = call.arguments && typeof call.arguments === "object" ? (call.arguments as Record<string, unknown>) : {};
					events.push({
						t: Math.max(t, toolStartAt.get(id) ?? t),
						event: { type: "tool_execution_start", toolCallId: id, toolName: name, intent: call.intent, args },
					});
				}
			}
			const next = messages[i + 1]?.message?.role;
			const retried = m.stopReason === "error" && next === "assistant";
			if ((calls.length === 0 || m.stopReason === "aborted") && ["stop", "aborted", "error"].includes(m.stopReason ?? "") && !retried) {
				events.push({ t: t + 1, event: { type: "agent_end", stopReason: m.stopReason } });
			}
		} else if (m.role === "toolResult") {
			events.push({
				t,
				event: {
					type: "tool_execution_end",
					toolCallId: String(m.toolCallId),
					toolName: m.toolName,
					isError: m.isError === true,
					details: m.details,
				},
			});
		}
	}
	events.sort((a, b) => a.t - b.t);
	for (const { t, event } of events) applyEvent(facts, event, t);
	if (isIdle && facts.running) applyEvent(facts, { type: "agent_end" }, now);
	else if (!isIdle && facts.requests > 0) facts.running = true;
}

/** Deterministic workflow phase inferred directly from observed OMP events. */
export function inferPhase(facts: Facts): WorkPhase {
	if (facts.requests === 0 && !facts.running) return "watching";
	if (!facts.running) return facts.requests > 0 ? "done" : "idle";
	if (facts.asking.size > 0) return "waiting";
	if (facts.retry) return "retrying";

	const liveTodos = facts.todos.filter(t => t.status !== "dropped");
	if (facts.todoUpdated && liveTodos.length > 0 && liveTodos.every(t => t.status === "done")) return "wrapping up";

	// Check underway tools first
	const underwayNames = [...facts.underway.values()].map(u => u.name);
	if (underwayNames.some(n => DELEGATE_TOOLS.has(n))) return "delegating";
	if (underwayNames.some(n => WRITE_TOOLS.has(n))) return "editing";
	if (underwayNames.some(n => EXEC_TOOLS.has(n))) return facts.firstWriteAtS !== null ? "verifying" : "exploring";
	if (underwayNames.some(n => n === "todo")) return "planning";
	if (underwayNames.some(n => EXPLORE_TOOLS.has(n))) return "exploring";

	if (facts.agentWrapUp && (facts.postWriteCleanExecCount > 0 || facts.toolCalls >= 3)) return "wrapping up";

	// Otherwise look at recent tool activity
	const rec = facts.toolHistory.slice(-3);
	if (rec.length === 0) return "planning";
	const last = rec[rec.length - 1]!.name;
	if (DELEGATE_TOOLS.has(last)) return "delegating";
	if (WRITE_TOOLS.has(last)) return "editing";
	if (EXEC_TOOLS.has(last)) return facts.firstWriteAtS !== null ? "verifying" : "exploring";
	if (EXPLORE_TOOLS.has(last)) return "exploring";
	return facts.firstWriteAtS !== null ? "editing" : "exploring";
}

/** Deterministic blocker extracted from observed errors, retries, or blocked todos (never hallucinated). */
export function inferBlocker(facts: Facts): string | null {
	if (facts.retry) return facts.retry;
	const blocked = facts.todos.find(t => t.status === "blocked");
	if (blocked && facts.todoUpdated) return `blocked todo: ${blocked.text}`;
	if (facts.errorStreak >= 2 && facts.failures.length > 0) {
		return `${facts.errorStreak} errors in a row (last: ${facts.failures[facts.failures.length - 1]})`;
	}
	return null;
}

/** Deterministic next step from the agent's real todo list (only when active in this prompt). */
export function inferNextHint(facts: Facts): string | null {
	if (!facts.running || !facts.todoUpdated) return null;
	const live = facts.todos.filter(t => t.status !== "dropped");
	if (live.length === 0) return null;
	const nextPending = live.find(t => t.status === "pending");
	if (nextPending) return nextPending.text;
	const inProg = live.find(t => t.status === "in_progress");
	if (inProg) return `finish: ${inProg.text}`;
	if (live.every(t => t.status === "done")) return "final reply";
	return null;
}
