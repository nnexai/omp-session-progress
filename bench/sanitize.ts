// Privacy boundary of the replay benchmark. Historic session text never
// reaches the estimator model: every free-text field is replaced by a
// description built from a closed vocabulary (counts, flags, topic words,
// known tool names). `assertSanitized` checks a payload against that
// vocabulary, so a leak fails loudly instead of reaching the provider.

/** Topic words, each a common English word, matched on stems of the original text. */
const TOPICS: Array<[string, RegExp]> = [
	["fix", /\b(fix|bug|broken|repair|regression|crash)/i],
	["test", /\btest/i],
	["build", /\b(build|compil|bundl)/i],
	["refactor", /\b(refactor|clean ?up|simplif|dedup|rename)/i],
	["review", /\b(review|audit)/i],
	["docs", /\b(doc|readme|changelog|comment)/i],
	["commit", /\b(commit|push|merge|rebase|branch)/i],
	["deploy", /\b(deploy|release|install|publish|rollout)/i],
	["investigate", /\b(investigat|debug|diagnos|root cause|why\b)/i],
	["implement", /\b(implement|add|create|support|feature|new\b)/i],
	["ui", /\b(ui|ux|layout|design|style|css|widget|screen|dialog|button|sidebar|tab)\b/i],
	["performance", /\b(perf|slow|fast|optimi|cpu|memory|latency)/i],
	["config", /\b(config|setting|option|env)/i],
	["research", /\b(research|search|find|explore|look)/i],
	["verify", /\b(verif|check|smoke|validat|confirm)/i],
	["plan", /\b(plan|todo|spec|roadmap)/i],
	["read", /\b(read|open|view|list|inspect|show)/i],
	["run", /\b(run|execut|start|launch)/i],
	["edit", /\b(edit|update|change|modif|replac|patch)/i],
	["data", /\b(data|database|sql|json|schema|migrat)/i],
	["network", /\b(http|api|request|fetch|server|port)/i],
];

/** OMP built-in tools; any other tool name (MCP, extensions) becomes "other_tool". */
const TOOLS = new Set([
	"read", "write", "edit", "bash", "eval", "grep", "glob", "find", "lsp", "ast_grep", "ast_edit", "todo", "task", "ask",
	"web_search", "fetch", "browser", "debug", "job", "wait", "await", "write_file", "notebook", "irc", "checkpoint",
	"rewind", "resolve", "report_finding", "yield", "manage_skill", "inspect_image", "generate_image", "github", "exit_plan_mode",
	"search_tool_bm25", "calc", "recipe", "render_mermaid", "ssh", "python", "todo_write", "submit_result",
]);

export function topics(text: string, max = 3): string[] {
	const found: string[] = [];
	for (const [word, pattern] of TOPICS) {
		if (found.length >= max) break;
		if (pattern.test(text)) found.push(word);
	}
	return found;
}

const words = (text: string) => text.split(/\s+/).filter(Boolean).length;
const about = (n: number) => (n < 15 ? "under 15" : n < 50 ? "about 30" : n < 150 ? "about 100" : n < 400 ? "about 250" : "over 400");

/** Lines that read as a list item: "1.", "2)", "-", "*", "•". */
export function listedItems(text: string): number {
	return text.split("\n").filter(line => /^\s*(\d{1,2}[.)]|[-*•])\s+\S/.test(line)).length;
}

export function describeRequest(text: string): string {
	const listed = listedItems(text);
	const kind = /\?\s*$/.test(text.trim()) ? "question" : "instruction";
	const parts = [`sanitized request: ${kind} of ${about(words(text))} words`];
	if (listed >= 2) parts.push(`${listed} listed items`);
	if (/```/.test(text)) parts.push("includes code");
	const t = topics(text);
	if (t.length) parts.push(`topics ${t.join(" ")}`);
	return parts.join("; ");
}

export function describeAgentMessage(text: string): string {
	const parts = [`sanitized agent message of ${about(words(text))} words`];
	if (/\b(done|complete[ds]?|finished|all set|implemented|fixed|committed|pushed|landed)\b/i.test(text)) parts.push("reports completion");
	if (/\b(fail(s|ed|ing)?|error|blocked|cannot|can't|unable)\b/i.test(text)) parts.push("mentions failure");
	if (/\?/.test(text)) parts.push("asks a question");
	const t = topics(text, 2);
	if (t.length) parts.push(`topics ${t.join(" ")}`);
	return parts.join("; ");
}

export function sanitizeToolName(name: unknown): string {
	return typeof name === "string" && TOOLS.has(name) ? name : "other_tool";
}

export function describeIntent(intent: unknown): string | undefined {
	if (typeof intent !== "string" || !intent.trim()) return undefined;
	const t = topics(intent, 2);
	return t.length ? `topics ${t.join(" ")}` : "unspecified";
}

/** Stable per-session placeholders for todo titles; the done/start ops still match by title. */
export class TodoTitles {
	#map = new Map<string, string>();
	title(original: unknown): string {
		const key = typeof original === "string" ? original.trim() : "";
		let label = this.#map.get(key);
		if (!label) {
			const t = topics(key, 2);
			label = `task ${this.#map.size + 1}${t.length ? ` topics ${t.join(" ")}` : ""}`;
			this.#map.set(key, label);
		}
		return label;
	}
}

// Every word the sanitized events and the production payload frame may contain.
const FRAME_WORDS = `
sanitized request instruction question of under about over words listed items includes code topics
agent message reports completion mentions failure asks a
task unspecified other_tool failed model error provider retry unknown
agent working turn ended idle the s own plan it may span several prompts pending in_progress done dropped
`;
const ALLOWED = new Set([...FRAME_WORDS.split(/\s+/).filter(Boolean), ...TOPICS.map(([w]) => w), ...TOOLS]);

function strings(value: unknown, out: string[] = []): string[] {
	if (typeof value === "string") out.push(value);
	else if (Array.isArray(value)) for (const v of value) strings(v, out);
	else if (value && typeof value === "object") for (const v of Object.values(value)) strings(v, out);
	return out;
}

/** Words of a payload outside the closed vocabulary; empty means the payload is safe to send. */
export function unexpectedWords(payload: string): string[] {
	const bad = new Set<string>();
	for (const s of strings(JSON.parse(payload))) {
		for (const word of s.toLowerCase().match(/[a-z_]+/g) ?? []) if (!ALLOWED.has(word)) bad.add(word);
	}
	return [...bad];
}

export function assertSanitized(payload: string): void {
	const bad = unexpectedWords(payload);
	if (bad.length) throw new Error(`payload outside the sanitized vocabulary (${bad.length} words)`);
}
