import { describe, expect, test } from "bun:test";
import { estimateProgress } from "../src/estimator";
import { applyEvent, type BranchEntry, clean, emptyFacts, IDLE_GAP_MS, inferBlocker, inferNextHint, inferPhase, seedFromBranch } from "../src/facts";
import { type Clock, Observer, type Report } from "../src/observer";
import { formatElapsed, formatRemaining, widgetLines } from "../src/widget";

class FakeClock implements Clock {
	time = 0;
	#timers = new Map<number, { at: number; fn: () => void }>();
	#next = 1;
	now() {
		return this.time;
	}
	setTimeout(fn: () => void, ms: number) {
		const id = this.#next++;
		this.#timers.set(id, { at: this.time + ms, fn });
		return id;
	}
	clearTimeout(handle: unknown) {
		this.#timers.delete(handle as number);
	}
	advance(ms: number) {
		const end = this.time + ms;
		for (;;) {
			const due = [...this.#timers.entries()].filter(([, t]) => t.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
			if (!due) break;
			this.#timers.delete(due[0]);
			this.time = Math.max(this.time, due[1].at);
			due[1].fn();
		}
		this.time = end;
	}
	pendingCount() {
		return this.#timers.size;
	}
}

function harness(options = {}) {
	const clock = new FakeClock();
	const observer = new Observer(clock, { tickMs: 1_000, smoothDisplay: true, allowEarlyAbstain: true, ...options });
	const reports: Report[] = [];
	observer.enable("s1", r => reports.push(r));
	let n = 0;
	const tool = (name = "bash", intent = "step", isError = false, sessionId = "s1", durMs = 1_000) => {
		n += 1;
		observer.observe(sessionId, { type: "tool_execution_start", toolCallId: `c${n}`, toolName: name, intent: `${intent} ${n}` });
		clock.advance(durMs);
		observer.observe(sessionId, { type: "tool_execution_end", toolCallId: `c${n}`, toolName: name, isError });
	};
	return { clock, observer, reports, tool };
}

describe("facts, sanitization and confirmed milestones", () => {
	test("clean redacts secrets, strips ANSI/code, and stays fast (no ReDoS) on huge inputs", () => {
		const red = clean("Deploy with key sk-live-abcdefghijklmnopqrstuv and password=hunter2 ```cat /etc/shadow``` \x1b[31mhi\x1b[0m", 200);
		expect(red).not.toContain("sk-live");
		expect(red).not.toContain("hunter2");
		expect(red).not.toContain("/etc/shadow");
		expect(red).not.toContain("\x1b");
		expect(red).toContain("[redacted]");

		const t0 = performance.now();
		const huge = clean("x".repeat(200_000), 500);
		const elapsed = performance.now() - t0;
		expect(huge.length).toBeLessThanOrEqual(500);
		expect(elapsed).toBeLessThan(50);
	});

	test("todo updates count confirmed milestones only; failed todo calls do not mark tasks done", () => {
		const facts = emptyFacts();
		applyEvent(facts, { type: "before_agent_start", prompt: "Do 2 steps" }, 0);
		applyEvent(
			facts,
			{ type: "tool_execution_start", toolCallId: "t1", toolName: "todo", args: { op: "init", list: [{ phase: "A", items: ["Step 1", "Step 2"] }] } },
			1_000,
		);
		// Before tool_execution_end, todo list is not yet confirmed
		expect(facts.todos.length).toBe(0);
		applyEvent(
			facts,
			{
				type: "tool_execution_end",
				toolCallId: "t1",
				toolName: "todo",
				isError: false,
				result: {
					details: {
						op: "init",
						phases: [{ name: "A", tasks: [{ content: "Step 1", status: "in_progress" }, { content: "Step 2", status: "pending" }] }],
					},
				},
			},
			2_000,
		);
		expect(facts.todos.length).toBe(2);
		expect(facts.todos.filter(t => t.status === "done").length).toBe(0);

		// A FAILED todo done call must NOT mark Step 1 done
		applyEvent(facts, { type: "tool_execution_start", toolCallId: "t2", toolName: "todo", args: { op: "done", task: "Step 1" } }, 5_000);
		applyEvent(facts, { type: "tool_execution_end", toolCallId: "t2", toolName: "todo", isError: true }, 6_000);
		expect(facts.todos.filter(t => t.status === "done").length).toBe(0);
		expect(facts.todoDoneEventsS.length).toBe(0);

		// A SUCCEEDED todo block + done via authoritative details.phases
		applyEvent(facts, { type: "tool_execution_start", toolCallId: "t3", toolName: "todo", args: { op: "block", task: "Step 2" } }, 10_000);
		applyEvent(
			facts,
			{
				type: "tool_execution_end",
				toolCallId: "t3",
				toolName: "todo",
				isError: false,
				result: {
					details: {
						op: "block",
						phases: [{ name: "A", tasks: [{ content: "Step 1", status: "completed" }, { content: "Step 2", status: "blocked" }] }],
					},
				},
			},
			11_000,
		);
		expect(facts.todos.filter(t => t.status === "done").length).toBe(1);
		expect(facts.todos.filter(t => t.status === "blocked").length).toBe(1);
		expect(inferBlocker(facts)).toContain("blocked todo: Step 2");
	});

	test("active time leaves out open ask waits and caps silent gaps at IDLE_GAP_MS", () => {
		const facts = emptyFacts();
		const min = 60_000;
		applyEvent(facts, { type: "before_agent_start", prompt: "Refactor parser" }, 0);
		applyEvent(facts, { type: "tool_execution_start", toolCallId: "a", toolName: "ask" }, 2 * min);
		expect(estimateProgress(facts, 50 * min).phase).toBe("waiting");
		applyEvent(facts, { type: "tool_execution_end", toolCallId: "a", toolName: "ask", isError: false }, 60 * min);
		// One silent hour after answer counts at most IDLE_GAP_MS (10 min) -> 2 + 10 = 12 min.
		expect(facts.activeMs).toBe(2 * min);
		applyEvent(facts, { type: "tool_execution_start", toolCallId: "b", toolName: "read" }, 120 * min);
		expect(facts.activeMs).toBe(2 * min + IDLE_GAP_MS);
	});

	test("seedFromBranch recovers elapsed time, todos, tools and prior prompts on mid-session opt-in", () => {
		const t0 = 1_700_000_000_000;
		const branch: BranchEntry[] = [
			{ type: "message", timestamp: new Date(t0).toISOString(), message: { role: "user", content: "First quick task" } },
			{ type: "message", timestamp: new Date(t0 + 45_000).toISOString(), message: { role: "assistant", stopReason: "stop", content: [{ type: "text", text: "Done." }] } },
			{ type: "message", timestamp: new Date(t0 + 60_000).toISOString(), message: { role: "user", content: "Build feature X" } },
			{
				type: "message",
				timestamp: new Date(t0 + 70_000).toISOString(),
				message: {
					role: "assistant",
					stopReason: "toolUse",
					model: "fixture-model",
					usage: { input: 5000, output: 200 },
					content: [{ type: "toolCall", id: "c1", name: "todo", arguments: { op: "init" } }],
				},
			},
			{
				type: "message",
				timestamp: new Date(t0 + 72_000).toISOString(),
				message: {
					role: "toolResult",
					toolCallId: "c1",
					toolName: "todo",
					isError: false,
					details: {
						op: "init",
						phases: [{ name: "Build", tasks: [{ content: "Write code", status: "completed" }, { content: "Run tests", status: "in_progress" }] }],
					},
				},
			},
		];
		const facts = emptyFacts();
		seedFromBranch(facts, branch, t0 + 120_000, false);
		expect(facts.priorPromptDurationsS.length).toBe(1);
		expect(facts.requests).toBe(2);
		expect(facts.goal).toBe("Build feature X");
		expect(facts.running).toBe(true);
		expect(facts.todos.filter(t => t.status === "done").length).toBe(1);
		expect(facts.toolCalls).toBe(1);
		const r = estimateProgress(facts, t0 + 120_000);
		expect(r.estimate.progress).not.toBeNull();
		expect(r.estimate.remainingMinutes).not.toBeNull();
	});
});

describe("local estimator consistency & behavior", () => {
	test("withholds estimate when no request or active plan was seen", () => {
		const facts = emptyFacts();
		applyEvent(facts, { type: "tool_execution_start", toolCallId: "1", toolName: "read" }, 10_000);
		const r = estimateProgress(facts, 15_000);
		expect(r.estimate.progress).toBeNull();
		expect(r.estimate.remainingMinutes).toBeNull();
		expect(r.estimate.abstainReason).toBe("no request seen yet");
	});

	test("progress and remainingMinutes are mathematically consistent and have ordered quantile bounds", () => {
		const facts = emptyFacts();
		applyEvent(facts, { type: "before_agent_start", prompt: "Implement the parser and add tests" }, 0);
		for (let i = 1; i <= 6; i++) {
			applyEvent(facts, { type: "tool_execution_start", toolCallId: `t${i}`, toolName: i < 4 ? "read" : "edit" }, i * 10_000);
			applyEvent(facts, { type: "tool_execution_end", toolCallId: `t${i}`, toolName: i < 4 ? "read" : "edit", isError: false }, i * 10_000 + 5_000);
		}
		const r = estimateProgress(facts, 90_000);
		expect(r.estimate.progress).not.toBeNull();
		expect(r.estimate.remainingMinutes).not.toBeNull();
		expect(r.estimate.lowMinutes).not.toBeNull();
		expect(r.estimate.highMinutes).not.toBeNull();
		expect(r.estimate.lowMinutes!).toBeLessThanOrEqual(r.estimate.remainingMinutes! + 1e-6);
		expect(r.estimate.highMinutes!).toBeGreaterThanOrEqual(r.estimate.remainingMinutes! - 1e-6);
		// Exact mathematical consistency: progress == elapsed / (elapsed + remaining)
		const expectedProg = Math.min(0.99, Math.max(0.01, 90 / (90 + r.estimate.remainingMinutes! * 60)));
		expect(Math.abs(r.estimate.progress! - expectedProg)).toBeLessThan(1e-4);
	});

	test("completing todos increases estimated progress and reduces remaining time", () => {
		const facts = emptyFacts();
		applyEvent(facts, { type: "before_agent_start", prompt: "Complete 4 steps" }, 0);
		const mkPhases = (doneCount: number) => ({
			op: doneCount === 0 ? "init" : "done",
			phases: [
				{
					name: "Main",
					tasks: [1, 2, 3, 4].map(i => ({
						content: `Step ${i}`,
						status: i <= doneCount ? "completed" : i === doneCount + 1 ? "in_progress" : "pending",
					})),
				},
			],
		});
		applyEvent(facts, { type: "tool_execution_start", toolCallId: "t0", toolName: "todo", args: { op: "init" } }, 5_000);
		applyEvent(facts, { type: "tool_execution_end", toolCallId: "t0", toolName: "todo", isError: false, result: { details: mkPhases(0) } }, 6_000);
		const early = estimateProgress(facts, 30_000);

		for (let step = 1; step <= 4; step++) {
			const base = step * 30_000;
			applyEvent(facts, { type: "tool_execution_start", toolCallId: `e${step}`, toolName: "edit" }, base);
			applyEvent(facts, { type: "tool_execution_end", toolCallId: `e${step}`, toolName: "edit", isError: false }, base + 5_000);
			applyEvent(facts, { type: "tool_execution_start", toolCallId: `td${step}`, toolName: "todo", args: { op: "done" } }, base + 10_000);
			applyEvent(facts, { type: "tool_execution_end", toolCallId: `td${step}`, toolName: "todo", isError: false, result: { details: mkPhases(step) } }, base + 11_000);
		}
		const allDone = estimateProgress(facts, 135_000);
		expect(allDone.estimate.progress!).toBeGreaterThan(early.estimate.progress! + 0.3);
		expect(allDone.estimate.remainingMinutes!).toBeLessThan(early.estimate.remainingMinutes!);
		expect(allDone.phase).toBe("wrapping up");
		expect(inferNextHint(facts)).toBe("final reply");
	});
});

describe("observer lifecycle, prompt attribution and live ticking", () => {
	test("prompt attribution resets on new prompt; live tick updates while running and stops when idle or disabled", () => {
		const { clock, observer, reports, tool } = harness();
		expect(clock.pendingCount()).toBe(0); // idle -> no timer running!

		observer.observe("s1", { type: "before_agent_start", prompt: "Prompt A" });
		expect(observer.report().reading?.promptId).toBe(1);
		expect(observer.report().reading?.estimate.abstainReason).toBe("starting");
		expect(clock.pendingCount()).toBe(1);

		tool("edit", "Editing file");
		clock.advance(5_000);
		expect(observer.report().reading?.estimate.progress).not.toBeNull();
		const countBeforeTick = reports.length;
		clock.advance(3_000);
		expect(reports.length).toBeGreaterThanOrEqual(countBeforeTick + 3);

		// Turn ends: ticking stops!
		observer.observe("s1", { type: "agent_end" });
		expect(clock.pendingCount()).toBe(0);
		expect(observer.report().reading?.phase).toBe("done");

		// New prompt B: promptId increments and old reading cannot bleed over
		observer.observe("s1", { type: "before_agent_start", prompt: "Prompt B" });
		expect(observer.report().reading?.promptId).toBe(2);
		expect(observer.report().facts.toolCalls).toBe(0);

		// Disable stops everything
		const seen = reports.length;
		observer.disable();
		expect(clock.pendingCount()).toBe(0);
		clock.advance(10_000);
		expect(reports.length).toBe(seen);
	});

	test("events from another session are ignored", () => {
		const { observer } = harness();
		observer.observe("other-session", { type: "before_agent_start", prompt: "Wrong session" });
		expect(observer.report().facts.requests).toBe(0);
	});
});

describe("widget layout and responsiveness", () => {
	const plain = (_tone: string, text: string) => text;

	test("renders progress bar, percentage, remaining time, elapsed, phase badge, and fits all widths", () => {
		const facts = emptyFacts();
		applyEvent(facts, { type: "before_agent_start", prompt: "Build feature" }, 0);
		applyEvent(facts, { type: "tool_execution_start", toolCallId: "t0", toolName: "todo", args: { op: "init" } }, 1_000);
		applyEvent(
			facts,
			{
				type: "tool_execution_end",
				toolCallId: "t0",
				toolName: "todo",
				isError: false,
				result: {
					details: {
						op: "init",
						phases: [{ name: "Core", tasks: [{ content: "Write parser", status: "completed" }, { content: "Add tests", status: "in_progress" }, { content: "Update docs", status: "pending" }] }],
					},
				},
			},
			2_000,
		);
		applyEvent(facts, { type: "tool_execution_start", toolCallId: "t1", toolName: "bash", intent: "Running unit tests" }, 40_000);
		const reading = estimateProgress(facts, 85_000);
		const report: Report = { phase: "on", sessionId: "s1", facts, reading };

		const wide = widgetLines(report, 85_000, 100, plain);
		expect(wide).toHaveLength(1);
		expect(wide[0]).toContain("PROGRESS");
		expect(wide[0]).toContain("━");
		expect(wide[0]).toContain("1m 25s");
		expect(wide[0]).toContain("exploring");
		// Does not duplicate OMP's native tool spinner or TODO next item
		expect(wide[0]).not.toContain("last:");
		expect(wide[0]).not.toContain("next:");

		for (const width of [40, 54, 72, 100]) {
			for (const line of widgetLines(report, 85_000, width, plain)) {
				expect(line.length).toBeLessThan(width);
			}
		}

		// Narrow mode compacts labels and drops phase tag
		const narrow = widgetLines(report, 85_000, 54, plain);
		expect(narrow).toHaveLength(1);
		expect(narrow[0]).toContain("1m25s");
		expect(narrow[0]).not.toContain("exploring");
	});

	test("renders waiting for you, done state, blocker row, and empty when off", () => {
		const facts = emptyFacts();
		applyEvent(facts, { type: "before_agent_start", prompt: "Fix bug" }, 0);
		applyEvent(facts, { type: "tool_execution_start", toolCallId: "q", toolName: "ask" }, 20_000);
		const waitingReport: Report = { phase: "on", sessionId: "s1", facts, reading: estimateProgress(facts, 30_000) };
		const waitingLines = widgetLines(waitingReport, 30_000, 100, plain);
		expect(waitingLines).toHaveLength(1);
		expect(waitingLines[0]).toContain("waiting for you");
		expect(waitingLines[0]).toContain("waiting");

		// Active retry/blocker adds a 2nd line
		applyEvent(facts, { type: "tool_execution_end", toolCallId: "q", toolName: "ask", isError: false }, 35_000);
		applyEvent(facts, { type: "auto_retry_start", attempt: 1, maxAttempts: 3, errorMessage: "rate limited" }, 40_000);
		const blockedReport: Report = { phase: "on", sessionId: "s1", facts, reading: estimateProgress(facts, 42_000) };
		const blockedLines = widgetLines(blockedReport, 42_000, 100, plain);
		expect(blockedLines).toHaveLength(2);
		expect(blockedLines[0]).toContain("blocked");
		expect(blockedLines[1]).toContain("rate limited");
		applyEvent(facts, { type: "auto_retry_end", success: true }, 45_000);

		applyEvent(facts, { type: "agent_end" }, 65_000);
		const doneReport: Report = { phase: "on", sessionId: "s1", facts, reading: estimateProgress(facts, 65_000) };
		const doneLines = widgetLines(doneReport, 65_000, 100, plain);
		expect(doneLines).toHaveLength(1);
		expect(doneLines[0]).toContain("100%");
		expect(doneLines[0]).toContain("done in 50s");
		expect(doneLines[0]).not.toContain("left");

		expect(widgetLines({ ...doneReport, phase: "off" }, 65_000, 100, plain)).toEqual([]);
	});

	test("tracks async background jobs, wait snapshots, edit details paths, and edit-verify cycles", () => {
		const facts = emptyFacts();
		applyEvent(facts, { type: "before_agent_start", prompt: "Implement parser and run tests" }, 0);
		applyEvent(facts, { type: "tool_execution_start", toolCallId: "e1", toolName: "edit", intent: "Edit parser" }, 5_000);
		applyEvent(facts, { type: "tool_execution_end", toolCallId: "e1", toolName: "edit", isError: false, result: { details: { path: "src/parser.ts" } } }, 7_000);
		applyEvent(facts, { type: "tool_execution_start", toolCallId: "b1", toolName: "bash", intent: "Running unit tests" }, 10_000);
		applyEvent(facts, { type: "tool_execution_end", toolCallId: "b1", toolName: "bash", isError: false }, 15_000);
		applyEvent(facts, { type: "tool_execution_start", toolCallId: "tk1", toolName: "task", intent: "Delegate review", args: { tasks: [null] } }, 20_000);
		applyEvent(
			facts,
			{
				type: "tool_execution_end",
				toolCallId: "tk1",
				toolName: "task",
				isError: false,
				result: { details: { async: true, progress: [{ id: "Rev1", agent: "reviewer", modelRole: "slow", status: "pending" }] } },
			},
			20_100,
		);
		expect(facts.filesEdited.has("src/parser.ts")).toBe(true);
		expect(facts.editVerifyCycles).toBe(1);
		expect(facts.intentTestCount).toBe(1);
		expect(facts.bgJobs.size).toBe(1);
		expect(facts.bgJobs.get("Rev1")?.isSlowModel).toBe(true);

		applyEvent(facts, { type: "tool_execution_start", toolCallId: "w1", toolName: "wait" }, 30_000);
		applyEvent(
			facts,
			{
				type: "tool_execution_end",
				toolCallId: "w1",
				toolName: "wait",
				isError: false,
				result: { details: { jobs: [{ id: "Rev1", type: "task", status: "completed", durationMs: 45_000 }] } },
			},
			75_000,
		);
		expect(facts.bgJobs.size).toBe(0);
		expect(facts.bgDoneCount).toBe(1);
		expect(facts.bgDoneMaxS).toBe(45);
	});

	test("formatting helpers produce clean human durations", () => {
		expect(formatElapsed(14_000)).toBe("14s");
		expect(formatElapsed(88_000)).toBe("1m 28s");
		expect(formatElapsed(88_000, true)).toBe("1m28s");
		expect(formatElapsed(720_000)).toBe("12m");
		expect(formatRemaining(0.3, 0.1, 0.5, false)).toBe("<30s left");
		expect(formatRemaining(3.0, 1.8, 5.2, false)).toBe("~2–5m left");
		expect(formatRemaining(3.0, 2.6, 3.4, false)).toBe("~3m left");
		expect(formatRemaining(3.0, 1.8, 5.2, true)).toBe("~3m");
	});
});
