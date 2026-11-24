import { describe, expect, test } from "bun:test";
import { applyEvent, buildPayload, emptyFacts, type ObservedEvent } from "../src/facts";
import { label, type TraceEvent } from "../bench/extract";
import { type Caller, replay } from "../bench/replay";
import { assertSanitized, describeAgentMessage, describeIntent, describeRequest, TodoTitles, unexpectedWords } from "../bench/sanitize";
import { checkpoints, cost, progressAt } from "../bench/score";

const reply = (stage: string | null) => JSON.stringify({ summary: "s", estimate: { scope: "current prompt", stage, basis: "b" }, blocker: null, next: "run the tests" });

function session(): TraceEvent[] {
	const titles = new TodoTitles();
	const ev: TraceEvent[] = [];
	let t = 1_000_000;
	const at = (dt: number, event: ObservedEvent) => {
		t += dt;
		ev.push({ t, event });
	};
	at(0, { type: "before_agent_start", prompt: describeRequest("1. build it\n2. test it"), listed: 2 });
	at(1000, { type: "tool_execution_start", toolCallId: "a", toolName: "todo", args: { op: "init", items: [titles.title("build it"), titles.title("test it")] } });
	at(10, { type: "tool_execution_end", toolCallId: "a", toolName: "todo", isError: false });
	for (let i = 0; i < 6; i++) {
		at(20_000, { type: "tool_execution_start", toolCallId: `b${i}`, toolName: "bash", intent: describeIntent("Running tests") });
		at(5_000, { type: "tool_execution_end", toolCallId: `b${i}`, toolName: "bash", isError: i === 2 });
	}
	at(1000, { type: "tool_execution_start", toolCallId: "c", toolName: "todo", args: { op: "done", task: titles.title("build it") } });
	at(10, { type: "tool_execution_end", toolCallId: "c", toolName: "todo", isError: false });
	// The canary: a todo item that only exists late in the session.
	at(30_000, { type: "tool_execution_start", toolCallId: "d", toolName: "todo", args: { op: "append", items: [titles.title("canary step")] } });
	at(10, { type: "tool_execution_end", toolCallId: "d", toolName: "todo", isError: false });
	at(1000, { type: "message_end", message: { role: "assistant", stopReason: "stop", content: [{ type: "text", text: describeAgentMessage("All done.") }] } });
	at(1, { type: "agent_end", stopReason: "stop" });
	return ev;
}

describe("replay", () => {
	test("each call sees only the prefix up to its send time", async () => {
		const events = session();
		const sent: string[] = [];
		const caller: Caller = async ({ user }) => {
			sent.push(user);
			return { text: reply("middle") };
		};
		const result = await replay("s-test", events, { name: "t", observer: { minIntervalMs: 0, debounceMs: 1000, maxWaitMs: 1000 } }, caller);
		expect(result.calls.length).toBeGreaterThan(3);
		const canaryAt = events.find(e => e.event.toolCallId === "d")?.t ?? 0;
		for (const [i, call] of result.calls.entries()) {
			expect(call.prefixEnd).toBeLessThanOrEqual(call.at);
			// Exactly the events applied before the call, all of them at or before its send time.
			const prefix = events.slice(0, call.prefixEvents);
			expect(prefix.every(e => e.t <= call.at)).toBe(true);
			const facts = emptyFacts();
			for (const { event } of prefix) applyEvent(facts, event);
			expect(sent[i]).toBe(buildPayload(facts));
			if (call.at < canaryAt) expect(sent[i]).not.toContain("task 3");
		}
		expect(sent.some(s => s.includes("task 3"))).toBe(true);
	});

	test("a payload outside the sanitized vocabulary fails the replay", async () => {
		const events: TraceEvent[] = [{ t: 1, event: { type: "before_agent_start", prompt: "please refactor frobnicator.ts" } }];
		await expect(replay("s-raw", events, { name: "t" }, async () => ({ text: reply(null) }))).rejects.toThrow("replay invariant");
	});
});

describe("sanitizer", () => {
	test("descriptions keep no words of the original text", () => {
		const secret = "Fix the zorblax parser in /home/alice/proj/zorblax.ts, token=ghp_abcdefghijklmnopqrstuvwx. Then?";
		const payload = JSON.stringify({ a: describeRequest(secret), b: describeAgentMessage(secret), c: describeIntent(secret), d: new TodoTitles().title(secret) });
		expect(unexpectedWords(payload)).toEqual([]);
		expect(payload).not.toMatch(/zorblax|alice|home|ghp_/);
		expect(() => assertSanitized(JSON.stringify({ x: "zorblax" }))).toThrow();
	});
});

describe("labels and scoring", () => {
	test("todo progress, outcomes and checkpoints", () => {
		const events = session();
		const prompts = label(events);
		expect(prompts.length).toBe(1);
		const p = prompts[0];
		if (!p) throw new Error("no prompt");
		expect(p.outcome).toBe("completed");
		expect(p.structured).toBe(true);
		// One item finished in the prompt: before it 0, after it 1.
		expect(progressAt(p, p.start + 1)).toBe(0);
		expect(progressAt(p, p.end)).toBe(1);
		const steer = label([
			{ t: 1, event: { type: "before_agent_start", prompt: "x" } },
			{ t: 2, event: { type: "tool_execution_start", toolCallId: "a", toolName: "read" } },
			{ t: 3, event: { type: "before_agent_start", prompt: "y" } },
			{ t: 4, event: { type: "message_end", message: { role: "assistant", stopReason: "aborted", content: [] } } },
			{ t: 5, event: { type: "agent_end", stopReason: "aborted" } },
		]);
		expect(steer.map(x => x.outcome)).toEqual(["superseded", "aborted"]);
		expect(steer[0]?.structured).toBe(false);
		// A premature "appears done" while running early.
		const cps = checkpoints(
			{ id: "s", project: "p", events, prompts },
			{
				trace: "s",
				variant: "t",
				calls: [],
				readings: [{ prompt: 0, reading: { summary: "s", estimate: { scope: "current prompt", stage: "appears done", basis: "b" }, blocker: null, next: "run tests", at: p.start + 2000 } }],
				skippedUnchanged: 0,
				capped: false,
				paused: false,
				failures: 0,
			},
		);
		expect(cps[0]?.premature).toBe(true);
		expect(cps[0]?.nextActual).toBe("run");
		expect(cps[0]?.nextGuess).toBe("run");
	});

	test("cost sums only model calls", () => {
		const base = { at: 0, prefixEnd: 0, prefixEvents: 0, prompt: 0, running: true, payloadChars: 100 };
		const c = cost([
			{
				trace: "s",
				variant: "t",
				calls: [
					{ ...base, model: true, inputTokens: 900, cacheRead: 500, cacheWrite: 0, outputTokens: 120, costUsd: 0.001, ms: 1500 },
					{ ...base, model: true, inputTokens: 1000, cacheRead: 600, cacheWrite: 10, outputTokens: 100, costUsd: 0.002, ms: 2500 },
					{ ...base, model: false },
				],
				readings: [],
				skippedUnchanged: 2,
				capped: true,
				paused: false,
				failures: 0,
			},
		]);
		expect([c.calls, c.modelCalls, c.inputTokens, c.cacheRead, c.outputTokens, c.skippedUnchanged, c.cappedSessions]).toEqual([3, 2, 1900, 1100, 220, 2, 1]);
		expect(c.costUsd).toBeCloseTo(0.003);
	});
});
