// Chronological replay of one trace through the production observer: the
// same event ingress (applyEvent), payload (buildPayload), cadence (Observer
// timers on a virtual clock driven by the historic timestamps) and reply
// parsing. A model call happens at virtual time `at`; the replay waits for it
// before applying any later event, so a call can only ever see the prefix of
// the session up to `at`. Every sent payload is re-derived from that prefix
// and compared, and checked against the sanitized vocabulary.

import { applyEvent, buildPayload, emptyFacts } from "../src/facts";
import { type Clock, Observer, type ObserverOptions, type Reading, type Report, type Transport } from "../src/observer";
import type { Usage } from "../src/transport";
import type { TraceEvent } from "./extract";
import { assertSanitized } from "./sanitize";

/** What a call returned, with the provider's measurements when a model was called. */
export interface CallerResult {
	text: string;
	inputTokens?: number;
	outputTokens?: number;
	usage?: Usage;
	ms?: number;
	model?: string;
}

export type Caller = (request: { system: string; user: string; signal: AbortSignal }) => Promise<CallerResult>;

export interface Variant {
	name: string;
	/** Replaces the production system prompt. */
	system?: string;
	observer?: Partial<ObserverOptions>;
	/** Answer locally instead of calling the model: a reply text, or undefined to call. */
	local?: (payload: string) => string | undefined;
	/** Rewrites the production payload before it is sent, e.g. to reproduce an earlier payload format. */
	transformPayload?: (payload: string) => string;
}

export interface CallLog {
	/** Virtual send time. */
	at: number;
	/** Time of the last event applied before the call; always ≤ at. */
	prefixEnd: number;
	/** Number of trace events applied before the call. */
	prefixEvents: number;
	/** 0-based prompt index the call belongs to (-1 before any prompt). */
	prompt: number;
	running: boolean;
	payloadChars: number;
	/** True when the provider was called; false for a local answer. */
	model: boolean;
	ms?: number;
	inputTokens?: number;
	outputTokens?: number;
	cacheRead?: number;
	cacheWrite?: number;
	costUsd?: number;
	error?: string;
}

export interface ShownReading {
	prompt: number;
	reading: Reading;
}

export interface ReplayResult {
	trace: string;
	variant: string;
	calls: CallLog[];
	readings: ShownReading[];
	skippedUnchanged: number;
	capped: boolean;
	paused: boolean;
	failures: number;
}

interface Timer {
	at: number;
	fn: () => void;
}

class VirtualClock implements Clock {
	#now: number;
	#next = 1;
	#timers = new Map<number, Timer>();
	constructor(start: number) {
		this.#now = start;
	}
	now(): number {
		return this.#now;
	}
	setTimeout(fn: () => void, ms: number): unknown {
		const id = this.#next++;
		this.#timers.set(id, { at: this.#now + Math.max(0, ms), fn });
		return id;
	}
	clearTimeout(handle: unknown): void {
		if (typeof handle === "number") this.#timers.delete(handle);
	}
	/** Fire every timer due up to `t`, in order, settling each before the next. */
	async advanceTo(t: number, settle: () => Promise<void>): Promise<void> {
		for (;;) {
			let due: [number, Timer] | undefined;
			for (const entry of this.#timers) if (entry[1].at <= t && (!due || entry[1].at < due[1].at)) due = entry;
			if (!due) break;
			this.#timers.delete(due[0]);
			this.#now = Math.max(this.#now, due[1].at);
			due[1].fn();
			await settle();
		}
		this.#now = Math.max(this.#now, t);
	}
}

const tick = () => new Promise<void>(resolve => setImmediate(resolve));

export async function replay(traceId: string, events: TraceEvent[], variant: Variant, caller: Caller): Promise<ReplayResult> {
	const first = events[0]?.t ?? 0;
	const clock = new VirtualClock(first - 1);
	const observer = new Observer(clock, variant.observer);
	const calls: CallLog[] = [];
	const readings: ShownReading[] = [];
	let applied = 0;
	let inFlight: Promise<unknown> | null = null;
	let lastReading: Reading | null = null;
	let violation: string | undefined;
	// Folded independently from exactly the events applied so far.
	const shadow = emptyFacts();

	const transport: Transport = ({ system, user, signal }) => {
		const facts = observer.report().facts;
		// The payload must be exactly what the applied prefix yields, and must stay inside the sanitized vocabulary.
		if (buildPayload(shadow) !== user) violation ??= "payload differs from the payload of the applied prefix";
		else {
			try {
				assertSanitized(user);
			} catch (error) {
				violation ??= error instanceof Error ? error.message : String(error);
			}
		}
		// The observer treats a throwing transport as a provider failure; the violation fails the replay below.
		if (violation) return Promise.reject(new Error(`replay invariant: ${violation}`));
		const sent = variant.transformPayload?.(user) ?? user;
		const log: CallLog = {
			at: clock.now(),
			prefixEnd: applied ? (events[applied - 1]?.t ?? first) : first,
			prefixEvents: applied,
			prompt: facts.requests - 1,
			running: facts.running,
			payloadChars: sent.length,
			model: false,
		};
		calls.push(log);
		const local = variant.local?.(sent);
		const run = (async (): Promise<CallerResult> => {
			if (local !== undefined) return { text: local };
			log.model = true;
			try {
				const result = await caller({ system: variant.system ?? system, user: sent, signal });
				log.ms = result.ms;
				log.inputTokens = result.inputTokens;
				log.outputTokens = result.outputTokens;
				log.cacheRead = result.usage?.cacheRead;
				log.cacheWrite = result.usage?.cacheWrite;
				log.costUsd = result.usage?.cost?.total;
				return result;
			} catch (error) {
				log.error = error instanceof Error ? error.message.slice(0, 160) : String(error);
				throw error;
			}
		})();
		inFlight = run.catch(() => undefined);
		return run;
	};

	const sink = (report: Report) => {
		if (report.reading && report.reading !== lastReading) {
			lastReading = report.reading;
			readings.push({ prompt: report.facts.requests - 1, reading: report.reading });
		}
	};

	const settle = async () => {
		await tick();
		while (inFlight) {
			const pending = inFlight;
			await pending;
			if (inFlight === pending) inFlight = null;
			// Let the observer finish the call (parse, emit, reschedule).
			await tick();
		}
	};

	observer.enable(traceId, transport, sink);
	for (const { t, event } of events) {
		await clock.advanceTo(t, settle);
		observer.observe(traceId, event);
		applyEvent(shadow, event);
		applied += 1;
	}
	// Let the last pending refresh happen.
	await clock.advanceTo((events.at(-1)?.t ?? first) + 120_000, settle);
	const end = observer.report();
	observer.disable();
	if (violation) throw new Error(`replay invariant violated in ${traceId}: ${violation}`);
	return {
		trace: traceId,
		variant: variant.name,
		calls,
		readings,
		skippedUnchanged: end.skippedUnchanged,
		capped: end.phase === "capped",
		paused: end.phase === "paused",
		failures: calls.filter(c => c.error).length,
	};
}
