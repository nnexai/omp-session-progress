// The local progress observer: folds OMP extension events into observed facts,
// computes instant zero-LLM progress estimates on every event and on a 1s tick
// while the agent is actively working, and pushes updates to the display sink.
// Zero OMP imports, so it runs deterministically under `bun test` with a FakeClock.

import { estimateProgress, type Reading } from "./estimator";
import { activeMs, applyEvent, type BranchEntry, emptyFacts, type Facts, type ObservedEvent, seedFromBranch } from "./facts";
import { SMOOTH_S } from "./features";
import { MODEL_DATA } from "./model-data";

export interface ObserverOptions {
	/** Live UI refresh interval while the agent is working (ms). Set 0 to disable ticking. */
	tickMs: number;
	/** Apply gentle EMA display smoothing on log-remaining-time between closely spaced events. */
	smoothDisplay: boolean;
	/** Withhold point estimate for the first ~3s before any tool or plan signal arrives. */
	allowEarlyAbstain: boolean;
}

export const DEFAULT_OPTIONS: ObserverOptions = {
	tickMs: 1_000,
	smoothDisplay: true,
	allowEarlyAbstain: true,
};

export interface Clock {
	now(): number;
	setTimeout(fn: () => void, ms: number): unknown;
	clearTimeout(handle: unknown): void;
}

export type Phase = "off" | "on";

export interface Report {
	phase: Phase;
	sessionId: string | null;
	facts: Facts;
	reading: Reading | null;
}

export type Sink = (report: Report) => void;

export class Observer {
	readonly options: ObserverOptions;
	#clock: Clock;
	#sink: Sink | null = null;
	#phase: Phase = "off";
	#sessionId: string | null = null;
	#facts: Facts = emptyFacts();
	#reading: Reading | null = null;
	#epoch = 0;
	#timer: unknown = null;

	// Live display smoothing state (strictly scoped to one promptId).
	#smoothPromptId = -1;
	#smoothZ: number | null = null;
	#smoothAt = 0;
	#smoothTodoDone = 0;

	// Strictly causal online calibration state:
	// - `#sessionBiases`: recency-weighted mean log residuals of prior completed prompts in this session.
	// - `#promptSamples`: active-time forecast milestones within the current prompt for centered slippage.
	#sessionBiases: number[] = [];
	#promptSamples: Array<{ e: number; zBase: number; totS: number }> = [];

	constructor(clock: Clock, options: Partial<ObserverOptions> = {}) {
		this.#clock = clock;
		this.options = { ...DEFAULT_OPTIONS, ...options };
	}

	get phase(): Phase {
		return this.#phase;
	}

	get sessionId(): string | null {
		return this.#sessionId;
	}

	enable(sessionId: string, sink: Sink): void {
		this.#stop();
		this.#phase = "on";
		this.#sessionId = sessionId;
		this.#sink = sink;
		this.#facts = emptyFacts();
		this.#reading = null;
		this.#resetSmoothing();
		this.#resetCalibration(true);
		this.#refresh();
	}

	/** Turn off: cancel any tick timer and clear state so nothing reaches the sink again. */
	disable(): void {
		if (this.#phase === "off" && this.#sessionId === null) return;
		this.#stop();
		this.#phase = "off";
		this.#sessionId = null;
		this.#sink = null;
		this.#facts = emptyFacts();
		this.#reading = null;
		this.#resetSmoothing();
		this.#resetCalibration(true);
	}

	/** Forget the observed history (branch/tree/clear) without leaving the session. */
	reset(): void {
		if (this.#phase === "off") return;
		this.#stop();
		this.#facts = emptyFacts();
		this.#reading = null;
		this.#resetSmoothing();
		this.#resetCalibration(true);
		this.#refresh();
	}

	/** Feed one event from the observed session; ignored unless on and for this session. */
	observe(sessionId: string, event: ObservedEvent): void {
		if (this.#phase === "off" || sessionId !== this.#sessionId) return;
		const prevPrompt = this.#facts.requests;
		const prevPriorCount = this.#facts.priorPromptDurationsS.length;
		if (!applyEvent(this.#facts, event, this.#clock.now())) return;
		if (this.#facts.priorPromptDurationsS.length > prevPriorCount) {
			const totalS = this.#facts.priorPromptDurationsS[this.#facts.priorPromptDurationsS.length - 1]!;
			this.#recordCompletedPromptBias(totalS);
		}
		if (this.#facts.requests !== prevPrompt) {
			// Prompt changed: immediately clear any prior prompt's reading, smoothing, and slippage landmarks.
			this.#reading = null;
			this.#resetSmoothing();
			this.#resetCalibration(false);
		}
		this.#refresh();
	}

	/** Seed a single event from prior context (mainly for tests or fallback). */
	seed(sessionId: string, event: ObservedEvent): void {
		if (this.#phase === "off" || sessionId !== this.#sessionId) return;
		applyEvent(this.#facts, event, this.#clock.now());
		this.#resetSmoothing();
		this.#resetCalibration(false);
		this.#refresh();
	}

	/** Replay the current OMP session branch on mid-session opt-in so elapsed time, todos, and tools are accurate immediately. */
	seedBranch(sessionId: string, branch: BranchEntry[], isIdle: boolean): void {
		if (this.#phase === "off" || sessionId !== this.#sessionId) return;
		seedFromBranch(this.#facts, branch, this.#clock.now(), isIdle);
		this.#resetSmoothing();
		this.#resetCalibration(true);
		this.#refresh();
	}

	report(): Report {
		return {
			phase: this.#phase,
			sessionId: this.#sessionId,
			facts: this.#facts,
			reading: this.#reading,
		};
	}

	#resetSmoothing(): void {
		this.#smoothPromptId = -1;
		this.#smoothZ = null;
		this.#smoothAt = 0;
		this.#smoothTodoDone = 0;
	}

	#resetCalibration(clearSession: boolean): void {
		this.#promptSamples = [];
		if (clearSession) this.#sessionBiases = [];
	}

	#recordCompletedPromptBias(totalS: number): void {
		if (totalS > 5 && this.#promptSamples.length > 0) {
			let sum = 0;
			let count = 0;
			for (let i = 0; i < this.#promptSamples.length; i++) {
				const s = this.#promptSamples[i]!;
				if (s.e < totalS) {
					sum += Math.log(totalS - s.e + SMOOTH_S) - s.zBase;
					count++;
				}
			}
			if (count > 0) {
				const bias = Math.max(-2, Math.min(2, sum / count));
				this.#sessionBiases.push(bias);
				if (this.#sessionBiases.length > 8) this.#sessionBiases.shift();
			}
		}
		this.#promptSamples = [];
	}

	#sessionBias(): number {
		const n = this.#sessionBiases.length;
		if (n === 0) return 0;
		let sum = 0;
		let wt = 0;
		for (let i = 0; i < n; i++) {
			const w = Math.pow(0.8, n - 1 - i);
			sum += w * this.#sessionBiases[i]!;
			wt += w;
		}
		return sum / (wt + 2.5);
	}

	#centeredSlippage(elapsedS: number, curTotS: number): number {
		if (elapsedS < 60 || this.#promptSamples.length === 0) return 0;
		let prevRef: { e: number; totS: number } | null = null;
		const cutoff = elapsedS * 0.55;
		for (let i = 0; i < this.#promptSamples.length; i++) {
			const lm = this.#promptSamples[i]!;
			if (lm.e <= cutoff) prevRef = lm;
			else break;
		}
		if (!prevRef) return 0;
		const rawSlip = Math.max(-2, Math.min(2, Math.log((curTotS + SMOOTH_S) / (prevRef.totS + SMOOTH_S))));
		const eBin = Math.min(7, Math.max(0, Math.floor(Math.log2(Math.max(1, elapsedS / 30)))));
		const center = MODEL_DATA.slipCenter?.[eBin] ?? 0;
		return rawSlip - center;
	}

	#stop(): void {
		this.#epoch += 1;
		if (this.#timer !== null) this.#clock.clearTimeout(this.#timer);
		this.#timer = null;
	}

	#refresh(): void {
		if (this.#phase !== "on") return;
		const now = this.#clock.now();
		let reading = estimateProgress(this.#facts, now, {
			allowEarlyAbstain: this.options.allowEarlyAbstain,
		});

		if (this.#facts.running && reading.estimate.logRemZ !== null) {
			const baseZ = reading.estimate.logRemZ;
			const elapsedS = activeMs(this.#facts, now) / 1000;
			const baseRemS = Math.max(0, Math.exp(Math.min(10.5, Math.max(0, baseZ))) - SMOOTH_S);
			const curTotS = elapsedS + baseRemS;

			// Strictly causal online calibration (prior-prompt Empirical Bayes + centered forecast slippage).
			const onlineZ = baseZ + 0.6 * this.#sessionBias() + 0.25 * this.#centeredSlippage(elapsedS, curTotS);

			// Record milestone at >=14s active spacing for future slippage & post-prompt residual calibration.
			const lastSample = this.#promptSamples[this.#promptSamples.length - 1];
			if (elapsedS >= 5 && (!lastSample || elapsedS - lastSample.e >= 14)) {
				this.#promptSamples.push({ e: elapsedS, zBase: baseZ, totS: curTotS });
			}

			let targetZ = onlineZ;
			if (this.options.smoothDisplay) {
				const todoDone = this.#facts.todoDoneEventsS.length;
				const samePrompt = this.#smoothPromptId === reading.promptId && this.#smoothZ !== null;
				const milestoneChanged = todoDone !== this.#smoothTodoDone;
				const dtMs = now - this.#smoothAt;
				if (samePrompt && !milestoneChanged && dtMs >= 0 && dtMs < 15_000) {
					// Gentle EMA prevents visual jitter during fast tool bursts while still tracking trend smoothly.
					const alpha = dtMs < 2_000 ? 0.35 : 0.6;
					targetZ = alpha * onlineZ + (1 - alpha) * this.#smoothZ!;
				}
				this.#smoothZ = targetZ;
				this.#smoothPromptId = reading.promptId;
				this.#smoothAt = now;
				this.#smoothTodoDone = todoDone;
			}
			if (Math.abs(targetZ - baseZ) > 1e-6) {
				reading = estimateProgress(this.#facts, now, {
					allowEarlyAbstain: this.options.allowEarlyAbstain,
					overrideZ: targetZ,
				});
			}
		} else if (!this.#facts.running) {
			this.#resetSmoothing();
		}

		this.#reading = reading;
		this.#emit();
		this.#scheduleTick();
	}

	#scheduleTick(): void {
		if (this.#timer !== null) {
			this.#clock.clearTimeout(this.#timer);
			this.#timer = null;
		}
		if (this.#phase !== "on" || !this.#facts.running || this.options.tickMs <= 0) return;
		const epoch = this.#epoch;
		this.#timer = this.#clock.setTimeout(() => {
			if (epoch !== this.#epoch) return;
			this.#timer = null;
			this.#refresh();
		}, this.options.tickMs);
	}

	#emit(): void {
		try {
			this.#sink?.(this.report());
		} catch {
			// A failing display must never disturb the observed session.
		}
	}
}
