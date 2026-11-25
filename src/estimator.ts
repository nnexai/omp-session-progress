// Local, zero-LLM progress & remaining-time estimator.
// Uses a compact 3-anchor GBDT ensemble + calibrated P25/P75 quantile heads
// trained on historic OMP sessions. Runs in < 0.1 ms with zero network calls.

import { type Facts, inferBlocker, inferNextHint, inferPhase, type WorkPhase } from "./facts";
import { extractFeatures, SMOOTH_S } from "./features";
import { type EnsembleModel, evalEnsembleZ, evalQuantileRangeZ } from "./gbdt";
import { MODEL_DATA } from "./model-data";
import { calibratedRange } from "./range-calibration";
import RANGE_DATA from "./range-data.json";

export type Confidence = "low" | "medium" | "high";

export interface Estimate {
	/** Elapsed share of active time (0..1), mathematically consistent with remainingMinutes. */
	progress: number | null;
	/** Median (P50) expected active minutes remaining. */
	remainingMinutes: number | null;
	/** Calibrated lower bound (P25) remaining minutes. */
	lowMinutes: number | null;
	/** Calibrated upper bound (P75) remaining minutes. */
	highMinutes: number | null;
	/** Raw log-scale target z = ln(remainingS + 30) for optional live display smoothing. */
	logRemZ: number | null;
	/** Confidence tier based on available evidence. */
	confidence: Confidence;
	/** Reason when an estimate is withheld. */
	abstainReason: string | null;
}

export interface Reading {
	/** Prompt number (`facts.requests`) that produced this reading — prevents cross-prompt bleed. */
	promptId: number;
	estimate: Estimate;
	phase: WorkPhase;
	blocker: string | null;
	next: string | null;
	at: number;
}

export interface EstimateOptions {
	model?: EnsembleModel;
	/** When true (live widget), abstain during the first couple seconds before any tool/plan signal arrives. */
	allowEarlyAbstain?: boolean;
	/** Optional externally smoothed log-scale z value to keep the live progress bar steady. */
	overrideZ?: number;
}

function zToMinutes(z: number): number {
	return Math.max(0, Math.exp(Math.min(10.5, Math.max(0, z))) - SMOOTH_S) / 60;
}

export function estimateProgress(facts: Facts, now: number, opts: EstimateOptions = {}): Reading {
	const model = opts.model ?? MODEL_DATA;
	const phase = inferPhase(facts);
	const blocker = inferBlocker(facts);
	const next = inferNextHint(facts);
	const promptId = facts.requests;

	const liveTodos = facts.todos.filter(t => t.status !== "dropped");
	const hasBasis = Boolean(facts.goal) || (facts.todoUpdated && liveTodos.length > 0);
	if (!hasBasis) {
		return {
			promptId,
			estimate: {
				progress: null,
				remainingMinutes: null,
				lowMinutes: null,
				highMinutes: null,
				logRemZ: null,
				confidence: "low",
				abstainReason: "no request seen yet",
			},
			phase,
			blocker,
			next,
			at: now,
		};
	}

	const feat = extractFeatures(facts, now);

	// In live interactive use, briefly show "starting…" before the first tool/plan arrives.
	if (opts.allowEarlyAbstain && facts.running && feat.elapsedS < 3 && facts.toolCalls === 0 && !feat.hasActiveTodo) {
		return {
			promptId,
			estimate: {
				progress: null,
				remainingMinutes: null,
				lowMinutes: null,
				highMinutes: null,
				logRemZ: null,
				confidence: "low",
				abstainReason: "starting",
			},
			phase,
			blocker,
			next,
			at: now,
		};
	}

	const rawZ = evalEnsembleZ(model, feat.x, feat.logElapsed, feat.anchorZ, {
		elapsedS: feat.elapsedS,
		toolCalls: feat.toolCalls,
		hasActiveTodo: feat.hasActiveTodo,
	});
	const z = opts.overrideZ ?? rawZ;
	const remMinutes = zToMinutes(z);
	const remS = remMinutes * 60;
	const progress =
		feat.elapsedS <= 0
			? 0.02
			: Math.min(0.99, Math.max(0.01, feat.elapsedS / (feat.elapsedS + remS + 1e-6)));

	const { lowZ, highZ } = evalQuantileRangeZ(model, feat.x, feat.logElapsed, feat.anchorZ, rawZ);
	const shift = z - rawZ;
	let lowMinutes = Math.min(remMinutes, zToMinutes(lowZ + shift));
	let highMinutes = Math.max(remMinutes, zToMinutes(highZ + shift));
	// This residual distribution was fit out-of-fold for this predictor. Do not
	// recalibrate unrelated injected models, or alter short-prompt readings.
	if (model === MODEL_DATA && feat.elapsedS > 60) {
		const [lowS, highS] = calibratedRange(RANGE_DATA, feat.elapsedS, remS, lowMinutes * 60, highMinutes * 60);
		lowMinutes = lowS / 60;
		highMinutes = highS / 60;
	}

	// Confidence tier: high when active todos have real velocity or post-edit verification is underway.
	let confidence: Confidence = "medium";
	if ((feat.hasActiveTodo && feat.todoDoneThisPrompt >= 1) || (facts.postWriteCleanExecCount > 0 && facts.toolCalls >= 4)) {
		confidence = "high";
	} else if (feat.elapsedS < 15 && facts.toolCalls < 3 && !feat.hasActiveTodo) {
		confidence = "low";
	}

	return {
		promptId,
		estimate: {
			progress,
			remainingMinutes: remMinutes,
			lowMinutes,
			highMinutes,
			logRemZ: rawZ,
			confidence,
			abstainReason: null,
		},
		phase,
		blocker,
		next,
		at: now,
	};
}
