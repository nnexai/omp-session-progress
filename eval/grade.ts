// Grader of the progress estimate against offline truth. Pure; the runner
// calls it after the model answered, and nothing here reaches the model.
//
// Truth is the prompt's active time (eval/extract.ts `activeSegments`): from
// the request to the reply, without time waiting on the user in `ask` calls,
// each silent gap capped at 10 minutes.
// Per probe (a moment inside one historic prompt):
// - progress accuracy = 1 − |predicted share − elapsed / total|
// - remaining accuracy = 1 − |ln((predicted s + 30) / (true s + 30))| / ln 8,
//   floored at 0: 1 when exact, 0.67 at 2× off, 0 at 8× off or worse
// - score = mean of the two; a probe without an estimate (unparsable reply,
//   withheld estimate) scores 0 on both.
// A case's grade is the mean over its probes.

export interface Probe {
	/** Probe time (ms epoch): the estimator sees the events before it and is asked at this time. */
	at: number;
	/** Index into the session's events: the prefix the estimator sees is events[0, cut). */
	cut: number;
	/** Active seconds of the prompt before the probe. */
	elapsedS: number;
	/** Active seconds of the prompt after the probe. */
	remainingS: number;
}

export interface Prediction {
	progress: number | null;
	remainingMinutes: number | null;
}

export interface Grade {
	score: number;
	progress: number;
	remaining: number;
	/** Probes whose remaining guess is within 2× of the truth (on the +30 s scale). */
	within2x: number;
}

const LN8 = Math.log(8);
const SMOOTH_S = 30;

export function gradeProbe(probe: Probe, p: Prediction): Grade {
	if (p.progress === null || p.remainingMinutes === null) return { score: 0, progress: 0, remaining: 0, within2x: 0 };
	const truth = probe.elapsedS / (probe.elapsedS + probe.remainingS);
	const progress = 1 - Math.abs(p.progress - truth);
	const ratio = Math.abs(Math.log((p.remainingMinutes * 60 + SMOOTH_S) / (probe.remainingS + SMOOTH_S)));
	const remaining = Math.max(0, 1 - ratio / LN8);
	return { score: (progress + remaining) / 2, progress, remaining, within2x: ratio <= Math.LN2 ? 1 : 0 };
}

export function gradeCase(probes: Probe[], predictions: Prediction[]): Grade {
	const grades = probes.map((probe, i) => gradeProbe(probe, predictions[i] ?? { progress: null, remainingMinutes: null }));
	const mean = (key: keyof Grade) => grades.reduce((a, g) => a + g[key], 0) / grades.length;
	return { score: mean("score"), progress: mean("progress"), remaining: mean("remaining"), within2x: mean("within2x") };
}

/** No-model reference: "we are halfway", i.e. as much time left as spent. */
export const halfway = (probe: Probe): Prediction => ({ progress: 0.5, remainingMinutes: probe.elapsedS / 60 });
