// Grader of the progress estimate against offline truth. Pure; the runner
// calls it after the model answered, and nothing here reaches the model.
//
// Per probe (a point inside one historic prompt, k of its N tool calls started):
// - progress accuracy = 1 − |predicted share − k/N|
// - remaining accuracy = 1 − |ln((predicted + 1) / (N − k + 1))| / ln 8,
//   floored at 0: 1 when exact, 0.67 at 2× off, 0 at 8× off or worse
// - score = mean of the two; a probe without an estimate (unparsable reply,
//   withheld estimate) scores 0 on both.
// A case's grade is the mean over its probes.

export interface Probe {
	/** Index into the session's events: the prefix the estimator sees is events[0, cut). */
	cut: number;
	/** Tool calls of the prompt started before the cut. */
	done: number;
	/** Tool calls of the prompt not yet started at the cut. */
	remaining: number;
}

export interface Prediction {
	progress: number | null;
	remaining: number | null;
}

export interface Grade {
	score: number;
	progress: number;
	remaining: number;
	/** Probes whose remaining guess is within 2× of the truth (on the +1 scale). */
	within2x: number;
}

const LN8 = Math.log(8);

export function gradeProbe(probe: Probe, p: Prediction): Grade {
	if (p.progress === null || p.remaining === null) return { score: 0, progress: 0, remaining: 0, within2x: 0 };
	const truth = probe.done / (probe.done + probe.remaining);
	const progress = 1 - Math.abs(p.progress - truth);
	const ratio = Math.abs(Math.log((p.remaining + 1) / (probe.remaining + 1)));
	const remaining = Math.max(0, 1 - ratio / LN8);
	return { score: (progress + remaining) / 2, progress, remaining, within2x: ratio <= Math.LN2 ? 1 : 0 };
}

export function gradeCase(probes: Probe[], predictions: Prediction[]): Grade {
	const grades = probes.map((probe, i) => gradeProbe(probe, predictions[i] ?? { progress: null, remaining: null }));
	const mean = (key: keyof Grade) => grades.reduce((a, g) => a + g[key], 0) / grades.length;
	return { score: mean("score"), progress: mean("progress"), remaining: mean("remaining"), within2x: mean("within2x") };
}

/** No-model reference: "we are halfway", i.e. as many calls left as done. */
export const halfway = (probe: Probe): Prediction => ({ progress: 0.5, remaining: probe.done });
