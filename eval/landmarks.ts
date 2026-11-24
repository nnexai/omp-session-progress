import type { Probe } from "./grade";

export type Sample = "sparse" | "dense" | "clock";
type Segment = { index: number; t: number; ms: number };

/** One chronological pass. The clock grid is independent of the eventual duration:
 * every prompt still running at a landmark contributes, even with zero tool calls. */
export function landmarks(segments: Segment[], structuredSample: boolean): Array<Probe & { sample: Sample }> {
	const total = segments.reduce((sum, segment) => sum + segment.ms, 0);
	const targets: Array<{ ms: number; sample: Sample }> = [];
	if (structuredSample) {
		for (const fraction of [0.15, 0.4, 0.65, 0.9]) targets.push({ ms: total * fraction, sample: "sparse" });
		for (let i = 0; i < 16; i++) targets.push({ ms: total * ((i + 0.5) / 16), sample: "dense" });
	}
	for (let s = 5; s * 1000 < total; s += Math.max(15, Math.round(0.12 * s))) {
		targets.push({ ms: s * 1000, sample: "clock" });
	}
	targets.sort((a, b) => a.ms - b.ms);
	let i = 0, elapsed = 0;
	return targets.map(({ ms, sample }) => {
		while (elapsed + segments[i]!.ms < ms) elapsed += segments[i++]!.ms;
		const s = segments[i]!;
		return { sample, at: Math.round(s.t + ms - elapsed), cut: s.index + 1, elapsedS: ms / 1000, remainingS: (total - ms) / 1000 };
	});
}
