// Correct the width of the existing central interval, not the point forecast.
// The 6×6 grid uses octave-spaced active ages (90 s) and point ETAs (45 s).
export interface RangeModel { low: number[]; high: number[]; trust: number[] }
export const gridCoordinate = (seconds: number, scale: number) =>
	Math.min(5, Math.max(0, Math.log2(Math.max(1, seconds / scale))));

export function calibratedRange(
	model: RangeModel, elapsedS: number, pointS: number, lowS: number, highS: number,
): [number, number] {
	if (elapsedS <= 60) return [lowS, highS];
	const a = gridCoordinate(elapsedS, 90), b = gridCoordinate(pointS, 45);
	const i = Math.floor(a), j = Math.floor(b), p = a - i, q = b - j;
	const right = Math.min(5, j + 1), down = Math.min(5, i + 1) * 6, row = i * 6;
	const interpolate = (v: number[]) =>
		(1 - p) * ((1 - q) * v[row + j]! + q * v[row + right]!) + p * ((1 - q) * v[down + j]! + q * v[down + right]!);
	const weight = interpolate(model.trust) * Math.min(1, (elapsedS - 60) / 120);
	const center = Math.log(pointS + 30);
	const bound = (old: number, delta: number) =>
		Math.max(0, Math.exp((1 - weight) * Math.log(old + 30) + weight * (center + delta)) - 30);
	return [Math.min(pointS, bound(lowS, interpolate(model.low))), Math.max(pointS, bound(highS, interpolate(model.high)))];
}
