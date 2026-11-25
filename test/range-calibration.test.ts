import { expect, test } from 'bun:test';
import { calibratedRange, type RangeModel } from '../src/range-calibration';

const table: RangeModel = { low: Array(36).fill(Math.log(.5)), high: Array(36).fill(Math.log(2)), trust: Array(36).fill(1) };

test('a short prompt keeps its original interval; a steady long prompt uses residual quartiles in seconds', () => {
	expect(calibratedRange(table, 60, 90, 20, 900)).toEqual([20, 900]);
	const [lo, hi] = calibratedRange(table, 180, 90, 20, 900);
	expect(lo).toBeCloseTo(30); // (90 + 30) × .5 − 30
	expect(hi).toBeCloseTo(210);
	const halfway = calibratedRange(table, 120, 90, 20, 900);
	expect(halfway[0]).toBeGreaterThan(20); expect(halfway[0]).toBeLessThan(lo);
	expect(halfway[1]).toBeGreaterThan(hi); expect(halfway[1]).toBeLessThan(900);
});

test('very old prompts use the final observed bin rather than extrapolating an unbounded interval', () => {
	const trust = Array(36).fill(0); trust[35] = 1;
	const [lo, hi] = calibratedRange({ ...table, trust }, 86_400, 3600, 60, 20_000);
	expect(lo).toBeCloseTo(1785); expect(hi).toBeCloseTo(7230);
});
