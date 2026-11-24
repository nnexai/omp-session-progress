import { expect, test } from "bun:test";
import { landmarks } from "../eval/landmarks";
import { evalHead, trainGBDTHead } from "../src/gbdt";

test("a later reply doesn't change clock landmarks already observed before an ask", () => {
	const prefix = [{ index: 0, t: 0, ms: 22_000 }, { index: 1, t: 22_000, ms: 0 }];
	const probes = (tail: number) => landmarks([...prefix, { index: 2, t: 82_000, ms: tail }], false);
	const location = ({ at, cut }: { at: number; cut: number }) => [at, cut];
	expect(probes(6_000).map(location)).toEqual([[5_000, 1], [20_000, 1]]);
	expect(probes(50_000).slice(0, 3).map(location)).toEqual([[5_000, 1], [20_000, 1], [95_000, 3]]);
	expect(probes(50_000)[2]!.remainingS).toBe(37);
});

test("boosted forecasts distinguish adjacent observed tool-count bins at their trained boundary", () => {
	const x = Array.from({ length: 128 }, (_, i) => [i < 64 ? Math.log(3) : Math.log(4)]);
	const y = x.map((_, i) => i < 64 ? 4 : 6);
	const head = trainGBDTHead(x, y, y.map(() => 0), y.map(() => 60), y.map(() => 1), {
		anchor: "zero", nTrees: 1, lr: 1, maxDepth: 1, minLeaf: 8, l2: 0, colSample: 1, seed: 7,
	});
	expect(evalHead(head, [Math.log(3)])).toBeCloseTo(4);
	expect(evalHead(head, [Math.log(4)])).toBeCloseTo(6);
});
