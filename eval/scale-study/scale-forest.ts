// Numeric CatBoost oblivious-tree export. Feature index x.length denotes active age e.
// Head transforms: 0 = log1p(seconds), 1 = minutes. The return order follows
// the fixed training recipes in eval/scale-study/boost.py.
export interface ScaleForest {
	version: number;
	featureCount: number;
	heads: {
		transform: number;
		scale: number;
		bias: number;
		trees: { features: number[]; borders: number[]; leaves: number[] }[];
	}[];
}

export function forestSeconds(model: ScaleForest, x: number[], e: number): number[] {
	const result: number[] = [];
	for (const head of model.heads) {
		let sum = 0;
		for (const tree of head.trees) {
			let leaf = 0;
			for (let i = 0; i < tree.features.length; i++) {
				const feature = tree.features[i]!;
				const value = feature === model.featureCount - 1 ? e : x[feature]!;
				if (Math.fround(value) > tree.borders[i]!) leaf |= 1 << i;
			}
			sum += tree.leaves[leaf]!;
		}
		const prediction = head.scale * sum + head.bias;
		result.push(Math.max(0, head.transform === 0 ? Math.expm1(prediction) : prediction * 60));
	}
	return result;
}
