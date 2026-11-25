#!/usr/bin/env python3
"""Train fixed, numeric-only CatBoost ETA heads without reading the held-out fold.

Usage: boost.py ROOT fit 0|1|2|final
Inputs are ROOT/dev{0,1,2}.jsonl; output is ROOT/forest{split}.json.
The fixed heads target log-median, long-weighted log-median, Huber minute
error and a duration-stratified quantile, respectively.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from collections import Counter

import numpy as np
from catboost import CatBoostRegressor


# Keep every recipe fixed across cross-fitting and the final fit.
RECIPES = (
    # transform, loss, depth, iterations, learning rate, long-prompt tilt
    (0, "MAE", 6, 800, 0.035, 0.0),
    (0, "MAE", 6, 800, 0.035, 0.5),
    (1, "Huber:delta=5", 6, 800, 0.035, 0.0),
    # Fit the actual minute-error priority under a mixture of prompt strata,
    # rather than treating late checkpoints or abundant short jobs as more votes.
    (0, "Quantile:alpha=0.55", 5, 600, 0.045, -1.0),
)


def load_rows(root, split):
    folds = range(3) if split == "final" else (i for i in range(3) if i != int(split))
    rows = []
    for fold in folds:
        with (root / f"dev{fold}.jsonl").open() as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    if not rows:
        raise ValueError("no training rows")
    width = len(rows[0]["x"])
    if not width or any(len(row["x"]) != width for row in rows):
        raise ValueError("inconsistent feature width")
    # Baseline b is deliberately not read; IDs are only used for weights.
    x = np.asarray([row["x"] + [row["e"]] for row in rows], dtype=np.float32)
    y = np.asarray([row["y"] for row in rows], dtype=np.float64)
    age = np.asarray([row["elapsed"] for row in rows], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(age).all():
        raise ValueError("non-finite feature or label")
    if np.any(y < 0) or np.any(age < 0):
        raise ValueError("negative age or remaining label")
    counts = Counter((row["sid"], row["pid"]) for row in rows)
    weights = np.asarray([1.0 / counts[(row["sid"], row["pid"])] for row in rows])
    return x, y, age, weights


def export_head(model, transform, filename):
    model.save_model(str(filename), format="json")
    exported = json.loads(filename.read_text())
    scale, biases = exported["scale_and_bias"]
    trees = []
    for tree in exported["oblivious_trees"]:
        splits = tree["splits"] or []
        if any(split["split_type"] != "FloatFeature" for split in splits):
            raise ValueError("unexpected non-numeric CatBoost split")
        trees.append({
            "features": [split["float_feature_index"] for split in splits],
            "borders": [split["border"] for split in splits],
            "leaves": tree["leaf_values"],
        })
    return {"transform": transform, "scale": scale, "bias": biases[0], "trees": trees}


def verify_js(model, x, expected):
    # Exercise the TypeScript scorer against training-only CatBoost predictions.
    scorer = Path(__file__).with_name("scale-forest.ts")
    script = (
        'import {forestSeconds} from ' + json.dumps(scorer.as_uri()) + ';'
        'const d=JSON.parse(await Bun.stdin.text());'
        'console.log(JSON.stringify(d.x.map(r=>forestSeconds(d.model,r.slice(0,-1),r.at(-1)))));'
    )
    payload = json.dumps({"model": model, "x": x.tolist()}, separators=(",", ":"))
    result = subprocess.run(["bun", "-e", script], input=payload, text=True,
                            capture_output=True, check=True)
    actual = np.asarray(json.loads(result.stdout))
    if not np.allclose(actual, expected, rtol=2e-6, atol=1e-4):
        raise RuntimeError("JavaScript export predictions disagree with CatBoost")


def fit(root, split):
    x, y, age, base_weight = load_rows(root, split)
    artifact = {"version": 1, "featureCount": x.shape[1], "heads": []}
    # Deterministic sample spans the train set; no held-out data is opened.
    sample = np.linspace(0, len(x) - 1, min(256, len(x)), dtype=int)
    predictions = []
    with tempfile.TemporaryDirectory(dir=root) as tmp:
        for transform, loss, depth, iterations, rate, tilt in RECIPES:
            target = np.log1p(y) if transform == 0 else y / 60.0
            # Weight prompts equally and mildly emphasize longer total durations.
            weight = base_weight * np.maximum(1.0, np.minimum(4.0, (age + y) / 900.0) ** tilt)
            if tilt < 0:
                duration = age + y
                weight = 0.30 * base_weight / base_weight.sum()
                for mask, share in ((duration >= 180, 0.20), (duration >= 900, 0.30), (duration >= 1800, 0.20)):
                    den = base_weight[mask].sum()
                    if den > 0:
                        weight[mask] += share * base_weight[mask] / den
            weight *= len(weight) / weight.sum()
            model = CatBoostRegressor(loss_function=loss, iterations=iterations, depth=depth,
                                      learning_rate=rate, thread_count=4, random_seed=42,
                                      verbose=False, allow_writing_files=False)
            model.fit(x, target, sample_weight=weight)
            artifact["heads"].append(export_head(model, transform, Path(tmp) / "model.json"))
            raw = model.predict(x[sample])
            predictions.append(np.maximum(0, np.expm1(raw) if transform == 0 else raw * 60))
        verify_js(artifact, x[sample], np.column_stack(predictions))
    # Atomic replacement prevents exposing a partially written artifact.
    output = root / f"forest{split}.json"
    with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False) as handle:
        name = handle.name
        try:
            json.dump(artifact, handle, separators=(",", ":"), allow_nan=False)
        except BaseException:
            os.unlink(name)
            raise
    os.replace(name, output)


def main():
    if len(sys.argv) != 4 or sys.argv[2] != "fit" or sys.argv[3] not in ("0", "1", "2", "final"):
        raise SystemExit("usage: boost.py ROOT fit 0|1|2|final")
    fit(Path(sys.argv[1]), sys.argv[3])


if __name__ == "__main__":
    main()
