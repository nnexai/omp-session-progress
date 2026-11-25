#!/usr/bin/env python3
"""Matched landmark analysis.

Run: python3 eval/long-landmarks/matched.py --rows /PRIVATE/rows.jsonl \
    --output /PRIVATE/matched.json

Design fixed before inspecting outcomes: at ages 300 and 600 active seconds,
compare completed prompts lasting at least 1800 seconds with shorter survivors.
Match separately within each split, across distinct sessions, without replacement.
Use log1p prefix counts and development-risk-set population standard deviations;
zero development variance uses scale 1. Greedily take the closest available pair
by standardized RMS distance, with RMS <= 1 and every coordinate <= 2 SD.
Ties follow input order. Neither onset nor time remaining enters matching.

For post-landmark first-onset comparisons restrict already-matched pairs to those
where both members survive a fixed further 300 seconds. Also report the subset
where neither member had that signal by the landmark. This restriction uses
outcomes only to define a common observation window, not to select matches.
"""
import argparse
import json
import math
import statistics
from collections import Counter

FEATURES = ("calls", "turns", "writes", "tests", "todoOpen", "promptChars", "failures", "delegates")
SIGNALS = ("execError", "scope", "blocked", "slowTool", "debugLoop", "delegate")
AGES = (300, 600)
SPLITS = ("dev", "official", "enriched")
WINDOW = 300
LONG = 1800


def exact_p(a, b):
    """Two-sided exact sign test for discordant pairs; descriptive, not clustered."""
    n = a + b
    if not n:
        return None
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(a, b) + 1)) / 2**n)


def comparison(pairs, predicate):
    counts = Counter((bool(predicate(a)), bool(predicate(b))) for a, b, _ in pairs)
    long_only = counts[True, False]
    short_only = counts[False, True]
    return {
        "pairs": len(pairs),
        "both": counts[True, True],
        "long_only": long_only,
        "short_only": short_only,
        "neither": counts[False, False],
        "long_rate": (counts[True, True] + long_only) / len(pairs) if pairs else None,
        "short_rate": (counts[True, True] + short_only) / len(pairs) if pairs else None,
        "exact_sign_p_unclustered": exact_p(long_only, short_only),
    }


def transformed(row):
    values = tuple(float(row["x"][name]) for name in FEATURES)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("prefix features must be finite nonnegative counts")
    return tuple(math.log1p(v) for v in values)


def summarize(rows, age, scales):
    eligible = [r for r in rows if r["age"] == age and r["total"] > age]
    long = [r for r in eligible if r["total"] >= LONG]
    short = [r for r in eligible if r["total"] < LONG]
    lv = [transformed(r) for r in long]
    sv = [transformed(r) for r in short]
    candidates = []
    for i, a in enumerate(long):
        for j, b in enumerate(short):
            if a["sid"] == b["sid"]:
                continue
            diff = tuple(abs(x - y) / s for x, y, s in zip(lv[i], sv[j], scales))
            distance = math.sqrt(sum(d * d for d in diff) / len(FEATURES))
            if distance <= 1 and max(diff) <= 2:
                candidates.append((distance, i, j))
    candidates.sort()
    used_long, used_short, pairs = set(), set(), []
    for distance, i, j in candidates:
        if i not in used_long and j not in used_short:
            used_long.add(i)
            used_short.add(j)
            pairs.append((long[i], short[j], distance))
    fixed = [(a, b, d) for a, b, d in pairs
             if a["total"] >= age + WINDOW and b["total"] >= age + WINDOW]
    quality = {}
    for k, name in enumerate(FEATURES):
        signed = [(transformed(a)[k] - transformed(b)[k]) / scales[k] for a, b, _ in pairs]
        quality[name] = {
            "mean_signed_development_sd": statistics.fmean(signed) if signed else None,
            "mean_absolute_development_sd": statistics.fmean(map(abs, signed)) if signed else None,
            "max_absolute_development_sd": max(map(abs, signed)) if signed else None,
        }
    signals = {}
    for signal in SIGNALS:
        def before(row):
            onset = row["onsets"][signal]
            return onset is not None and onset <= age

        def after(row):
            onset = row["onsets"][signal]
            return onset is not None and age < onset <= age + WINDOW

        neither_before = [(a, b, d) for a, b, d in fixed if not before(a) and not before(b)]
        signals[signal] = {
            "before_or_at_landmark_all_pairs": comparison(pairs, before),
            "before_or_at_landmark_fixed_window_pairs": comparison(fixed, before),
            "first_onset_after_landmark_fixed_window_pairs": comparison(fixed, after),
            "first_onset_after_landmark_neither_previously_observed": comparison(neither_before, after),
        }
    distances = [d for _, _, d in pairs]
    return {
        "risk_set": {"long": len(long), "short": len(short),
                     "long_sessions": len({r['sid'] for r in long}),
                     "short_sessions": len({r['sid'] for r in short})},
        "match": {"pairs": len(pairs), "long_coverage": len(pairs) / len(long) if long else None,
                  "short_coverage": len(pairs) / len(short) if short else None,
                  "fixed_window_pairs": len(fixed),
                  "fixed_window_pair_fraction": len(fixed) / len(pairs) if pairs else None,
                  "candidate_edges": len(candidates),
                  "mean_rms_distance": statistics.fmean(distances) if distances else None,
                  "max_rms_distance": max(distances) if distances else None,
                  "feature_balance": quality},
        "signals": signals,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with open(args.rows, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    keys = [(r["split"], r["sid"], r["pid"], r["age"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate prompt-landmark rows")
    result = {
        "design": {
            "endpoint": "completed prompt total active seconds >= 1800",
            "risk_set": "completed prompts still running at landmark; no wall-clock durations",
            "matching": "within split; distinct sessions; greedy ascending standardized RMS; without replacement; log1p prefix features; dev population SD by age; zero SD uses 1; RMS <= 1 and each absolute coordinate <= 2",
            "features": FEATURES, "signals": SIGNALS,
            "signal_definitions": {
                "execError": "failed tool receipt for an execution with verification or E2E intent",
                "scope": "recorded scope expansion",
                "blocked": "updated todo list has a blocked item",
                "slowTool": "non-ask tool reaches 60 active seconds; not proof of blocking",
                "debugLoop": "recorded debug loop",
                "delegate": "recorded delegation",
            },
            "post_window_active_seconds": WINDOW,
            "post_window_population": "already matched pairs with both surviving through landmark + 300 seconds",
            "limitations": [
                "Observational matching is not causal; unmeasured differences and greedy matching remain.",
                "Fixed-window survivor conditioning excludes short-window completions and can select on outcome.",
                "Only first signal onset is represented; pre-landmark onset precludes a later first onset.",
                "Exact sign p-values ignore repeated prompts and sessions, multiple signals and landmarks; treat as descriptive.",
                "Missing recorded signal is not proof of absence; small or zero discordance is inconclusive.",
                "Development, official, and enriched cohorts are reported separately; no holdout tuning.",
            ],
        },
        "landmarks": {},
    }
    for age in AGES:
        dev = [transformed(r) for r in rows if r["split"] == "dev" and r["age"] == age and r["total"] > age]
        if not dev:
            raise ValueError("empty development risk set")
        scales = [statistics.pstdev(column) or 1.0 for column in zip(*dev)]
        result["landmarks"][str(age)] = {
            "development_scales_log1p": dict(zip(FEATURES, scales)),
            "cohorts": {split: summarize([r for r in rows if r["split"] == split], age, scales) for split in SPLITS},
        }
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(output)
    print(output, end="")


if __name__ == "__main__":
    main()
