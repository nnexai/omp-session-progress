"""Offline diagnostic, not an ETA or a deployed alarm. Requires NumPy and SciPy.

python separation.py PRIVATE_ROWS.jsonl > result.json

Lock before scoring: separate ridge logistic classifier per age, fixed penalty 4,
features below, zero holdout tuning. Each fit gets one row per surviving prompt.
Development out-of-fold 90th percentile negative score sets each alert threshold.
Retain every completed prompt >5s, including ones with no tools. Never train on
per-prompt onsets, eventual duration (except target), outcome, or source identity.
"""
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

AGES = (5, 60, 180, 300, 600, 900, 1200, 1500)
CONTEXT = ('promptChars', 'listItems', 'heavy', 'approval', 'priorMax', 'priorLast')
WORK = ('calls', 'turns', 'writes', 'tests', 'todoOpen', 'todoDone', 'failures', 'delegates',
        'maxOpenAge', 'maxDoneAge', 'files', 'reads', 'scope', 'debug', 'cycles', 'bg',
        'open', 'think', 'tokens', 'wrap')


def fit(rows, names):
    def vector(rr):
        return np.log1p([[r['x'][name] for name in names] for r in rr])
    x = vector(rows)
    mean, sd = x.mean(axis=0), np.maximum(.25, x.std(axis=0))
    def design(x):
        return np.column_stack([np.ones(len(x)), np.clip((x - mean) / sd, -5, 5)])
    x = design(x)
    y = np.array([r['total'] >= 1800 for r in rows], float)
    def objective(beta):
        z = x @ beta
        penalty = np.r_[0., 4 * beta[1:]]
        return np.sum(np.logaddexp(0, z) - y * z) + beta @ penalty / 2, x.T @ (expit(z) - y) + penalty
    opt = minimize(objective, np.zeros(x.shape[1]), jac=True, method='L-BFGS-B')
    if not opt.success:
        raise RuntimeError(opt.message)
    return lambda rr: expit(design(vector(rr)) @ opt.x)


def rounded(x):
    return round(float(x), 4) if np.isfinite(x) else None


def rank_compare(rows, scores, other=None):
    """AUC or paired difference; resample whole sessions with the same weights for both scores."""
    y = np.array([r['total'] >= 1800 for r in rows], bool)
    if not y.any() or y.all():
        return {'value': None, 'ci95': None}
    clusters, idx = np.unique([r['sid'] for r in rows], return_inverse=True)
    n = len(clusters)
    def pairs(score):
        delta = score[y, None] - score[None, ~y]
        out = np.zeros((n, n))
        np.add.at(out, (idx[y, None], idx[None, ~y]), (delta > 0).astype(float) + .5 * (delta == 0))
        return out
    wins = pairs(np.array(scores))
    if other is not None:
        wins -= pairs(np.array(other))
    pos = np.bincount(idx[y], minlength=n)
    neg = np.bincount(idx[~y], minlength=n)
    w = np.random.default_rng(92741).multinomial(n, np.full(n, 1 / n), size=2000)
    den = (w @ pos) * (w @ neg)
    valid = den > 0
    boot = ((w[valid] @ wins) * w[valid]).sum(axis=1) / den[valid]
    return {'value': rounded(wins.sum() / (y.sum() * (~y).sum())),
            'ci95': list(map(rounded, np.quantile(boot, [.025, .975]))), 'validDraws': int(valid.sum())}


def confusion(rows, fired):
    y = np.array([r['total'] >= 1800 for r in rows], bool)
    return {'long': int(y.sum()), 'shorter': int((~y).sum()),
            'longFlagged': int(fired[y].sum()), 'shorterFlagged': int(fired[~y].sum())}


def key(r):
    return r['sid'], r['pid']


def analyze(rows):
    data = defaultdict(list)
    for r in rows:
        data[r['split'], r['age']].append(r)
    thresholds, landmarks = {}, {k: {} for k in ('official', 'enriched')}
    scorebook = defaultdict(dict)
    for age in AGES:
        dev = data['dev', age]
        if len(set(key(r) for r in dev)) != len(dev):
            raise ValueError('Duplicate prompt at landmark')
        xnames = CONTEXT + WORK
        oof = []
        # Integer session indices are private and unique within split; no prompt-level splitting.
        for f in range(5):
            val = [r for r in dev if r['sid'] % 5 == f]
            pred = fit([r for r in dev if r['sid'] % 5 != f], xnames)(val)
            oof.extend(v for r, v in zip(val, pred) if r['total'] < 1800)
        threshold = float(np.quantile(oof, .9, method='higher'))
        thresholds[age] = threshold
        full, context = fit(dev, xnames), fit(dev, CONTEXT)
        for split in landmarks:
            rr = data[split, age]
            s = full(rr)
            c = context(rr)
            scorebook[split, age] = {key(r): p for r, p in zip(rr, s)}
            hard = np.array([r['total'] >= 900 for r in rr])
            landmarks[split][age] = {
                **confusion(rr, s >= threshold), 'contributingSessions': len({r['sid'] for r in rr}),
                'actualMinutesLeftQuartiles': list(map(rounded, np.quantile(
                    [(r['total'] - age) / 60 for r in rr], [.25, .5, .75]))),
                'auc': rank_compare(rr, s), 'promptAndHistoryAuc': rank_compare(rr, c),
                'gainOverPromptAndHistory': rank_compare(rr, s, c),
                'aucExcludingUnder15min': rank_compare([r for flag, r in zip(hard, rr) if flag], s[hard]),
            }
    paired, alerts, clues = {}, {}, {}
    for split in landmarks:
        paired[split], alerts[split], clues[split] = {}, {}, {}
        for age in (300, 600, 900, 1200, 1500):
            rr = data[split, age]
            paired[split][age] = {}
            # Conditioning on surviving to the later landmark is for retrospective scoring ONLY.
            for prev in (5, 300, 600, 900, 1200):
                if prev >= age:
                    break
                now = [scorebook[split, age][key(r)] for r in rr]
                old = [scorebook[split, prev][key(r)] for r in rr]
                paired[split][age][prev] = {'earlierAuc': rank_compare(rr, old)['value'],
                                            'delta': rank_compare(rr, now, old)}
        all_rows = data[split, 5]
        seen = set()
        for age in AGES:
            seen.update(k for k, s in scorebook[split, age].items() if s >= thresholds[age])
            fired = np.array([key(r) in seen for r in all_rows])
            alerts[split][age] = {**confusion(all_rows, fired),
                                   'subMinuteFlagged': sum(int(v) for r, v in zip(all_rows, fired) if r['total'] < 60),
                                   'subMinuteScored': sum(r['total'] < 60 for r in all_rows)}
        for age in (300, 600):
            rr = data[split, age]
            # Descriptive, unadjusted associations; not a feature selection procedure.
            clues[split][age] = {name: confusion(rr, np.array([r['x'][k] >= t for r in rr]))
                                 for name, k, t in [('threeOpenTodos', 'todoOpen', 3), ('anyDispatch', 'delegates', 1),
                                    ('toolOpen60s', 'maxOpenAge', 60), ('anyError', 'failures', 1), ('debugLoop', 'debug', 1)]}
    return {'target': 'total active duration >= 1800s; completed prompts still running at fixed active age',
            'devPromptCountsByAge': [len(data['dev', a]) for a in AGES],
            'thresholds': {a: rounded(t) for a, t in thresholds.items()}, 'landmarks': landmarks,
            'sameLaterSurvivors': paired, 'everFlaggedByAge': alerts, 'descriptiveEvidence': clues}


if __name__ == '__main__':
    with open(sys.argv[1]) as f:
        print(json.dumps(analyze(map(json.loads, f)), indent=2, allow_nan=False))
