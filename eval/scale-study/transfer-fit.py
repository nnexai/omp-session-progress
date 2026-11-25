#!/usr/bin/env python3
"""Local-only session-cross-fitted Codex prior, plus untouched OMP dev evaluation.

Run: bun eval/scale-study/transfer-extract.ts PRIVATE_ROOT
     PRIVATE_ROOT/venv/bin/python eval/scale-study/transfer-fit.py PRIVATE_ROOT
Only numeric model and aggregate results are produced; no transcript inputs here.
"""
import collections
import json
import math
import os
from pathlib import Path
import sys
import numpy as np

root = Path(sys.argv[1])
if root.stat().st_mode & 0o077:
    raise ValueError('Expected private root')
data = np.asarray(json.loads((root / 'transfer-data.json').read_text()), dtype=float)
if len(data) < 100:
    raise ValueError('Insufficient independent training prompts')

def matrix(rows):
    scope = rows[:, 1].astype(np.int64)
    return np.column_stack([np.ones(len(rows)), rows[:, 2]] + [((scope >> i) & 1) for i in range(24)])

def fit(rows):
    x = matrix(rows)
    sessions, counts = np.unique(rows[:, 0], return_counts=True)
    counts = dict(zip(sessions, counts))
    w = np.array([1 / counts[s] for s in rows[:, 0]])
    # Fixed regularization; length is on a log scale and categories are bits.
    reg = np.eye(x.shape[1]) * 5
    reg[0, 0] = 0.001
    coef = np.linalg.solve(x.T @ (w[:, None] * x) + reg, x.T @ (w * np.log1p(rows[:, 3])))
    residual = np.log1p(rows[:, 3]) - x @ coef
    # Weighted session-balanced residual empirical distribution, bounded numeric size.
    order = np.argsort(residual); residual = residual[order]; weights = w[order]
    quantiles = np.interp((np.arange(101)+0.5)/101, np.cumsum(weights)/weights.sum(), residual)
    return {'coefficients': coef.tolist(), 'residuals': quantiles.tolist()}

def predict(model, scope, logchars, elapsed):
    c = model['coefficients']; z = c[0] + c[1] * min(math.log1p(100000), max(0, logchars))
    z += sum(c[i+2] * ((int(scope) >> i) & 1) for i in range(24))
    threshold = math.log1p(max(0, elapsed))-z
    residuals = model['residuals']; survivors = [r for r in residuals if r > threshold]
    residual = survivors[(len(survivors)-1)//2] if survivors else threshold+0.5
    return max(0, math.expm1(min(16, z+residual))-elapsed)

def summary(groups):
    return {str(k): {'prompts':len(v), 'mean_abs_seconds':round(float(np.mean(v)),1)} for k,v in groups.items() if v}

# All folds are grouped by root; OOF estimates are never trained on their own session.
out = collections.defaultdict(list)
for fold in range(5):
    train = data[data[:, 0].astype(int)%5 != fold]; test = data[data[:, 0].astype(int)%5 == fold]
    model = fit(train)
    for _, scope, chars, total in test:
        times = np.linspace(5, total-0.01, min(40, max(1, int(total/15))))
        errors = [abs(predict(model,scope,chars,t)-(total-t)) for t in times]
        baseline = [abs(max(0, 180-t)-(total-t)) for t in times]
        band = 'short' if total<180 else 'medium' if total<900 else 'long'
        out[band+'_prior'].append(np.mean(errors)); out[band+'_fixed180'].append(np.mean(baseline))
model=fit(data)
encoded=json.dumps(model,separators=(',',':'))
fd=os.open(root/'transfer-model.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as f:f.write(encoded)
# The OMP model's x[140:164] is the same public scope vocabulary;
# x[63] is log prompt chars. Aggregate by prompt before averaging.
omp=collections.defaultdict(list)
for fold in range(3):
    rows=collections.defaultdict(list)
    with (root/f'dev{fold}.jsonl').open() as f:
        for line in f:
            r=json.loads(line); rows[(r['sid'],r['pid'])].append(r)
    for readings in rows.values():
        total=readings[0]['elapsed']+readings[0]['y']
        band='short' if total<180 else 'medium' if total<900 else 'long'
        for weight in (0,0.1,0.25,1):
            errors=[]
            for r in readings:
                scope=sum((int(r['x'][140+i])<<i) for i in range(24))
                p=predict(model,scope,r['x'][63],r['elapsed'])
                estimate=(1-weight)*r['b']+weight*p
                errors.append(abs(estimate-r['y']))
            omp[f'{band}_blend{weight}'].append(np.mean(errors))
print(json.dumps({'codex_oof':summary(out),'omp_dev':summary(omp),'artifact_bytes':len(encoded),'sessions':len(set(data[:,0])),'prompts':len(data)}))
