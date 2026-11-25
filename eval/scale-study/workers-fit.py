#!/usr/bin/env python3
"""Cross-fit an auxiliary worker transfer against matched main-only CatBoost.

Run with the study venv: workers-fit.py PRIVATE_ROOT. Reads development folds
only. Worker folds are bound to their parent main; each worker parent has equal
auxiliary weight, and the auxiliary share is capped at 15% of training weight.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from catboost import CatBoostRegressor
from boost import export_head

root = Path(sys.argv[1])
if root.stat().st_mode & 0o077:
    raise ValueError('private root required')
def load(name):
    with (root / name).open() as f:
        return [json.loads(line) for line in f if line.strip()]
main = [load(f'dev{i}.jsonl') for i in range(3)]
workers = [load(f'workers{i}.jsonl') for i in range(3)]
def arrays(rows):
    x = np.array([r['x'] + [r['e']] for r in rows], dtype=np.float32)
    y = np.array([r['y'] for r in rows], dtype=np.float64)
    total = np.array([r['elapsed'] + r['y'] for r in rows])
    counts = Counter((r['sid'], r['pid']) for r in rows)
    w = np.array([1 / counts[(r['sid'], r['pid'])] for r in rows])
    return x, y, total, w

def main_weights(w, total):
    # Match the existing long-horizon priority recipe without checkpoint bias.
    out = .3 * w / w.sum()
    for mask, share in ((total >= 180, .2), (total >= 900, .3), (total >= 1800, .2)):
        den = w[mask].sum()
        if den: out[mask] += share * w[mask] / den
    return out / out.sum()

def worker_weights(rows, w):
    # Equalize parents before normalizing, rather than treating each worker as a person.
    sums = defaultdict(float)
    for row, weight in zip(rows, w): sums[row['sid']] += weight
    out = np.array([weight / sums[row['sid']] for row, weight in zip(rows, w)])
    return out / out.sum()

pred = {'control': [], 'main': [], 'worker': []}
observed = []
with TemporaryDirectory(dir=root) as tmp:
    for fold in range(3):
        train = [r for i in range(3) if i != fold for r in main[i]]
        aux = [r for i in range(3) if i != fold for r in workers[i]]
        if not aux: raise ValueError('no eligible auxiliary data in training folds')
        x, y, total, w = arrays(train)
        ax, ay, _, aw = arrays(aux)
        tw = main_weights(w, total)
        aw = worker_weights(aux, aw)
        xx = np.concatenate((x, ax)); yy = np.concatenate((y, ay))
        testx, _, _, _ = arrays(main[fold])
        for label, share in [('main', 0), ('worker', .15)]:
            path = root / f'workers-{label}-forest{fold}.json'
            if path.exists():
                head = json.loads(path.read_text())['heads'][0]
                raw = np.zeros(len(testx))
                for tree in head['trees']:
                    leaf = np.zeros(len(testx), dtype=int)
                    for bit, (feature, border) in enumerate(zip(tree['features'], tree['borders'])):
                        leaf |= (testx[:, feature] > border).astype(int) << bit
                    raw += np.asarray(tree['leaves'])[leaf]
                predicted = head['scale'] * raw + head['bias']
            else:
                weights = np.concatenate(((1-share)*tw, share*aw))
                model = CatBoostRegressor(loss_function='Quantile:alpha=0.55', iterations=600, depth=5,
                                          learning_rate=.045, thread_count=2, random_seed=42,
                                          verbose=False, allow_writing_files=False)
                model.fit(xx, np.log1p(yy), sample_weight=weights * len(weights))
                head = export_head(model, 0, Path(tmp) / 'model.json')
                with path.open('x') as out:
                    json.dump({'version':1,'featureCount':xx.shape[1],'heads':[head]},out,separators=(',',':'),allow_nan=False)
                predicted = model.predict(testx)
            pred[label].extend(np.maximum(0, np.expm1(predicted)).tolist())
        pred['control'].extend(r['b'] for r in main[fold])
        observed.extend(main[fold])

def score(values):
    groups = defaultdict(list)
    for row, v in zip(observed, values): groups[(row['sid'],row['pid'])].append((row,v))
    result={}
    for name, accept in [('all',lambda t:t>=0),('short',lambda t:t<60),('medium',lambda t:180<=t<900),('long',lambda t:t>=900),('halfHour',lambda t:t>=1800),('slow',lambda t:t>=0)]:
        prompts=[]; points=0
        for group in groups.values():
            total=group[0][0]['elapsed']+group[0][0]['y']
            if not accept(total):continue
            group=[(r,v) for r,v in group if name!='slow' or r['slow']]
            if not group:continue
            points+=len(group)
            prompts.append((sum(abs(v-r['y']) for r,v in group)/len(group)/60,
                            sum(v>=900 for r,v in group)/len(group),
                            sum(abs(v-r['y'])>=900 for r,v in group)/len(group)))
        result[name]={'prompts':len(prompts),'points':points,'maeMin':sum(p[0] for p in prompts)/max(1,len(prompts)),
                      'over15':sum(p[1] for p in prompts)/max(1,len(prompts)),
                      'miss15':sum(p[2] for p in prompts)/max(1,len(prompts))}
    return result

def priority(s):return .35*s['medium']['maeMin']+.45*s['long']['maeMin']+.2*s['halfHour']['maeMin']
control=np.array(pred['control']); baseline=score(control); result={'control':baseline,'models':{}}
age=np.array([r['e'] for r in observed])
for name in ('main','worker'):
    model=np.array(pred[name]); result['models'][name]={}
    for blend in (.35,.65,1.):
        for onset in (0,60,180,450):
            for forecast in ((False,True) if onset else (False,)):
                use=(age>=onset) | ((control>=onset) if forecast else False)
                s=score(np.where(use,control*(1-blend)+model*blend,control))
                s['priority']=priority(s)
                s['eligible']=bool(s['medium']['maeMin']<=baseline['medium']['maeMin']+.2 and s['all']['maeMin']<=baseline['all']['maeMin']+.1 and s['short']['maeMin']<=baseline['short']['maeMin']+.1 and s['short']['over15']<=baseline['short']['over15'])
                result['models'][name][f'{blend}/{onset}/{forecast}']=s
with (root/'workers-results.json').open('x') as out:json.dump(result,out,separators=(',',':'),allow_nan=False)
print(json.dumps(result,separators=(',',':')))
