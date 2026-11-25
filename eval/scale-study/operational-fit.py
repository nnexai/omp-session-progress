"""Fit a small, family-balanced managed-work residual specialist; assess native DEV only.

All inputs and outputs are numeric and private. The native control is unchanged.
Managed actors are OOD training observations, never independent personal sessions.
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

root = Path(sys.argv[1])
if root.stat().st_mode & 0o077:
    raise ValueError('private root required')
def load(name):
    with (root / name).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]
aux = load('operational.jsonl')
dev = [load(f'dev{i}.jsonl') for i in range(3)]
# All inputs existed at the reading: age, control forecast, work and stage facts.
indices = [1, 3, 23, 30, 31, 32, 42, 48, 51, 70, 92, 101, 102]
def features(rows):
    x = np.asarray([r['x'] for r in rows]); age = np.array([r['e'] for r in rows]); b = np.array([r['b'] for r in rows])
    la = np.log1p(age/60); lb = np.log1p(b/60)
    return np.column_stack((la, lb, lb-la, x[:,indices]))
def fit(rows):
    z=features(rows); b=np.array([r['b'] for r in rows]); y=np.array([r['y'] for r in rows]);
    count=Counter((r['sid'],r['pid']) for r in rows)
    # Equal family mass and equal prompt mass within a family, not equal actors/checkpoints.
    family=defaultdict(set)
    for r in rows: family[r['sid']].add(r['pid'])
    w=np.array([1/(len(family[r['sid']])*count[(r['sid'],r['pid'])]) for r in rows]);w*=len(family)/w.sum()
    mu=np.average(z,axis=0,weights=w); sd=np.maximum(.2,np.sqrt(np.average((z-mu)**2,axis=0,weights=w)))
    z=np.column_stack((np.ones(len(z)),np.clip((z-mu)/sd,-3,3))); target=np.clip(np.log((y+30)/(b+30)),-3,3)
    # Smooth absolute loss targets a robust conditional median, with conservative shrinkage.
    def loss(coef):
        d=z@coef-target; delta=.3; s=np.sqrt(d*d+delta*delta); penalty=np.r_[0,coef[1:]]
        return np.sum(w*s)+20*(penalty@penalty)/2, z.T@(w*d/s)+20*penalty
    result=minimize(loss,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':150})
    return {'mu':mu.tolist(),'sd':sd.tolist(),'coef':result.x.tolist(),'success':bool(result.success),'families':len(family)}
def predict(rows,model):
    z=np.clip((features(rows)-model['mu'])/model['sd'],-3,3); return np.clip(np.column_stack((np.ones(len(z)),z))@model['coef'],-1.5,1.5)
def score(rows,pred):
    groups=defaultdict(list)
    for r,v in zip(rows,pred):groups[(r['sid'],r['pid'])].append((r,v))
    out={}
    for name,accept in [('all',lambda t:t>=0),('short',lambda t:t<60),('medium',lambda t:180<=t<900),('long',lambda t:t>=900),('halfHour',lambda t:t>=1800),('slow',lambda t:t>=0)]:
        vals=[];points=0
        for group in groups.values():
            total=group[0][0]['elapsed']+group[0][0]['y']
            if not accept(total):continue
            group=[(r,v) for r,v in group if name!='slow' or r['slow']]
            if not group:continue
            points+=len(group);vals.append((np.mean([abs(v-r['y']) for r,v in group])/60,np.mean([v>=900 for r,v in group]),np.mean([abs(v-r['y'])>=900 for r,v in group])))
        out[name]={'prompts':len(vals),'points':points,'maeMin':float(np.mean([v[0] for v in vals])) if vals else None,'over15':float(np.mean([v[1] for v in vals])) if vals else None,'miss15':float(np.mean([v[2] for v in vals])) if vals else None}
    return out
def evaluate(rows,residual):
    b=np.array([r['b'] for r in rows]); age=np.array([r['e'] for r in rows]); result={'control':score(rows,b)}
    for shrink in [.25,.5,1.0]:
        for onset in [0,180,600]:
            corrected=np.where(age>=onset,np.maximum(0,(b+30)*np.exp(shrink*residual)-30),b)
            result[f'{shrink}/{onset}']=score(rows,corrected)
    return result
# Family-heldout diagnostic is OOD managed-work only, not a personal evaluation.
pred=np.zeros(len(aux));family_ids=sorted({r['sid'] for r in aux})
for fold in range(3):
    valid={sid for i,sid in enumerate(family_ids) if i%3==fold};tr=[r for r in aux if r['sid'] not in valid];ix=[i for i,r in enumerate(aux) if r['sid'] in valid]
    if not tr or not ix:raise ValueError('insufficient distinct families')
    pred[ix]=predict([aux[i] for i in ix],fit(tr))
model=fit(aux);native=[evaluate(rows,predict(rows,model)) for rows in dev]
result={'managed_family_oof':evaluate(aux,pred),'native_dev_folds':native,'model':model,'features':indices,'assessment':'exploratory native DEV only; managed actors train-only OOD'}
with (root/'operational-results.json').open('x') as stream:json.dump(result,stream,separators=(',',':'),allow_nan=False)
for label,report in [('managed_family_oof',result['managed_family_oof'])]+[(f'dev{i}',v) for i,v in enumerate(native)]:
    print(label, json.dumps({k:{s:round(v[s]['maeMin'],3) if v[s]['maeMin'] is not None else None for s in ['all','short','medium','long','halfHour']} for k,v in report.items()},separators=(',',':')))
