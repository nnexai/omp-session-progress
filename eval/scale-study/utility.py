"""Fit a low-variance decision policy, not a duration regressor. Numeric inputs only.
Each head estimates the benefit of one incremental change to the control ETA,
with equal prompt weight in each priority stratum. Inputs are nested-cross-fitted.
"""
from pathlib import Path
import json, sys
import numpy as np
from collections import Counter
from scipy.special import expit

RATIOS=np.array([.45,.7,1,1.65,2.7,4.5]); CUTS=np.sqrt(RATIOS[:-1]*RATIOS[1:]); BASE=[1,3,6,7,8,11,16,18,21,22,23,27,28,30,31,32,33,37,42,45,46,48,49,52,54,55,59,61,62,63,64,66,67,68,69,70,71,72,73,74,78,81,88,93,96,102,104,110,112,118,121,122,124,128,134,137,138]

def features(rows,b):
    x=np.asarray([r['x'] for r in rows]); e=np.array([r['e'] for r in rows]);lb=np.log1p(b/60);le=np.log1p(e/60)
    primary=x[:,BASE];sm=x[:,140:]; main=np.column_stack((le,lb,lb-le,np.maximum(0,le-1.4),np.maximum(0,le-2.7),np.maximum(0,lb-1.6),primary,sm));return main

def arr(f):return [json.loads(l) for l in f.open()]
def make(root,f):
    train=[];pred=[];valid=None
    for i in range(3):
        rows=arr(root/f'dev{i}.jsonl'); b=np.array([r['b'] for r in rows]);
        if i==f:valid=(rows,b);continue
        if f!=3:b=np.array(json.loads((root/f'inner{3-i-f}-{i}.json').read_text()))
        if len(rows)!=len(b):raise ValueError('inner refit cadence mismatch')
        train+=rows;pred.extend(b)
    return train,np.array(pred),valid

def fit(rows,b):
    z=features(rows,b); y=np.array([r['y'] for r in rows]);d=np.array([r['elapsed']+r['y'] for r in rows]);cnt=Counter((r['sid'],r['pid']) for r in rows);w=np.array([1/cnt[r['sid'],r['pid']] for r in rows]);mu=np.average(z,axis=0,weights=w);sd=np.maximum(.15,np.sqrt(np.average((z-mu)**2,axis=0,weights=w)));z=np.column_stack((np.ones(len(b)),np.clip((z-mu)/sd,-4,4)));
    # Separate distributions for all tasks, 3–15, 15+ and 30+, not a cost proportional
    # to raw duration. Priors alter the cost of *error*, never the observed features.
    masks=[np.ones(len(w),bool),(d>=180)&(d<900),d>=900,d>=1800];params=[]
    target=np.stack([np.clip(((y+30)/(b+30)-l)/(h-l)*2-1,-1,1) for l,h in zip(RATIOS[:-1],RATIOS[1:])],axis=1)
    for priors in ([.62,.19,.14,.05],[.30,.25,.30,.15],[.15,.30,.30,.25]):
        wt=sum(w*m*pr/max(1,(w*m).sum()) for m,pr in zip(masks,priors));wt*=len(cnt);A=z.T@(z*wt[:,None]);U=z.T@(target*wt[:,None]);
        for penalty in [25,100]:
            ridge=np.eye(A.shape[0])*penalty;ridge[0,0]=.001;params.append(np.linalg.solve(A+ridge,U).T.tolist())
    return dict(mu=mu.tolist(),sd=sd.tolist(),base=BASE,coef=params,ratios=RATIOS.tolist())

def predict(rows,b,model):
    z=np.column_stack((np.ones(len(b)),np.clip((features(rows,b)-model['mu'])/model['sd'],-4,4)));vals=[]
    for coef in model['coef']:
        sign=z@np.array(coef).T; benefits=np.column_stack((np.zeros(len(b)),np.cumsum(sign*np.diff(RATIOS),axis=1)));
        # Soft action choice, relative to each reading's forecast scale.
        prob=np.exp(np.clip((benefits-benefits.max(axis=1,keepdims=True))/.15,-100,0));s=(prob@np.log(RATIOS))/prob.sum(axis=1);vals.append(np.maximum(0,(b+30)*np.exp(s)-30))
    return np.stack(vals,axis=1)

if __name__=='__main__':
    root=Path(sys.argv[1]);f=int(sys.argv[2]);rows,b,valid=make(root,f);m=fit(rows,b);(root/f'utility{f}.json').write_text(json.dumps(m,separators=(',',':')))
    if valid:
        rows,b=valid;out=predict(rows,b,m);(root/f'utility-oof{f}.json').write_text(json.dumps(out.round(4).tolist(),separators=(',',':')))
