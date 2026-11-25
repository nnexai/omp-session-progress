"""Regularized residual in *minutes*: for each prompt bracket the actual cost is
absolute ETA error, not log error. Anchor all unsupported corrections at zero.
Use nested control forecasts, including during fit of the retrospective policy.
"""
import json,sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from utility import features,make
from collections import Counter

# Low-dimensional evidence basis, minimizing dependence on sparse text categories.
def basis(rows,b):
    x=np.array([r['x'] for r in rows]);e=np.array([r['e'] for r in rows]);le=np.log1p(e/60);lb=np.log1p(b/60);q=lambda a:np.log1p(np.maximum(0,a));
    return np.column_stack((le,lb,lb-le,np.maximum(0,le-1.3),np.maximum(0,le-2.7),np.maximum(0,lb-1.8),np.minimum(3,x[:,7]),x[:,3],x[:,6],x[:,10],x[:,16],np.minimum(3,x[:,18]),x[:,23]-le,x[:,25],x[:,28],x[:,31],x[:,33],x[:,35],x[:,37],q(x[:,46]),x[:,48],x[:,49],x[:,52],x[:,55]-le,x[:,59],x[:,61],x[:,62],x[:,63],x[:,64],x[:,67],x[:,68],x[:,71],x[:,73],x[:,74],x[:,76],q(x[:,78]),q(x[:,81]),q(x[:,104]),x[:,118],x[:,128],x[:,134],x[:,137],x[:,138],x[:,140:164].sum(axis=1)))

def fit(z,rows,b):
    cnt=Counter((r['sid'],r['pid']) for r in rows);w=np.array([1/cnt[r['sid'],r['pid']] for r in rows]);e=np.array([r['elapsed'] for r in rows]);y=np.array([r['y'] for r in rows]);d=e+y;mu=np.average(z,axis=0,weights=w);sd=np.maximum(.2,np.sqrt(np.average((z-mu)**2,axis=0,weights=w)));z=np.column_stack((np.ones(len(z)),np.clip((z-mu)/sd,-3,3)));params=[]
    for weights in ([.42,.25,.21,.12],[.25,.30,.25,.20],[.12,.30,.32,.26]):
        masks=[np.ones(len(y),bool),(d>=180)&(d<900),d>=900,d>=1800];wt=sum(a*w*m/(w*m).sum() for a,m in zip(weights,masks));a=z*((b+30)/60)[:,None];offset=(b-y)/60
        for reg in [1.5,6,20]:
            def obj(v):
                r=offset+a@v;loss=np.sqrt(.06**2+r**2);return wt@loss+reg*(v@v)/2,a.T@(wt*r/loss)+reg*v
            opt=minimize(obj,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':90,'ftol':1e-9});params.append(opt.x.tolist())
    return dict(mu=mu.tolist(),sd=sd.tolist(),params=params)

def pred(m,z,b):
    z=np.column_stack((np.ones(len(b)),np.clip((z-m['mu'])/m['sd'],-3,3)));return np.maximum(0,b[:,None]+(b[:,None]+30)*np.clip(z@np.array(m['params']).T,-.8,3))
if __name__=='__main__':
    root=Path(sys.argv[1]);f=int(sys.argv[2]);r,b,valid=make(root,f);m=fit(basis(r,b),r,b);(root/f'minutes{f}.json').write_text(json.dumps(m,separators=(',',':')))
    if valid:
        rr,bb=valid;ps=pred(m,basis(rr,bb),bb);(root/f'minutes-oof{f}.json').write_text(json.dumps(ps.round(4).tolist(),separators=(',',':')))
