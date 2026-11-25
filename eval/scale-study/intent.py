"""Does *what the agent is doing* predict mistakes after conditioning on the
existing ETA, age and measured route? Orthogonalize the intent basis first:
no blanket inflation on runs just because they have survived a while.
"""
import json,sys
from pathlib import Path
import numpy as np
from collections import Counter
from utility import make
from minutes import basis
from scipy.optimize import minimize

def fit(rows,b,ss):
    y=np.array([r['y'] for r in rows]);total=y+np.array([r['elapsed'] for r in rows]);groups=Counter((r['sid'],r['pid']) for r in rows);w=np.array([1/groups[r['sid'],r['pid']] for r in rows]);z=basis(rows,b);S=np.array([r['i'] for r in ss]);mu=np.average(z,axis=0,weights=w);sd=np.maximum(.3,np.sqrt(np.average((z-mu)**2,axis=0,weights=w)));Z=np.column_stack((np.ones(len(w)),np.clip((z-mu)/sd,-3,3)));den=Z.T@(Z*w[:,None])+np.eye(Z.shape[1])*8;proj=np.linalg.solve(den,Z.T@(w[:,None]*S));X=S-Z@proj;scale=np.maximum(.22,np.sqrt(np.average(X**2,axis=0,weights=w)));X=np.clip(X/scale,-4,4);res=(y-b)/60;a=X*((b+30)/60)[:,None];params=[]
    for mixture in [[.65,.16,.14,.05],[.30,.30,.25,.15]]:
        masks=[np.ones(len(w),bool),(total>=180)&(total<900),total>=900,total>=1800];wt=sum(c*w*m/(w*m).sum() for c,m in zip(mixture,masks));
        for reg in [3,12,42]:
            def obj(v):
                r=a@v-res;loss=np.sqrt(.1**2+r**2);return wt@loss+reg*(v@v)/2,a.T@(wt*r/loss)+reg*v
            v=minimize(obj,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':100,'ftol':1e-8}).x;params.append(v.tolist())
    return dict(mu=mu.tolist(),sd=sd.tolist(),proj=proj.tolist(),scale=scale.tolist(),params=params)

def predict(m,r,b,ss):
    Z=np.column_stack((np.ones(len(b)),np.clip((basis(r,b)-m['mu'])/m['sd'],-3,3)));S=np.array([v['i'] for v in ss]);X=np.clip((S-Z@np.array(m['proj']))/m['scale'],-4,4);return np.maximum(0,b[:,None]+(b[:,None]+30)*np.clip(X@np.array(m['params']).T,-.80,2.5))
if __name__=='__main__':
    root=Path(sys.argv[1]);f=int(sys.argv[2]);r,b,valid=make(root,f);s=[];v=None
    for i in range(3):
        ss=json.loads((root/f'intent{i}.json').read_text());
        if i==f:v=ss
        else:s+=ss
    assert len(s)==len(r);m=fit(r,b,s);(root/f'intent-model{f}.json').write_text(json.dumps(m,separators=(',',':')))
    if valid:
        rr,bb=valid;vals=predict(m,rr,bb,v);(root/f'intent-oof{f}.json').write_text(json.dumps(vals.round(4).tolist(),separators=(',',':')))
