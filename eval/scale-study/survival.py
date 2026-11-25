"""Estimate a conditional residual-life distribution around the control forecast.
The compact ordinal model pools thresholds; the display decision is made from the
whole distribution, not just a regression on realized task length. A second fit
uses forecast errors from earlier completed prompts and this prompt's past landmarks.
"""
import json,sys
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from utility import features,make
from collections import Counter
AGE=[15,60,180,450,900]; KNOT=np.array([-3,-2,-1.4,-.9,-.45,0,.45,.9,1.4,2,2.7,3.5]); EDGES=np.exp(KNOT)

def context(rows,baseline):
    hist=[];anchors=[];out=[];sid=pid=None;last=None
    for r,b in zip(rows,baseline):
        if (r['sid'],r['pid'])!=(sid,pid):
            if sid!=r['sid']:hist=[]
            elif last is not None:
                total=last['elapsed']+last['y'];hist.append((total,anchors));hist=hist[-8:]
            anchors=[];sid=r['sid'];pid=r['pid']
        e=r['e'];v=[]
        for i,a in enumerate(AGE):
            if len(anchors)==i and e>=a:anchors.append((e,b))
            err=[np.clip(np.log((max(0,d-h[i][0])+30)/(h[i][1]+30)),-2.5,2.5) for d,h in hist if len(h)>i];
            v+=([np.mean(err[-4:]),err[-1],min(4,len(err))/4] if err else [0,0,0]);
            v+=[min(3,max(-3,np.log((b+30)/(anchors[i][1]+30)))) if len(anchors)>i else 0];
        out.append(v);last=r
    return np.array(out)

def fit(z,rows,b,reg):
    cnt=Counter((v['sid'],v['pid']) for v in rows);w=np.array([1/cnt[v['sid'],v['pid']] for v in rows]);y=np.array([r['y'] for r in rows]);d=y+np.array([r['elapsed'] for r in rows]);
    # Some oversampling of long jobs lets the tail shape be learned. It is the same
    # weighting at every landmark of a prompt; heldout labels never enter prediction.
    w*=1+.4*(d>=900)+.5*(d>=1800);w/=w.sum()/len(cnt);mu=np.average(z,axis=0,weights=w);sd=np.maximum(.20,np.sqrt(np.average((z-mu)**2,axis=0,weights=w)));z=np.clip((z-mu)/sd,-4,4)
    truth=(np.log((y+30)/(b+30))[:,None]>KNOT).astype(float);D=z.shape[1];K=len(KNOT)
    # Non-proportional odds: the tail and the center may have different drivers.
    mix=np.tanh(KNOT);init=np.r_[np.zeros(2*D),-KNOT];
    def loss(v):
        beta,skew,inter=np.split(v,[D,D*2]);h=(z@beta)[:,None]+(z@skew)[:,None]*mix+inter;p=expit(h);res=(p-truth)*w[:,None]/K
        return (np.logaddexp(0,(1-2*truth)*h)*w[:,None]).sum()/K+reg*(beta@beta+3*(skew@skew))/2,np.r_[z.T@res.sum(axis=1)+reg*beta,z.T@(res@mix)+reg*3*skew,res.sum(axis=0)]
    opt=minimize(loss,init,jac=True,method='L-BFGS-B',options={'maxiter':90,'ftol':1e-8});return {'mu':mu.tolist(),'sd':sd.tolist(),'params':opt.x.tolist()}

def predict(m,z,rows,b):
    z=np.clip((z-m['mu'])/m['sd'],-4,4);D=z.shape[1];beta,skew,inter=np.split(m['params'],[D,2*D]);cdf=np.maximum.accumulate(expit(-((z@beta)[:,None]+(z@skew)[:,None]*np.tanh(KNOT)+inter)),axis=1);edges=np.r_[-4,KNOT,4.3];prob=np.diff(np.column_stack((np.zeros(len(z)),cdf,np.ones(len(z)))),axis=1);mid=np.exp((edges[:-1]+edges[1:])*.5);e=np.array([r['e'] for r in rows]);t=e[:,None]+np.maximum(0,(b[:,None]+30)*mid-30);vals=[]
    for long,tail in [(0,0),(1,1.5),(2.5,3.5),(4,6)]:
        p=prob*(1+long*expit((t-900)/120)+tail*expit((t-1800)/240));cs=np.cumsum(p,axis=1);goal=cs[:,-1]*.5;idx=(cs<goal[:,None]).sum(axis=1);rr=np.arange(len(b));frac=1-(cs[rr,idx]-goal)/np.maximum(1e-9,p[rr,idx]);v=np.maximum(0,(b+30)*np.exp(edges[idx]+frac*(edges[idx+1]-edges[idx]))-30);vals.append(v)
    return vals

if __name__=='__main__':
    root=Path(sys.argv[1]);f=int(sys.argv[2]);r,b,valid=make(root,f);z=features(r,b);models=[];vals=[]
    for use_context in [False,True]:
        X=np.column_stack((z,context(r,b))) if use_context else z
        for reg in [8,35]:
            m=fit(X,r,b,reg);models.append(m)
            if valid:
                rr,bb=valid;xx=features(rr,bb);xx=np.column_stack((xx,context(rr,bb))) if use_context else xx;vals+=predict(m,xx,rr,bb)
    (root/f'survival{f}.json').write_text(json.dumps(models,separators=(',',':')))
    if valid:(root/f'survival-oof{f}.json').write_text(json.dumps(np.stack(vals,axis=1).round(4).tolist(),separators=(',',':')))
