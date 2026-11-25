"""Use the estimated chance of a long route, not a uniform long-task premium.
Guard short jobs by preserving their control estimate when the conditional
survival distribution doesn't support a correction.
"""
from survival import *

def policies(m,z,rows,b):
    z=np.clip((z-m['mu'])/m['sd'],-4,4);D=z.shape[1];beta,skew,inter=np.split(m['params'],[D,2*D]);cdf=np.maximum.accumulate(expit(-((z@beta)[:,None]+(z@skew)[:,None]*np.tanh(KNOT)+inter)),axis=1);edges=np.r_[-4,KNOT,4.3];p=np.diff(np.column_stack((np.zeros(len(z)),cdf,np.ones(len(z)))),axis=1);mid=np.exp((edges[:-1]+edges[1:])*.5);e=np.array([r['e'] for r in rows]);t=e[:,None]+np.maximum(0,(b[:,None]+30)*mid-30);L=expit((t-900)/120);H=expit((t-1800)/240);probL=(p*L).sum(axis=1);probH=(p*H).sum(axis=1);vals=[]
    for q,scale,gate in [(.5,3,.15),(.5,5,.25),(.55,5,.20),(.6,7,.30)]:
        # A scaling of the asymmetric cost only where posterior evidence supports it.
        strength=np.clip((probL+.5*probH-gate)/.5,0,1);mass=p*(1+scale*strength[:,None]*(L+H));cs=np.cumsum(mass,axis=1);goal=cs[:,-1]*q;idx=(cs<goal[:,None]).sum(axis=1);rr=np.arange(len(b));frac=1-(cs[rr,idx]-goal)/np.maximum(1e-9,mass[rr,idx]);v=np.maximum(0,(b+30)*np.exp(edges[idx]+frac*(edges[idx+1]-edges[idx]))-30)
        for soft in [False,True]:
            alpha=np.clip((probL+.5*probH-.15)/.55,0,1) if soft else 1;vals.append((b+30)*np.exp(alpha*np.log((v+30)/(b+30)))-30)
    return vals
if __name__=='__main__':
    root=Path(sys.argv[1]);f=int(sys.argv[2]);rows= [json.loads(l) for l in (root/f'dev{f}.jsonl').open()];b=np.array([r['b'] for r in rows]);ms=json.loads((root/f'survival{f}.json').read_text());z=features(rows,b);cx=context(rows,b);vals=[]
    for i,m in enumerate(ms):vals+=policies(m,np.column_stack((z,cx)) if i>=2 else z,rows,b)
    (root/f'posterior-oof{f}.json').write_text(json.dumps(np.stack(vals,axis=1).round(4).tolist(),separators=(',',':')))
