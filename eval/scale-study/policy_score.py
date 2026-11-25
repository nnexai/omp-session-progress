"""Score causal adjustments on development only; fail closed on invalid rows."""
import numpy as np,json,sys
from pathlib import Path
from collections import Counter

def select(root,kind):
    rows=[];preds=[]
    for f in range(3):
        g=[json.loads(l) for l in (root/f'dev{f}.jsonl').open()];p=json.loads((root/f'{kind}-oof{f}.json').read_text());assert len(g)==len(p);rows+=g;preds+=p
    b=np.array([v['b'] for v in rows]);y=np.array([v['y'] for v in rows]);e=np.array([v['e'] for v in rows]);d=np.array([v['elapsed'] for v in rows])+y;ps=np.array(preds);assert np.isfinite(ps).all()
    cnt=Counter((v['sid'],v['pid']) for v in rows);w=np.array([1/cnt[v['sid'],v['pid']] for v in rows]);groups=[d<60,(d>=180)&(d<900),d>=900,d>=1800,d>=0];
    def score(v):return np.array([np.average(np.abs(v[m]-y[m]),weights=w[m])/60 for m in groups]);
    control=score(b);out=[]
    for h,p in enumerate(ps.T):
        for weight in [.35,.65,1]:
            for onset in [0,60,180,450,900]:
                for forecast in ([False,True] if onset else [False]):
                    gate=1 if onset==0 else np.clip((e+forecast*np.minimum(600,b)*.5-onset)/onset,0,1);v=np.maximum(0,(b+30)*np.clip((p+30)/(b+30),.2,5)**(weight*gate)-30);diff=score(v)-control
                    ok=diff[0]<=.10 and diff[1]<=.20 and diff[4]<=.10 and (not np.any((d<60)&(v>=900)&(b<900)));out.append(dict(p=dict(head=h,weight=weight,onset=onset,useForecast=forecast),delta=diff.tolist(),priority=float(diff[1:4]@[.35,.45,.20]),ok=bool(ok)))
    out.sort(key=lambda q:q['priority']);best=next((a for a in out if a['ok']),None);print(kind,'control',control,'best',best,'unguarded',out[0]);(root/f'{kind}-selection.json').write_text(json.dumps({'best':best,'control':control.tolist(),'candidates':out}))
if __name__=='__main__':select(Path(sys.argv[1]),sys.argv[2])
