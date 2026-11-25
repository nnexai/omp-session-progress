// Predict the lifetime of a prompt that survived to this landmark, instead of
// fitting the volatile small remainder directly. Convert total to residual life
// before blending; duration strata and weights exist ONLY in training.
import { evalHead, trainGBDTHead, type GBDTHead } from '../../src/gbdt';
import type { Input,Row } from './data';
export function fit(rows:Row[]):GBDTHead {
  const groups=new Map<string,Row[]>();for(const r of rows){const key=`${r.sid}:${r.pid}`;let g=groups.get(key);if(!g){g=[];groups.set(key,g);}g.push(r);}
  const counts=[0,0,0,0];const band=(d:number)=>d<180?0:d<900?1:d<1800?2:3;
  for(const g of groups.values())counts[band(g[0]!.elapsed+g[0]!.y)]!++;
  const x:number[][]=[],y:number[]=[],anchors:number[]=[],e:number[]=[],w:number[]=[];
  for(const g of groups.values()){
    const total=g[0]!.elapsed+g[0]!.y,k=band(total),n=Math.min(24,g.length);
    for(let j=0;j<n;j++){
      const r=g[Math.floor((j+.5)*g.length/n)]!;
      x.push(r.x);y.push(Math.log(total+30));anchors.push(Math.log(r.e+30));e.push(r.e);
      w.push((0.40+0.60*groups.size/(4*counts[k]!))/n);
    }
  }
  return trainGBDTHead(x,y,anchors,e,w,{anchor:'elapsed',nTrees:120,lr:0.055,maxDepth:4,minLeaf:18,l2:2,colSample:0.6,maxBins:48,seed:527,quantile:0.55});
}
export function predict(head:GBDTHead,r:Input):number {
  if(r.e<=60)return r.b;
  const lifetime=(r.e+30)*Math.exp(evalHead(head,r.x))-30;
  const remaining=Math.max(0,lifetime-r.e),baseline=r.b+30;
  const ratio=Math.max(0.4,Math.min(3,(remaining+30)/baseline));
  return Math.max(0,baseline*Math.pow(ratio,0.45*Math.min(1,(r.e-60)/120))-30);
}
