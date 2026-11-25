// Out-of-fold uncertainty relative to the center forecast, rather than two
// unconditional adjustments to it. Shrink toward the existing range in thin cells.
import type { Row } from './data';
import { gridCoordinate as coord, type RangeModel } from '../../src/range-calibration';
type R=Pick<Row,'e'|'b'|'y'|'sid'|'pid'>;
const A=6,B=6;
export function fit(rows:R[]):RangeModel{
  const sizes=new Map<string,number>();for(const r of rows){const k=`${r.sid}:${r.pid}`;sizes.set(k,(sizes.get(k)??0)+1);}
  const points=rows.filter(r=>r.e>60).map(r=>{const k=`${r.sid}:${r.pid}`;return {a:coord(r.e,90),b:coord(r.b,45),z:Math.log((30+r.y)/(30+r.b)),k,w:(30+r.b)/sizes.get(k)!};}).sort((x,y)=>x.z-y.z);
  const m:RangeModel={low:[],high:[],trust:[]};
  for(let i=0;i<A;i++)for(let j=0;j<B;j++){
    const wg=new Map<string,number>(),ws=new Float64Array(points.length);let total=0;
    for(let k=0;k<points.length;k++){const p=points[k]!,w=p.w*Math.max(0,1-Math.abs(i-p.a)/1.4)*Math.max(0,1-Math.abs(j-p.b)/1.4);ws[k]=w;if(w){total+=w;wg.set(p.k,(wg.get(p.k)??0)+w);}}
    let lo=0,hi=0,sum=0;for(let k=0;k<ws.length&&total>0;k++){const prev=sum;sum+=ws[k]!;if(prev<0.25*total&&sum>=0.25*total)lo=points[k]!.z;if(sum>=0.75*total){hi=points[k]!.z;break;}}
    const eff=total?total*total/[...wg.values()].reduce((s,w)=>s+w*w,0):0;
    m.low.push(Math.max(-2,Math.min(0,lo)));m.high.push(Math.min(2,Math.max(0,hi)));m.trust.push(0.75*eff/(eff+10));
  }
  return m;
}
