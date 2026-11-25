// A compact residual surface: two causal coordinates (active age and current
// forecast), bounded log corrections, bilinear interpolation between octave knots.
// The extra 30 s keeps fractional errors at small etas from dominating the fit.
import type { Row } from './data';
type CalRow=Pick<Row,'sid'|'pid'|'e'|'b'|'elapsed'|'y'>;
const E=7,B=8;
const coord=(v:number,scale:number,n:number)=>Math.max(0,Math.min(n-1,Math.log2(Math.max(1,v/scale))));
export function fit(rows:CalRow[]):number[]{
  const groups=new Map<string,CalRow[]>();
  for(const r of rows){const key=`${r.sid}:${r.pid}`;let g=groups.get(key);if(!g){g=[];groups.set(key,g);}g.push(r);}
  const counts=[groups.size,0,0,0];
  for(const g of groups.values()){const d=g[0]!.elapsed+g[0]!.y;if(d>=180&&d<900)counts[1]!++;if(d>=900)counts[2]!++;if(d>=1800)counts[3]!++;}
  const points:{a:number;c:number;z:number;w:number;group:number}[]=[];
  let group=0;
  for(const g of groups.values()){
    const d=g[0]!.elapsed+g[0]!.y;
    const w=(0.20/counts[0]!+(d>=180&&d<900?0.35/counts[1]!:0)+(d>=900?0.30/counts[2]!:0)+(d>=1800?0.15/counts[3]!:0))/g.length;
    for(const r of g)if(r.e>60)points.push({a:coord(r.e,60,E),c:coord(r.b,30,B),z:Math.log((r.y+30)/(r.b+30)),w:w*(r.b+30),group});
    group++;
  }
  // Minimize absolute error in *seconds* for a multiplicative correction:
  // |(y+30) - m*(b+30)| = (b+30)*|ratio-m|. The weighted median solves it.
  points.sort((a,b)=>a.z-b.z);
  const corrections:number[]=[];
  const weight=(a:number,b:number,p:typeof points[number])=>p.w*Math.max(0,1-Math.abs(a-p.a)/1.5)*Math.max(0,1-Math.abs(b-p.c)/1.5);
  for(let a=0;a<E;a++)for(let b=0;b<B;b++){
    const wg=new Float64Array(groups.size);let total=0;
    for(const p of points){const w=weight(a,b,p);total+=w;wg[p.group]!+=w;}
    if(!total){corrections.push(0);continue;}
    let sum=0,median=0;for(const p of points){sum+=weight(a,b,p);if(sum>=total/2){median=p.z;break;}}
    const effective=total*total/wg.reduce((s,v)=>s+v*v,0);
    corrections.push(0.60*effective/(effective+8)*Math.max(-Math.log(2),Math.min(Math.log(2.4),median)));
  }
  return corrections;
}
export function predict(table:number[],e:number,b:number):number {
  if(e<=60)return b;
  const ac=coord(e,60,E),bc=coord(b,30,B),a=Math.floor(ac),c=Math.floor(bc),af=ac-a,cf=bc-c;
  const get=(i:number,j:number)=>table[Math.min(i,E-1)*B+Math.min(j,B-1)]!;
  const z=(1-af)*((1-cf)*get(a,c)+cf*get(a,c+1))+af*((1-cf)*get(a+1,c)+cf*get(a+1,c+1));
  return Math.max(0,(b+30)*Math.exp(z*Math.min(1,(e-60)/120))-30);
}
