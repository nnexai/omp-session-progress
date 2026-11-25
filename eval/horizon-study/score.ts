import type { Row } from './data';
export type Observation={sid:number;pid:number;elapsed:number;y:number;slow:boolean;v:number|null;lo?:number|null;hi?:number|null};
export function score(rows:Observation[]){
  const groups=new Map<string, {total:number;points:Observation[]}>();
  for(const r of rows){const k=`${r.sid}:${r.pid}`;let g=groups.get(k);if(!g){g={total:r.elapsed+r.y,points:[]};groups.set(k,g);}g.points.push(r);}
  const summary=(accept:(total:number)=>boolean,slow=false)=>{
    let prompts=0,points=0,mae=0,miss=0,over15=0,over30=0,prem=0,missing=0,covered=0,ranges=0;
    for(const {total,points:all} of groups.values()){
      if(!accept(total))continue;
      let n=0,err=0,large=0,long=0,extreme=0,early=0,hit=0,intervals=0;
      for(const r of all){if(slow&&!r.slow)continue;if(r.v===null){missing++;continue;}if(!Number.isFinite(r.v)||r.v<0)throw Error('Invalid ETA');const d=Math.abs(r.v-r.y);n++;err+=d;large+=Number(d>=900);long+=Number(r.v>=900);extreme+=Number(r.v>=1800);early+=Number(r.elapsed/(r.elapsed+r.v)>=0.85&&r.y/total>0.35);if(r.lo!=null&&r.hi!=null){intervals++;hit+=Number(r.y>=r.lo&&r.y<=r.hi);}}
      if(n){prompts++;points+=n;mae+=err/n;miss+=large/n;over15+=long/n;over30+=extreme/n;prem+=early/n;if(intervals){covered+=hit/intervals;ranges++;}}
    }
    return {prompts,points,missing,maeMin:mae/Math.max(1,prompts)/60,miss15:miss/Math.max(1,prompts),over15:over15/Math.max(1,prompts),over30:over30/Math.max(1,prompts),premature:prem/Math.max(1,prompts),coverage:ranges?covered/ranges:null};
  };
  return {all:summary(()=>true),short:summary(t=>t<60),medium:summary(t=>t>=180&&t<900),long:summary(t=>t>=900),halfHour:summary(t=>t>=1800),slow:summary(()=>true,true)};
}
// Long and 30+ overlap intentionally: heavier penalty for getting the longest tasks wrong.
export type Scores=ReturnType<typeof score>;
export const priority=(s:Scores)=>0.35*s.medium.maeMin+0.45*s.long.maeMin+0.20*s.halfHour.maeMin;
export const eligible=(base:Scores,c:Scores)=>c.medium.maeMin<=base.medium.maeMin+0.20 && c.all.maeMin<=base.all.maeMin+0.10 && c.short.maeMin<=base.short.maeMin+0.10 && c.short.over15<=base.short.over15 && c.all.missing===base.all.missing;
export function input({x,e,b,upper,tools,todo}:Row){return {x,e,b,upper,tools,todo};}
