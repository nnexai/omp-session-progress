import type { Band, Prefix } from '../milestones/types';
import { inputs } from './adjust';
import type { Plans } from './types';
export interface Reading {p:Prefix;plan:Plans;prob:number[][];eta:number[];remaining:number;band:Band}
export interface Scored {session:string;duration:number;rows:Reading[]}
const arms=['base',...inputs] as const;
type Arm=typeof arms[number];
const filters={all:()=>true,medium:(t:Scored)=>t.duration>=3&&t.duration<15,long:(t:Scored)=>t.duration>=15,'30+':(t:Scored)=>t.duration>=30,short:(t:Scored)=>t.duration<1,slow:(_:Scored,r:Reading)=>r.p.evidence.slowTool,verification:(_:Scored,r:Reading)=>r.p.verifying,verificationWithPlan:(_:Scored,r:Reading)=>r.p.verifying&&!!(r.plan.prose||r.plan.checklist)} satisfies Record<string,(t:Scored,r:Reading)=>boolean>;
const sum=(a:number[])=>a.reduce((s,v)=>s+v,0),avg=(a:number[])=>sum(a)/a.length;
const ratio=(a:number,b:number)=>b?a/b:null;
function metrics(rows:Reading[],i:number){let mae=0,brier=0,tp=0,fp=0,fn=0,major=0,over=0;
  for(const r of rows){const pred=r.eta[i]!,err=Math.abs(pred-r.remaining),actual=r.band===0,p=r.prob[i]![0]!,call=p>=.5;
    mae+=err;major+=Number(err>=15);over+=Number(pred>=15);brier+=(p-Number(actual))**2;
    tp+=Number(call&&actual);fp+=Number(call&&!actual);fn+=Number(!call&&actual);}
  const n=rows.length;return {mae:mae/n,brier:brier/n,tp:tp/n,fp:fp/n,fn:fn/n,major:major/n,over:over/n};
}
const sources=['prose','checklist','either'] as const;
function coverage(ts:Scored[]){
  return Object.fromEntries(sources.map(source=>{
    let prompts=0,readings=0,conflicts=0,finalRight=0,finalWrong=0,moreRight=0,moreWrong=0;
    const handoff={right:0,wrong:0,prompts:0},subsequentWork={right:0,wrong:0,prompts:0};
    for(const t of ts){let hit=false,handoffHit=false,workHit=false;for(const {plan,band} of t.rows){const routes=(source==='either'?[plan.prose,plan.checklist]:[plan[source]]).filter(r=>r!==null);
      if(!routes.length)continue;readings++;hit=true;
      const patterns=new Set(routes.filter(r=>!r.conditional).map(r=>{
        const v=r.steps.indexOf('verify');if(v<0)return null;
        const after=r.steps.slice(v+1);
        return after.some(s=>s==='edit'||s==='delegate'||s==='release')?'work':after.length&&after.every(s=>s==='report')?'handoff':null;
      }).filter(Boolean));
      if(patterns.size===1&&patterns.has('handoff')){handoffHit=true;handoff[band===0?'right':'wrong']++;}
      if(patterns.size===1&&patterns.has('work')){workHit=true;subsequentWork[band!==0?'right':'wrong']++;}
      const claims=new Set(routes.filter(r=>!r.conditional&&r.tail!=='unknown').map(r=>r.tail));
      if(claims.size>1)conflicts++;else if(claims.has('final')){if(band===0)finalRight++;else finalWrong++;}
      else if(claims.has('more')){if(band!==0)moreRight++;else moreWrong++;}}
      if(hit)prompts++;if(handoffHit)handoff.prompts++;if(workHit)subsequentWork.prompts++;}
    return [source,{prompts,readings,conflicts,finalRight,finalWrong,moreRight,moreWrong,handoff,subsequentWork}];
  }));
}
function byPrompt(ts:Scored[]){return ts.map(t=>({id:t.session,m:arms.map((_,i)=>metrics(t.rows,i)),local:avg(t.rows.map(r=>Math.abs(r.p.evidence.localMin-r.remaining)))}));}
type Group=ReturnType<typeof byPrompt>;
function paired(groups:Group[],ids:string[],arm:Arm,field:'mae'|'brier',vs:'base'|'local'|'bag'='base'){
  if(groups.some(g=>!g.length))return null;
  const by=groups.map(g=>{const sums=Array(ids.length).fill(0),counts=Array(ids.length).fill(0);let total=0;
    for(const t of g){const d=t.m[arms.indexOf(arm)]![field]-(vs==='local'?t.local:t.m[arms.indexOf(vs)]![field]);const k=ids.indexOf(t.id);sums[k]+=d;counts[k]++;total+=d;}
    return {sums,counts,estimate:total/g.length};});
  const deltas:number[]=[];let seed=481121;
  for(let n=0;n<3000;n++){const sums=by.map(()=>0),counts=by.map(()=>0);
    for(let j=0;j<ids.length;j++){seed=(Math.imul(1664525,seed)+1013904223)>>>0;const k=Math.floor(seed/4294967296*ids.length);
      for(let i=0;i<by.length;i++){sums[i]!+=by[i]!.sums[k]!;counts[i]!+=by[i]!.counts[k]!;}}
    if(counts.every(x=>x>0))deltas.push(avg(sums.map((s,i)=>s/counts[i]!)));}
  deltas.sort((a,b)=>a-b);return {delta:avg(by.map(g=>g.estimate)),ci95:[deltas[Math.floor(.025*deltas.length)],deltas[Math.floor(.975*deltas.length)]]};
}
function table(group:Group,i:number){const m=(key:keyof ReturnType<typeof metrics>)=>avg(group.map(t=>t.m[i]![key]));const tp=m('tp'),fp=m('fp'),fn=m('fn');
  return {mae:m('mae'),majorMiss:m('major'),forecast15:m('over'),brier:m('brier'),precision:ratio(tp,tp+fp),recall:ratio(tp,tp+fn)};}
function slice(ts:Scored[],ids:string[]){const g=byPrompt(ts);const armsTable=Object.fromEntries(arms.map((a,i)=>[a,table(g,i)])) as Record<Arm,ReturnType<typeof table>>;
  return {sessions:new Set(ts.map(t=>t.session)).size,prompts:ts.length,readings:sum(ts.map(t=>t.rows.length)),coverage:coverage(ts),arms:armsTable,orderedVsBase:paired([g],ids,'ordered','mae'),orderedVsBaseBrier:paired([g],ids,'ordered','brier')};}
export function summarize(scored:Scored[]){
  const ids=[...new Set(scored.map(t=>t.session))];
  const subset=(filter:(t:Scored,r:Reading)=>boolean)=>scored.map(t=>({...t,rows:t.rows.filter(r=>filter(t,r))})).filter(t=>t.rows.length);
  const slices=Object.fromEntries(Object.entries(filters).map(([name,filter])=>[name,slice(subset(filter),ids)])) as Record<keyof typeof filters,ReturnType<typeof slice>>;
  const groups=[byPrompt(subset(filters.medium)),byPrompt(subset(filters.long))];
  return {completed:scored.length,slices,balanced:{mae:Object.fromEntries(arms.map(a=>[a,(slices.medium.arms[a].mae+slices.long.arms[a].mae)/2])),orderedVsBase:paired(groups,ids,'ordered','mae'),orderedVsBag:paired(groups,ids,'ordered','mae','bag'),orderedVsLocal:paired(groups,ids,'ordered','mae','local')}};
}
