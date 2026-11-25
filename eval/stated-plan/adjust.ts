import type { Band, Prefix } from '../milestones/types';
import type { Plans, Route, Step } from './types';

export const inputs=['bag','prose','checklist','ordered'] as const;
export type Input=typeof inputs[number];
const steps:Step[]=['edit','verify','delegate','release','inspect','report','other'];
const bands:Band[]=[0,1,2,3];
export interface FitRow { p:Prefix; plan:Plans; probs:number[]; gold:Band; weight:number }
export interface Adjustment { predict(p:Prefix,plan:Plans,probs:number[]):number[] }
function routeFeatures(r:Route|null,ordered:boolean,verifying:boolean){
  const seq=r?.steps??[],check=seq.indexOf('verify');
  const after=check<0?[]:seq.slice(check+1);
  // The bag arm sees source, length, conditionality and exactly the same stage
  // vocabulary, but no ordering, interpretation of the tail, or stage positions.
  const x=[Number(!!r),Math.log1p(seq.length),Number(!!r?.conditional),...steps.map(s=>Math.log1p(seq.filter(v=>v===s).length))];
  if(ordered)x.push(Number(r?.tail==='final'),Number(r?.tail==='more'),...steps.map(s=>Number(seq[0]===s)),...steps.map(s=>Number(seq.at(-1)===s)),...steps.map(s=>Number(after.includes(s))));
  return [...x,...x.map(v=>verifying?v:0)];
}
function features(p:Prefix,r:Plans,arm:Input){
  const prose=arm==='checklist'?null:r.prose, todo=arm==='prose'?null:r.checklist;
  if(!prose&&!todo)return [];
  return [...routeFeatures(prose,arm!=='bag',p.verifying),...routeFeatures(todo,arm!=='bag',p.verifying)];
}
function predict(x:number[],base:number[],coef:number[][]){
  const z=coef.map((c,k)=>{let v=Math.log(Math.max(1e-12,base[k]!));for(let j=0;j<x.length;j++)v+=c[j]!*x[j]!;return v;});
  const high=Math.max(...z);const a=z.map(v=>Math.exp(v-high)),sum=a.reduce((s,n)=>s+n,0);return a.map(v=>v/sum);
}
// Offset the prior classifier's log odds, trained against actual milestone
// labels, not minutes. No route = exactly the old probabilities. Keep the ETA
// head fixed so a difference cannot be attributed to refitting the clock.
export function fitAdjustment(rows:FitRow[],arm:Input):Adjustment{
  const train=rows.map(r=>({x:features(r.p,r.plan,arm),r})).filter(v=>v.x.length);
  if(!train.length)return {predict:(_p,_r,probs)=>probs};
  const d=train[0]!.x.length,coef=bands.map(()=>new Float64Array(d)),total=train.reduce((s,v)=>s+v.r.weight,0);
  for(let i=0;i<140;i++){
    const grad=bands.map(()=>new Float64Array(d));
    for(const {x,r}of train){
      const logits=bands.map(k=>{let v=Math.log(Math.max(1e-12,r.probs[k]!));for(let j=0;j<d;j++)v+=coef[k]![j]!*x[j]!;return v;});
      const max=Math.max(...logits),exps=logits.map(v=>Math.exp(v-max)),sum=exps.reduce((s,n)=>s+n,0);
      for(let j=0;j<d;j++){if(x[j]===0)continue;const w=x[j]!*r.weight/total;for(const k of bands)grad[k]![j]!+=w*(exps[k]!/sum-Number(k===r.gold));}
    }
    for(const k of bands)for(let j=0;j<d;j++)coef[k]![j]!-=.25*(grad[k]![j]!+.025*coef[k]![j]!);
  }
  const fitted=coef.map(c=>Array.from(c));return {predict(p,r,probs){const x=features(p,r,arm);return x.length?predict(x,probs,fitted):probs;}};
}
