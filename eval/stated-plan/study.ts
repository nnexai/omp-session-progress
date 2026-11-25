import { readFileSync,writeFileSync } from 'node:fs';
import { join,resolve } from 'node:path';
import { fit } from '../milestones/model';
import type { Band, Example, Trial } from '../milestones/types';
import type { Entry } from '../tool-timing/cohort';
import reference from '../milestones/result.json';
import { frozenPlans } from './ingress';
import { sha } from '../frozen-session';
import { fitAdjustment,inputs,type FitRow } from './adjust';
import { summarize,type Reading,type Scored } from './summary';
import type { Plans } from './types';

interface Row extends Example {plan:Plans}
interface RouteTrial extends Trial {rows:Row[]}
const root=resolve(process.argv[2]??'.'),out=process.argv[3]; if(!out)throw Error('Usage: bun eval/stated-plan/study.ts FROZEN_MILESTONE_DIR NEW_AGGREGATE_JSON');
function files<T>(g:string,f:string):T{return JSON.parse(readFileSync(join(root,g,f),'utf8'));}
const foldNames=['fold0','fold1','fold2'];const entries=(g:string)=>files<Entry[]>(g,'entries.json');
const dev=foldNames.flatMap(entries),held=entries('test');
const setHash=(es:Entry[])=>sha(es.map(e=>e.digest).sort().join('\n'));
if(setHash(dev)!==reference.provenance.developmentContentSetSha256||setHash(held)!==reference.provenance.holdoutContentSetSha256||sha(readFileSync(join(root,'test/src/model-data.ts')))!==reference.provenance.refitWeightsSha256||new Set([...dev,...held].map(e=>e.id)).size!==dev.length+held.length)throw Error('Unexpected cohort or local control');
function load(name:string):RouteTrial[]{
  const cached:Trial[]=readFileSync(join(root,name,'milestones.jsonl'),'utf8').trim().split('\n').map(l=>JSON.parse(l));
  let i=0;const trials:RouteTrial[]=[];
  for(const e of entries(name)){
    const s=frozenPlans(e);if(s.id!==e.id)throw Error('Unexpected replay identity');
    for(const pts of s.prompts){const t=cached[i++];if(!t||t.session!==s.id||pts.length!==t.rows.length||t.missing)throw Error('Prompt or reading alignment changed');
      const rows=t.rows.map((r,j)=>{const p=pts[j]!;if(Math.abs(p.elapsedS-r.prefix.evidence.elapsedS)>1e-6)throw Error('Nonchronological replay');return {...r,plan:{prose:p.prose,checklist:p.checklist}};});trials.push({...t,rows});
    }
  }
  if(i!==cached.length)throw Error('Missing recorded prompt');return trials;
}
// Identical sampling order, prompt weighting and targets to the original stage
// study. Fold-local weights are recomputed when training an offset's OOF base.
function samples(ts:RouteTrial[]){
  const groups:[Row,number][][]=[[],[],[]],counts=[0,0,0];
  for(const t of ts){const n=Math.min(32,t.rows.length);if(!n)continue;const k=t.duration<3?0:t.duration<15?1:2;counts[k]!++;
    for(let i=0;i<n;i++)groups[k]!.push([t.rows[Math.floor((i+.5)*t.rows.length/n)]!,1/n]);}
  const total=groups.reduce((s,g)=>s+g.length,0);return groups.flatMap((g,k)=>g.map(([r,w])=>({row:r,prefix:r.prefix,band:r.gold.band,remainingMin:r.gold.remainingMin,weight:w*total*[.30,.35,.35][k]!/counts[k]!})));
}
const parts=foldNames.map(load),training=samples(parts.flat()),oof=new Map<Row,number[]>();
for(let i=0;i<parts.length;i++){
  const base=fit(samples(parts.filter((_,j)=>i!==j).flat()));
  for(const s of samples(parts[i]!))oof.set(s.row,base.forecast(s.prefix).probabilities);
}
const fitting:FitRow[]=training.map(s=>({p:s.prefix,plan:s.row.plan,probs:oof.get(s.row)!,gold:s.band,weight:s.weight}));
const base=fit(training),adjusters=inputs.map(arm=>fitAdjustment(fitting,arm));
console.log(`Frozen stage classifier and ${adjusters.length} plan ablations on ${dev.length} development sessions`);
const test=load('test'),scored:Scored[]=test.map(t=>({session:t.session,duration:t.duration,rows:t.rows.map((r):Reading=>{
  const p=r.prefix,prior=base.forecast(p),prob=[prior.probabilities,...adjusters.map(adj=>adj.predict(p,r.plan,prior.probabilities))];
  const choices=([0,1,2,3] as Band[]).map(b=>base.oracle(p,b));
  const eta=prob.map(ps=>ps.reduce((s,v,i)=>s+v*choices[i]!,0));
  for(const ps of prob)if(ps.some(x=>!Number.isFinite(x)||x<0)||Math.abs(1-ps.reduce((a,b)=>a+b,0))>1e-6)throw Error('Invalid band probabilities');
  return {p,plan:r.plan,prob,eta,remaining:r.gold.remainingMin,band:r.gold.band};
})}));
const result=summarize(scored);
if(result.slices.all.readings!==reference.stage.slices.all.readings||Math.abs(result.slices.all.arms.base.mae-reference.stage.slices.all.mae.live)>1e-8||Math.abs(result.slices.verification.arms.base.precision!-reference.stage.slices.verification.classification.finalPrecision)>1e-8)throw Error('Baseline does not reproduce prior milestone experiment');
writeFileSync(out,JSON.stringify({provenance:reference.provenance,developmentSessions:dev.length,heldoutSessions:held.length,...result},null,2)+'\n',{flag:'wx',mode:0o600});console.log('Paired comparison complete; aggregate only.');
