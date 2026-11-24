import { readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fit, type MilestoneModel } from './model';
import type { Band, Example, Sample, Trial } from './types';
import reference from '../tool-timing/result.json';

const arms=['local','direct','binaryOracle','oracle','live','priorMix'] as const;
type Arm=typeof arms[number];
type Scores=Record<Arm,number>;
const armMap=(f:(a:Arm)=>number):Scores=>({local:f('local'),direct:f('direct'),binaryOracle:f('binaryOracle'),oracle:f('oracle'),live:f('live'),priorMix:f('priorMix')});
const names=['all','medium','long','30+','slow','short','verification','longFirstMinute'] as const;
type Slice=typeof names[number];
type Label='band'|'todoBand';
function load(file:string):Trial[]{return readFileSync(file,'utf8').trimEnd().split('\n').filter(Boolean).map(line=>JSON.parse(line));}
function eligible(e:Example,key:Label){return key==='band'||e.prefix.evidence.hasCurrentPlan;}

// Offline weights only: each prompt gets one vote within its stratum. Cap rows
// per prompt to avoid repeated long stationary readings dominating the fit.
function samples(trials:Trial[],key:Label){
  const groups:Sample[][]=[[],[],[]];const counts=[0,0,0];
  for(const t of trials){
    const rs=t.rows.filter(r=>eligible(r,key));if(!rs.length)continue;
    const k=t.duration<3?0:t.duration<15?1:2,n=Math.min(32,rs.length);counts[k]!++;
    for(let i=0;i<n;i++){
      const r=rs[Math.floor((i+.5)*rs.length/n)]!;
      groups[k]!.push({prefix:r.prefix,band:r.gold[key],remainingMin:r.gold.remainingMin,weight:1/n});
    }
  }
  const rows=groups.flat(),mix=[.30,.35,.35];
  for(let i=0;i<3;i++){
    if(!counts[i])throw Error('No training prompts in a required horizon');
    for(const row of groups[i]!)row.weight*=rows.length*mix[i]!/counts[i]!;
  }
  return {rows,counts};
}
interface Acc {n:number;error:Scores;major:Scores;over15:Scores; matrix:number[]; priorHits:number;finalBrier:number;priorBrier:number; yes:number;tp:number;fp:number;fn:number}
function empty():Acc{return {n:0,error:armMap(()=>0),major:armMap(()=>0),over15:armMap(()=>0),matrix:Array(16).fill(0),priorHits:0,finalBrier:0,priorBrier:0,yes:0,tp:0,fp:0,fn:0};}
interface Row{session:string;a:Acc}
function table(rows:Row[]){
  const mean=(f:(a:Acc)=>number)=>rows.length?rows.reduce((s,r)=>s+f(r.a)/r.a.n,0)/rows.length:null;
  const confusion=Array.from({length:4},(_,i)=>Array.from({length:4},(_,j)=>mean(a=>a.matrix[i*4+j]!)??0));
  const tp=mean(a=>a.tp)??0,fp=mean(a=>a.fp)??0,fn=mean(a=>a.fn)??0;
  const scores=(k:'error'|'major'|'over15')=>Object.fromEntries(arms.map(arm=>[arm,mean(a=>a[k][arm])]));
  return {sessions:new Set(rows.map(r=>r.session)).size,prompts:rows.length,readings:rows.reduce((s,r)=>s+r.a.n,0),mae:scores('error'),miss15:scores('major'),forecast15:scores('over15'),
    classification:{accuracy:mean(a=>a.matrix[0]!+a.matrix[5]!+a.matrix[10]!+a.matrix[15]!),priorAccuracy:mean(a=>a.priorHits),finalShare:mean(a=>a.yes),finalBrier:mean(a=>a.finalBrier),priorFinalBrier:mean(a=>a.priorBrier),finalPrecision:tp+fp?tp/(tp+fp):null,finalRecall:tp+fn?tp/(tp+fn):null,confusion}};
}
// Paired, prompt-weighted mean; session bootstrap keeps co-occurring horizons together.
function compare(groups:Row[][],sessions:string[],a:Arm,b:Arm){
  if(groups.some(rs=>!rs.length))return null;
  const K=sessions.length,index=new Map(sessions.map((s,i)=>[s,i]));
  const by=groups.map(rs=>{
    const sums=new Float64Array(K),ns=new Uint32Array(K);let sum=0;
    for(const r of rs){const i=index.get(r.session)!,v=(r.a.error[a]-r.a.error[b])/r.a.n;sums[i]!+=v;ns[i]!++;sum+=v;}
    return {sums,ns,delta:sum/rs.length};
  });
  const sample:number[]=[];let seed=481121;
  for(let j=0;j<3000;j++){
    const sums=new Float64Array(by.length),counts=new Uint32Array(by.length);
    for(let k=0;k<K;k++){seed=(Math.imul(1664525,seed)+1013904223)>>>0;const i=Math.floor(seed/4294967296*K);for(let g=0;g<by.length;g++){sums[g]!+=by[g]!.sums[i]!;counts[g]!+=by[g]!.ns[i]!;}}
    if(counts.every(n=>n>0))sample.push(sums.reduce((s,v,i)=>s+v/counts[i]!,0)/by.length);
  }
  sample.sort((a,b)=>a-b);
  return {delta:by.reduce((s,r)=>s+r.delta,0)/by.length,ci95:[sample[Math.floor(.025*sample.length)],sample[Math.floor(.975*sample.length)]]};
}
function evaluate(model:MilestoneModel,trials:Trial[],key:Label){
  const pool=new Map<Slice,Row[]>(names.map(n=>[n,[]]));
  const priorBest=model.prior.indexOf(Math.max(...model.prior));
  for(const t of trials){const acc=new Map<Slice,Acc>();
    for(const r of t.rows){if(!eligible(r,key))continue;
      const p=r.prefix;
      // Live arms receive ONLY the prefix; neither sees future work, outcome, or duration.
      const live=model.forecast(p),probs=live.probabilities,chosen=probs.indexOf(Math.max(...probs));
      const possible=([0,1,2,3] as Band[]).map(b=>model.oracle(p,b));
      const direct=model.direct(p),priorMix=possible.reduce((s,v,i)=>s+v*model.prior[i]!,0);
      const gold=r.gold[key],truth=r.gold.remainingMin;
      const preds:Scores={local:p.evidence.localMin,direct,live:live.eta,priorMix,oracle:possible[gold]!,binaryOracle:model.oracleFinal(p,gold===0)};
      if(Object.values(preds).some(v=>!Number.isFinite(v)||v<0)||probs.length!==4||Math.abs(probs.reduce((s,v)=>s+v,0)-1)>1e-6)throw Error('Invalid forecast');
      const slices:Slice[]=['all'];if(t.duration>=3&&t.duration<15)slices.push('medium');if(t.duration>=15)slices.push('long');if(t.duration>=30)slices.push('30+');if(t.duration<1)slices.push('short');if(p.evidence.slowTool)slices.push('slow');if(p.verifying)slices.push('verification');if(t.duration>=15&&p.evidence.elapsedS<=60)slices.push('longFirstMinute');
      for(const s of slices){let z=acc.get(s);if(!z){z=empty();acc.set(s,z);}z.n++;
        for(const a of arms){const err=Math.abs(preds[a]-truth);z.error[a]+=err;z.major[a]+=Number(err>=15);z.over15[a]+=Number(preds[a]>=15);}
        z.matrix[4*gold+chosen]!++;z.priorHits+=Number(gold===priorBest);const final=Number(gold===0),saysFinal=probs[0]!>=.5;
        z.yes+=final;z.finalBrier+=(probs[0]!-final)**2;z.priorBrier+=(model.prior[0]!-final)**2;
        z.tp+=Number(saysFinal&&final);z.fp+=Number(saysFinal&&!final);z.fn+=Number(!saysFinal&&final);
      }
    }
    for(const [s,a]of acc)pool.get(s)!.push({session:t.session,a});
  }
  const sessions=[...new Set(trials.map(t=>t.session))];
  const comparisons=(groups:Row[][])=>({directVsLocal:compare(groups,sessions,'direct','local'),binaryVsDirect:compare(groups,sessions,'binaryOracle','direct'),oracleVsDirect:compare(groups,sessions,'oracle','direct'),liveVsDirect:compare(groups,sessions,'live','direct'),liveVsLocal:compare(groups,sessions,'live','local'),liveVsPriorMix:compare(groups,sessions,'live','priorMix')});
  const slices=Object.fromEntries(names.map(n=>{const rs=pool.get(n)!;return[n,{...table(rs),comparisons:comparisons([rs])}];}));
  const balanced=Object.fromEntries(arms.map(a=>[a,slices.medium.mae[a]!=null&&slices.long.mae[a]!=null?(slices.medium.mae[a]!+slices.long.mae[a]!)/2:null]));
  return {slices,balanced,comparisons:comparisons([pool.get('medium')!,pool.get('long')!])};
}
const root=resolve(process.argv[2]!),train=[0,1,2].flatMap(i=>load(join(root,`fold${i}/milestones.jsonl`)));
const stage=samples(train,'band'),todo=samples(train,'todoBand');
const stageModel=fit(stage.rows),todoModel=fit(todo.rows);
// Freeze both heads and both classifiers before loading a single held-out label.
const test=load(join(root,'test/milestones.jsonl'));
const devIds=new Set(train.map(t=>t.session));if(test.some(t=>devIds.has(t.session)))throw Error('Training and held-out sessions overlap');
const entries=(group:string):number=>JSON.parse(readFileSync(join(root,group,'entries.json'),'utf8')).length;
const output={sessions:{development:[0,1,2].reduce((n,i)=>n+entries(`fold${i}`),0),test:entries('test')},completed:test.length,missing:test.reduce((s,t)=>s+t.missing,0),
  fitPromptCounts:{stage:stage.counts,todo:todo.counts},stage:evaluate(stageModel,test,'band'),todo:evaluate(todoModel,test,'todoBand')};
const report={...output,provenance:reference.provenance};
if(output.stage.slices.all.readings!==reference.total || output.completed!==reference.completedPrompts ||
   output.missing!==reference.missing || Math.abs(output.stage.balanced.local!-reference.balanced.local)>1e-8)
  throw Error('Replayed local control differs from the frozen reference');
writeFileSync(join(root,'result.json'),JSON.stringify(report,null,2)+'\n',{mode:0o600,flag:'wx'});
console.log(JSON.stringify({completed:output.completed,stage:output.stage.balanced,todo:output.todo.balanced,stageContrasts:output.stage.comparisons}));
