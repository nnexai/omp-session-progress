// Private, numeric-only, out-of-fold development fixture for the duration challengers.
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { applyEvent, emptyFacts } from '../../src/facts';
import { extractFeatures } from '../../src/features';
import { evalEnsembleZ, evalQuantileRangeZ, type EnsembleModel } from '../../src/gbdt';
import { extractSessionText, activeSegments } from '../extract';
import { frozenText } from '../frozen-session';
import type { Entry } from '../tool-timing/cohort';

export interface Input { x:number[]; e:number; b:number; upper:number; tools:number; todo:boolean }
export interface Row extends Input { low:number; y:number; turns:number; sid:number; pid:number; fold:number; slow:boolean; elapsed:number }
const source=resolve(import.meta.dir,'../..');
export const entries=(corpus:string,split:string):Entry[]=>JSON.parse(readFileSync(join(corpus,split,'entries.json'),'utf8'));
export function privateDir(root:string) { if (!statSync(root).isDirectory() || (statSync(root).mode&0o077)!==0) throw Error('Use a private directory (mode 0700)'); }

// Same shipped training program and weights recipe, different (strictly disjoint) fit sessions.
// Store only under the private study root. Original transcripts never change.
export async function refit(root:string, train:Entry[], excluded:Entry[]):Promise<EnsembleModel> {
  privateDir(root);
  const excludedIds=new Set(excluded.map(e=>e.id)), seen=new Set<string>();
  const dir=mkdtempSync(join(root,'refit-')), sessions=join(dir,'home/.omp/agent/sessions/frozen');
  try {
    mkdirSync(sessions,{recursive:true}); mkdirSync(join(dir,'eval')); mkdirSync(join(dir,'.omp/ratchet/progress-time'),{recursive:true});
    cpSync(join(source,'src'),join(dir,'src'),{recursive:true});
    for(const f of ['train','extract','landmarks','paths']) cpSync(join(source,`eval/${f}.ts`),join(dir,`eval/${f}.ts`));
    writeFileSync(join(dir,'.omp/ratchet/progress-time/_state.json'),JSON.stringify({test_ids:[...excludedIds]}));
    for(const ent of train){
      if(seen.has(ent.id)||excludedIds.has(ent.id))throw Error('Duplicate or non-disjoint session'); seen.add(ent.id);
      writeFileSync(join(sessions,ent.id+'.jsonl'),frozenText(ent),{mode:0o600,flag:'wx'});
    }
    const proc=Bun.spawn([process.execPath,join(dir,'eval/train.ts')],{env:{...process.env,HOME:join(dir,'home')},stdout:'ignore',stderr:'inherit'});
    if(await proc.exited!==0)throw Error('Fold refit failed');
    return (await import(join(dir,'src/model-data.ts'))).MODEL_DATA;
  } finally { rmSync(dir,{recursive:true,force:true}); }
}
const seconds=(z:number)=>Math.max(0,Math.exp(Math.min(10.5,Math.max(0,z)))-30);
export function rowsFor(ent:Entry,sid:number,fold:number,model:EnsembleModel):Row[] {
  const trace=extractSessionText(frozenText(ent),ent.path), facts=emptyFacts(), rows:Row[]=[];
  if(trace.id!==ent.id)throw Error('Session identity mismatch');
  let cursor=0;
  for(const [pid,p] of trace.prompts.entries()){
    if(p.outcome!=='completed')continue;
    const seg=activeSegments(trace.events,p), total=seg.reduce((a,s)=>a+s.ms,0);
    // Label only: never provided to predict(). Exclude turns already visible at this checkpoint.
    // extractSessionText emits message_end only for assistant messages.
    const turnsAt=trace.events.slice(p.startIndex,p.endIndex+1).filter(v=>v.event.type==='message_end').map(v=>v.t);
    let j=0,summed=0,seenTurns=0;
    for(let t=5000;t<total;t=t===5000?15000:t+15000){
      while(summed+seg[j]!.ms<t)summed+=seg[j++]!.ms;
      const at=seg[j]!.t+t-summed;
      while(cursor<=p.endIndex&&trace.events[cursor]!.t<=at){const v=trace.events[cursor++]!;applyEvent(facts,v.event,v.t);}
      while(seenTurns<turnsAt.length&&turnsAt[seenTurns]!<=at)seenTurns++;
      const f=extractFeatures(facts,at), z=evalEnsembleZ(model,f.x,f.logElapsed,f.anchorZ,{elapsedS:f.elapsedS,toolCalls:f.toolCalls,hasActiveTodo:f.hasActiveTodo});
      const {lowZ,highZ}=evalQuantileRangeZ(model,f.x,f.logElapsed,f.anchorZ,z);
      rows.push({x:f.x,e:f.elapsedS,b:seconds(z),low:seconds(lowZ),upper:seconds(highZ),tools:f.toolCalls,todo:f.hasActiveTodo,y:(total-t)/1000,turns:turnsAt.length-seenTurns,sid,pid,fold,elapsed:t/1000,slow:[...facts.underway.values()].some(v=>v.name!=='ask'&&t/1000-v.startedAtS>60)});
    }
  }
  return rows;
}
if(import.meta.main){
  const [corpus,root,arg]=process.argv.slice(2);privateDir(root!);
  const fold=Number(arg);if(![0,1,2].includes(fold))throw Error('Expected dev fold 0..2');
  const groups=[0,1,2].map(i=>entries(corpus!,'fold'+i));
  const model=await refit(root!,groups.flatMap((g,i)=>i===fold?[]:g),groups[fold]!);
  const rows=groups[fold]!.flatMap((ent,i)=>rowsFor(ent,i*3+fold,fold,model));
  writeFileSync(join(root!,`dev${fold}.json`),JSON.stringify(rows),{flag:'wx',mode:0o600});
  console.log(`Development fold ${fold}: ${groups[fold]!.length} sessions, ${rows.length} checkpoints`);
}
