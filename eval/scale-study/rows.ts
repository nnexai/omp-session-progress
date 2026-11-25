import {createWriteStream,readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {once} from 'node:events';
import {extractFeatures} from '../../src/features';
import {addWorkShape,researchFacts,researchEvent} from './work-shape';
import {evalEnsembleZ,type EnsembleModel} from '../../src/gbdt';
import {activeSegments,extractSessionText} from '../extract';
import {frozenText} from '../frozen-session';
import {refit,privateDir} from '../horizon-study/data';
import type {Entry} from '../tool-timing/cohort';

export interface Split {dev:Entry[][];legacy:Entry[][];official:Entry[];enriched:Entry[];fresh:Entry[];aux:Entry[]}
export interface ScaleRow {x:number[];e:number;y:number;sid:number;pid:number;elapsed:number;b:number;slow:boolean}
export function* rowsFor(entry:Entry,sid:number,model:EnsembleModel):Generator<ScaleRow>{
  const trace=extractSessionText(frozenText(entry),entry.path);if(trace.id!==entry.id)throw Error('Wrong frozen session');
  const facts=researchFacts();let idx=0;
  for(const [pid,p] of trace.prompts.entries()){
    if(p.outcome!=='completed')continue;
    const seg=activeSegments(trace.events,p),total=seg.reduce((s,v)=>s+v.ms,0);let j=0,summed=0;
    for(let t=5000;t<total;t=t===5000?15000:t+15000){
      while(summed+seg[j]!.ms<t)summed+=seg[j++]!.ms;
      const at=seg[j]!.t+t-summed;
      while(idx<=p.endIndex&&trace.events[idx]!.t<=at){const v=trace.events[idx++]!;researchEvent(facts,v.event,v.t);}
      const f=extractFeatures(facts,at),z=evalEnsembleZ(model,f.x,f.logElapsed,f.anchorZ,{elapsedS:f.elapsedS,toolCalls:f.toolCalls,hasActiveTodo:f.hasActiveTodo});
      addWorkShape(f.x,facts,f.elapsedS);
      yield {sid,pid,x:f.x,e:f.elapsedS,y:(total-t)/1000,elapsed:t/1000,b:Math.max(0,Math.exp(Math.min(10.5,Math.max(0,z)))-30),slow:[...facts.underway.values()].some(v=>v.name!=='ask'&&t/1000-v.startedAtS>60)};
    }
  }
}
if(import.meta.main){
  const [root,arg]=process.argv.slice(2);privateDir(root!);const cohort:Split=JSON.parse(readFileSync(join(root!,'cohort.json'),'utf8'));
  const f=Number(arg);if(![0,1,2].includes(f))throw Error('Expected development fold');
  const safe=cohort.dev.flatMap((g,i)=>i===f?[]:g.filter(e=>cohort.legacy[i]!.some(v=>v.id===e.id)));
  const model=await refit(root!,safe,[...cohort.dev[f]!,...cohort.official,...cohort.enriched,...cohort.fresh]);
  writeFileSync(join(root!,`control${f}.json`),JSON.stringify(model),{mode:0o600,flag:'wx'});
  const out=createWriteStream(join(root!,`dev${f}.jsonl`),{flags:'wx',mode:0o600});let count=0;
  for(const [i,s] of cohort.dev[f]!.entries())for(const row of rowsFor(s,i*3+f,model)){if(!out.write(JSON.stringify(row)+'\n'))await once(out,'drain');count++;}
  out.end();await once(out,'finish');console.log(`Fold ${f}: ${count} causal readings`);
}
