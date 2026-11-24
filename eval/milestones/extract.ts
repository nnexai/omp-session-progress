import { appendFileSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { activeSegments, extractSession } from '../extract';
import { collectSession } from '../tool-timing/replay';
import { verifyEntry, type Entry } from '../tool-timing/cohort';
import { stageTimeline, todoTimeline } from './labels';
import type { Example, Trial } from './types';

const root=resolve(import.meta.dir,'../..'), out=join(root,'milestones.jsonl');
const entries:Entry[]=JSON.parse(readFileSync(join(root,'entries.json'),'utf8'));
writeFileSync(out,'',{mode:0o600,flag:'wx'});
for(const entry of entries){
  verifyEntry(entry);
  const trace=extractSession(entry.path), replay=collectSession(entry.path),todo=todoTimeline(trace);
  if(trace.id!==entry.id||replay.id!==entry.id)throw Error('Replay identity mismatch');
  let k=0;
  trace.prompts.forEach((p,i)=>{
    if(p.outcome!=='completed')return;
    const observed=replay.prompts[k++]!, timeline=stageTimeline(trace.events,p);
    const duration=activeSegments(trace.events,p).reduce((sum,s)=>sum+s.ms,0)/60_000;
    const rows:Example[]=observed.points.map((pt,j)=>{
      const {band,...visible}=timeline.at(pt.atMs,pt.elapsedS);
      const remainingMin=observed.truth[j]!.remainingS/60;
      if(Math.abs(pt.elapsedS/60+remainingMin-duration)>1e-6)throw Error('Checkpoint misalignment');
      return {prefix:{evidence:pt.evidence,...visible},gold:{band,todoBand:todo[i]!(pt.atMs),remainingMin}};
    });
    const trial:Trial={session:entry.id,duration,rows,missing:observed.missingPoints??0};
    appendFileSync(out,JSON.stringify(trial)+'\n');
  });
  if(k!==replay.prompts.length)throw Error('Prompt alignment mismatch');
  verifyEntry(entry);
}
console.log(`Replayed ${entries.length} sessions; private derived examples only.`);
