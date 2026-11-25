import {readFileSync,writeFileSync} from 'node:fs';import {join} from 'node:path';import {emptyFacts,applyEvent,activeMs} from '../../src/facts';import {activeSegments,extractSessionText} from '../extract';import {frozenText} from '../frozen-session';import type {Split} from './rows';import {IntentState} from './intent-state';import type {Entry} from '../tool-timing/cohort';import {privateDir} from '../horizon-study/data';
export function intentRows(entry:Entry){const trace=extractSessionText(frozenText(entry),entry.path),state=new IntentState(),facts=emptyFacts(),out:{i:number[];future:number}[]=[];let cursor=0;
  for(const p of trace.prompts){if(p.outcome!=='completed')continue;const seg=activeSegments(trace.events,p),total=seg.reduce((s,v)=>s+v.ms,0);let j=0,tick=0,completed=0;
    for(let t=5000;t<total;t=t===5000?15000:t+15000){while(tick+seg[j]!.ms<t)tick+=seg[j++]!.ms;const at=seg[j]!.t+t-tick;
      while(cursor<=p.endIndex&&trace.events[cursor]!.t<=at){const {t:ts,event}=trace.events[cursor++]!;applyEvent(facts,event,ts);if(event.type==='before_agent_start')state.reset(typeof event.prompt==='string'?event.prompt:'');if(event.type==='tool_execution_start')state.see(event.intent,activeMs(facts,ts)/1000);}
      while(completed<p.toolStartIndices.length&&p.toolStartIndices[completed]!<cursor)completed++;
      out.push({i:state.vector(activeMs(facts,at)/1000),future:p.toolStartIndices.length-completed});
    }
  }return out;
}
if(import.meta.main){const [root,k]=process.argv.slice(2);privateDir(root!);const f=Number(k),s:Split=JSON.parse(readFileSync(join(root!,'cohort.json'),'utf8'));const arr=s.dev[f]!.flatMap(intentRows);writeFileSync(join(root!,`intent${f}.json`),JSON.stringify(arr),{mode:0o600,flag:'wx'});console.log('intent checkpoints',f,arr.length);}
