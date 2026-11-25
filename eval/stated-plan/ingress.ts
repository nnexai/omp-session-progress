import { frozenText } from '../frozen-session';
import { activeSegments, extractSessionText, type Trace } from '../extract';
import { applyEvent, emptyFacts, activeMs } from '../../src/facts';
import type { Entry } from '../tool-timing/cohort';
import { proseRoute } from './prose';
import { checklistRoute } from './checklist';
import type { Plans, Route } from './types';

interface Checkpoint extends Plans { elapsedS:number }
interface RouteSession {id:string; prompts:Checkpoint[][]}
// Use persisted message_end timestamps (not message construction time). The most
// recent nonempty public assistant statement supersedes its predecessor. Unchanged
// prose expires after five active minutes; the confirmed checklist is a living plan.
export function replayPlans(trace:Trace):RouteSession {
  const facts=emptyFacts(),prompts:Checkpoint[][]=[];
  let cursor=0,prose:Route|null=null,proseAt=0,checklist:Route|null=null;
  for(const prompt of trace.prompts){
    if(prompt.outcome!=='completed')continue;
    const segments=activeSegments(trace.events,prompt),total=segments.reduce((n,s)=>n+s.ms,0);
    const rows:Checkpoint[]=[]; let elapsed=0,i=0;
    for(let target=5_000;target<total;target=target===5_000?15_000:target+15_000){
      while(elapsed+segments[i]!.ms<target)elapsed+=segments[i++]!.ms;
      const at=segments[i]!.t+target-elapsed;
      while(cursor<=prompt.endIndex && trace.events[cursor]!.t<=at){
        const {t,event}=trace.events[cursor++]!;
        const completingTodo=event.type==='tool_execution_end' && !event.isError && (facts.underway.get(String(event.toolCallId))?.name??event.toolName)==='todo';
        applyEvent(facts,event,t);
        if(event.type==='before_agent_start'){prose=null;checklist=null;}
        if(completingTodo && facts.todoUpdated)checklist=checklistRoute(facts.todos);
        if(event.type==='message_end'){
          const m=event.message as {role?:string;stopReason?:string;content?:{type?:string;text?:string}[]};
          if(m.role!=='assistant'||!Array.isArray(m.content))continue;
          const text=m.content.filter(b=>b.type==='text'&&typeof b.text==='string').map(b=>b.text).join('\n');
          if(!text.trim())continue;
          prose=m.stopReason==='aborted'||m.stopReason==='error'||m.stopReason==='stop'?null:proseRoute(text);
          proseAt=activeMs(facts,t)/1000;
        }
      }
      const elapsedS=target/1000;
      rows.push({elapsedS,prose:elapsedS-proseAt<=300?prose:null,checklist});
    }
    prompts.push(rows);
  }
  return {id:trace.id,prompts};
}

export function frozenPlans(e:Entry):RouteSession {
  return replayPlans(extractSessionText(frozenText(e),e.path));
}
