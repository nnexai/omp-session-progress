import { activeSegments, type Trace } from '../extract';
import { verificationHandoff, verificationIdentity, verificationNotices, verificationReceipt } from './identity';
import type { Frame, Replay, TimingPrefix } from './types';
export { verificationTrace } from './identity';

type Sample={exact:number;kind:string;seconds:number;prompt:number};
type Call={id:number;exact:number;kind:string;start:number;prompt:number; prefix:TimingPrefix; background:boolean};
type End={at:number;finished:boolean};
const median=(a:number[]):number|null=>{
  if(!a.length)return null;
  const sorted=[...a].sort((x,y)=>x-y), m=Math.floor(sorted.length/2);
  return sorted.length%2 ? sorted[m]! : (sorted[m-1]!+sorted[m]!)/2;
};

/**
 * History is frozen at launch. Native notices measure time to notification,
 * not a passing test or necessarily process exit. Notice-only durations enter
 * timing history unless a trusted envelope reports failure. Known foreground
 * failures are observed ends but do not enter history; missing ends censor.
 * Notices do not alter the baseline event clock or its capped gaps.
 */
export function replayVerifications(trace:Trace):Replay {
  const result:Replay={prompts:[],spans:[]}, history:Sample[]=[];
  let nextId=0;
  for(let p=0;p<trace.prompts.length;p++) {
    const prompt=trace.prompts[p]!, segments=activeSegments(trace.events,prompt);
    const offsets=new Map<number,number>(); let total=0;
    for(const segment of segments) { offsets.set(segment.index,total/1000); total+=segment.ms; }
    offsets.set(prompt.endIndex,total/1000);
    const open=new Map<string,Call>(), background=new Map<number,Call>(), ambiguous=new Set<number>(), seenJobs=new Set<number>(), ends=new Map<number,End>();
    const notices=verificationNotices(trace).filter(n=>n.at>=trace.events[prompt.startIndex]!.t && n.at<=trace.events[prompt.endIndex]!.t);
    let noticeIndex=0;
    const rows:Frame[]=[], selected:number[]=[];
    let cursor=prompt.startIndex;
    const close=(call:Call,at:number,finished:boolean,historyEligible:boolean,noticeOnly=false)=>{
      const seconds=Math.max(0,at-call.start);
      ends.set(call.id,{at,finished});
      result.spans.push({kind:call.kind,seconds,completed:finished,noticeOnly,background:call.background,exactAnchorS:median(call.prefix.exact),familyAnchorS:median(call.prefix.family)});
      if(historyEligible) history.push({exact:call.exact,kind:call.kind,seconds,prompt:p});
    };
    const completeJob=(job:number,at:number)=>{
      const call=background.get(job);
      if(call){close(call,at,true,true,true);background.delete(job);}
    };
    const process=(index:number)=>{
      const event=trace.events[index]!.event, at=offsets.get(index)!;
      if(event.type==='tool_execution_start') {
        const key=String(event.toolCallId), old=open.get(key);
        if(old) {close(old,at,false,false);open.delete(key);}
        const identity=verificationIdentity(event);
        if(!identity)return;
        const exact=history.filter(s=>s.exact===identity.exact).slice(-16);
        const family=history.filter(s=>s.kind===identity.kind).slice(-16);
        const id=nextId++;
        open.set(key,{id,...identity,start:at,prompt:p,background:false,prefix:{call:id,kind:identity.kind,ageS:0,background:false,exact:exact.map(s=>s.seconds),family:family.map(s=>s.seconds),priorPromptExact:exact.filter(s=>s.prompt<p).length,priorPromptFamily:family.filter(s=>s.prompt<p).length}});
      } else if(event.type==='tool_execution_end') {
        const receipt=verificationReceipt(event);
        if(receipt!==undefined)completeJob(receipt,at);
        const key=String(event.toolCallId), call=open.get(key);
        if(!call)return;
        const d=event.details && typeof event.details==='object' ? event.details as Record<string,unknown> : {};
        const job=verificationHandoff(event);
        const handedOff=job!==undefined || d.backgrounded===true || d.running===true || d.status==='running' || d.status==='backgrounded';
        const failed=event.isError===true || (typeof d.exitCode==='number' && d.exitCode!==0);
        open.delete(key);
        if(handedOff) {
          call.background=true;
          call.prefix.background=true;
          if(job===undefined || ambiguous.has(job)) close(call,at,false,false);
          else if(seenJobs.has(job)) {
            const previous=background.get(job);
            if(previous) { background.delete(job); close(previous,at,false,false); }
            ambiguous.add(job); close(call,at,false,false);
          } else { seenJobs.add(job); background.set(job,call); }
        } else close(call,at,true,!failed);
      }
    };
    const activeAt=(t:number)=>{
      let seconds=0;
      for(const s of segments) {
        if(t<s.t) break;
        seconds=(offsets.get(s.index) ?? 0)+Math.min(s.ms,Math.max(0,t-s.t))/1000;
      }
      return seconds;
    };
    const advance=(wall:number)=>{
      // Equal-time events are handled first, so a handoff at the same instant
      // as its notice can still be linked.
      while(true) {
        const eventTime=cursor<=prompt.endIndex ? trace.events[cursor]!.t : Infinity;
        const noticeTime=notices[noticeIndex]?.at ?? Infinity;
        if(Math.min(eventTime,noticeTime)>wall) break;
        if(eventTime<=noticeTime) process(cursor++);
        else {
          const n=notices[noticeIndex++]!;completeJob(n.job,activeAt(n.at));
        }
      }
    };
    let elapsed=0, segment=0;
    if(prompt.outcome==='completed') {
      for(let target=5000;target<total;target=target===5000?15000:target+15000) {
        while(elapsed+segments[segment]!.ms<target) elapsed+=segments[segment++]!.ms;
        const at=segments[segment]!.t+target-elapsed;
        advance(at);
        let chosen:Call|undefined;
        for(const call of open.values()) if(!chosen || call.start<chosen.start || (call.start===chosen.start && call.id<chosen.id)) chosen=call;
        if(!chosen) for(const call of background.values()) if(!chosen || call.start<chosen.start || (call.start===chosen.start && call.id<chosen.id)) chosen=call;
        rows.push({elapsedS:target/1000,timing:chosen?{...chosen.prefix,ageS:Math.max(0,target/1000-chosen.start)}:null,toolLeftS:null});
        selected.push(chosen?.id??-1);
      }
    }
    advance(trace.events[prompt.endIndex]!.t);
    for(const call of open.values()) close(call,total/1000,false,false);
    for(const call of background.values()) close(call,total/1000,false,false);
    for(let i=0;i<rows.length;i++) {
      const end=ends.get(selected[i]!);
      if(end?.finished) rows[i]!.toolLeftS=Math.max(0,end.at-rows[i]!.elapsedS);
    }
    if(prompt.outcome==='completed') result.prompts.push(rows);
  }
  return result;
}
