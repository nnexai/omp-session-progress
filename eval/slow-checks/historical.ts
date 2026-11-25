import { readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { activeSegments, type Trace } from '../extract';
import { frozenText, sha } from '../frozen-session';
import type { Entry } from '../tool-timing/cohort';
import reference from '../milestones/result.json';
import { verificationHandoff, verificationIdentity, verificationNotices, verificationReceipt, verificationReceiptCount, verificationTrace } from '../repeat-verification/identity';

type Interval = [number, number];
type Call = { exact:number; kind:string; start:number; selectedExact:boolean[]; selectedFamily:boolean[]; foregroundEnd?:number; end?:number; endKind?:'foreground'|'notice'|'censored'; failed?:boolean; job?:number; waits:Interval[]; ambiguousWaits:Interval[] };
type Tool = { start:number; name:string; call?:Call; pendingAtStart?:number[]; uncertainAtStart?:boolean; changed?:boolean };
const thresholds=[60,600];

// Clip to the existing replay clock: no added clock events, capped silent gaps,
// and no time attributed while an ask is open. Merge before summing.
export function intervalSeconds(intervals:Interval[], segments:{t:number;ms:number}[]):number {
  const pieces:Interval[]=[];
  for(const [a,b] of intervals) for(const s of segments) {
    const lo=Math.max(a,s.t), hi=Math.min(b,s.t+s.ms);
    if(hi>lo)pieces.push([lo,hi]);
  }
  pieces.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
  let total=0, lo=0, hi=0;
  for(const [a,b] of pieces) { if(a>hi){total+=hi-lo;lo=a;hi=b;} else hi=Math.max(hi,b); }
  return (total+hi-lo)/1000;
}

/** Returns ephemeral per-call facts; callers must only persist aggregates. */
export function analyze(trace:Trace):Call[] {
  const history:{exact:number;kind:string;seconds:number}[]=[], calls:Call[]=[];
  // Unresolved jobs from earlier prompts make exclusivity unprovable.
  let priorUnknown=false;
  for(const prompt of trace.prompts) {
    const segments=activeSegments(trace.events,prompt), open=new Map<string,Tool>(), jobs=new Map<number,Call|null>(), seen=new Set<number>();
    const promptStart=trace.events[prompt.startIndex]!.t, promptEnd=trace.events[prompt.endIndex]!.t;
    const notices=verificationNotices(trace).filter(n=>n.at>=promptStart && n.at<=promptEnd);
    let ni=0, uncertain=priorUnknown;
    const other:{interval:Interval;call?:Call}[]=[];
    const change=()=>{for(const t of open.values())if(t.name==='wait')t.changed=true;};
    const finishJob=(job:number,at:number)=>{const c=jobs.get(job);if(c && c.end===undefined){c.end=at;c.endKind='notice';} if(jobs.has(job)){change();jobs.delete(job);}};
    const finishTool=(key:string,at:number,event?:Trace['events'][number]['event'])=>{
      const tool=open.get(key); if(!tool)return;
      open.delete(key);
      if(tool.name!=='wait')other.push({interval:[tool.start,at],call:tool.call});
      if(tool.name==='wait' && event){
        const receipt=verificationReceipt(event), c=receipt===undefined?undefined:jobs.get(receipt);
        if(c && c.end===undefined) {
          const target:Interval=[tool.start,at];
          if(verificationReceiptCount(event)===1 && !tool.uncertainAtStart && !tool.changed && tool.pendingAtStart?.length===1 && tool.pendingAtStart[0]===receipt)c.waits.push(target);
          else c.ambiguousWaits.push(target);
        }
        if(receipt!==undefined)finishJob(receipt,at);
      }
      const c=tool.call;if(!c)return;
      c.foregroundEnd=at;
      const d=event?.details && typeof event.details==='object'?event.details as Record<string,unknown>:{};
      const job=event && verificationHandoff(event);
      const handed=job!==undefined || d.backgrounded===true || d.running===true || d.status==='running' || d.status==='backgrounded';
      c.failed=event?.isError===true || (typeof d.exitCode==='number' && d.exitCode!==0);
      if(handed){
        change();
        if(job===undefined){uncertain=true;c.end=at;c.endKind='censored';}
        else if(seen.has(job)){uncertain=true;const old=jobs.get(job);if(old && old.end===undefined){old.end=at;old.endKind='censored';}jobs.delete(job);c.end=at;c.endKind='censored';}
        else {seen.add(job);c.job=job;jobs.set(job,c);}
      } else if(event) {
        c.end=at;c.endKind='foreground';
        const seconds=intervalSeconds([[c.start,at]],segments);
        if(!c.failed)history.push({exact:c.exact,kind:c.kind,seconds});
      } else {c.end=at;c.endKind='censored';}
    };
    for(let i=prompt.startIndex;i<=prompt.endIndex;i++){
      const {t,event}=trace.events[i]!;
      // Same-time native events precede notices, as in replay.
      while(ni<notices.length && notices[ni]!.at<t){const n=notices[ni++]!;finishJob(n.job,n.at);}
      const key=String(event.toolCallId);
      if(event.type==='tool_execution_start'){
        if(open.has(key))finishTool(key,t);
        const identity=verificationIdentity(event);
        const call=identity?{...identity,start:t,selectedExact:thresholds.map(x=>history.some(h=>h.exact===identity.exact&&h.seconds>=x)),selectedFamily:thresholds.map(x=>history.some(h=>h.kind===identity.kind&&h.seconds>=x)),waits:[],ambiguousWaits:[]} satisfies Call:undefined;
        if(call)calls.push(call);
        open.set(key,{start:t,name:String(event.toolName),call,...(event.toolName==='wait'?{pendingAtStart:[...jobs.keys()],uncertainAtStart:uncertain}: {})});
      } else if(event.type==='tool_execution_end'){
        // Track handoffs of non-verification shell calls too: otherwise a wait
        // that can return another job would be wrongly attributed exclusively.
        const tool=open.get(key), job=verificationHandoff(event);
        if(tool && !tool.call && job!==undefined){change();if(seen.has(job)){uncertain=true;jobs.delete(job);}else{seen.add(job);jobs.set(job,null);}}
        if(tool && !tool.call && job===undefined && event.details && typeof event.details==='object') {
          const d=event.details as Record<string,unknown>;
          if(d.backgrounded===true || d.running===true || d.status==='running' || d.status==='backgrounded'){uncertain=true;change();}
        }
        finishTool(key,t,event);
        // Receipts may lack a traced start; they can close timing but never prove waiting.
        if(!tool && verificationReceipt(event)!==undefined)finishJob(verificationReceipt(event)!,t);
      }
    }
    while(ni<notices.length){const n=notices[ni++]!;finishJob(n.job,n.at);}
    for(const key of [...open.keys()])finishTool(key,promptEnd);
    for(const c of jobs.values())if(c && c.end===undefined){c.end=promptEnd;c.endKind='censored';}
    priorUnknown=uncertain || jobs.size>0;
    // Store only aggregate-relevant interval lengths, never the intervals themselves.
    for(const c of calls.filter(c=>c.start>=promptStart && c.start<=promptEnd && !(c as Measured).measured)){
      const end=c.end??promptEnd, foregroundEnd=Math.min(c.foregroundEnd??end,end);
      const overlap=other.filter(o=>o.call!==c).map(({interval:[a,b]})=>
        [Math.max(a,c.start),Math.min(b,end)] as Interval);
      const waits=c.waits.map(([a,b])=>[Math.max(a,c.start),Math.min(b,end)] as Interval);
      const waitOverlap=waits.flatMap(([a,b])=>overlap.map(([x,y])=>[Math.max(a,x),Math.min(b,y)] as Interval));
      const waitSeconds=intervalSeconds(waits,segments);
      Object.assign(c,{measured:true,seconds:intervalSeconds([[c.start,end]],segments),foregroundSeconds:intervalSeconds([[c.start,foregroundEnd]],segments),waitSeconds,exclusiveWaitSeconds:waitSeconds-intervalSeconds(waitOverlap,segments),ambiguousSeconds:intervalSeconds(c.ambiguousWaits.map(([a,b])=>[Math.max(a,c.start),Math.min(b,end)]),segments),overlapSeconds:intervalSeconds(overlap,segments)});
    }
  }
  return calls;
}
type Measured=Call & {measured:boolean;seconds:number;foregroundSeconds:number;waitSeconds:number;exclusiveWaitSeconds:number;ambiguousSeconds:number;overlapSeconds:number};

export function aggregate(traces:Iterable<Trace>) {
  const rows=Array.from(traces, analyze) as Measured[][];
  const by=(threshold:number,filter:(c:Measured)=>boolean)=>{
    const selected=rows.map(cs=>cs.filter(filter)), all=selected.flat();
    const fg=all.filter(c=>c.endKind==='foreground'), bg=all.filter(c=>c.job!==undefined);
    const sum=(get:(c:Measured)=>number)=>all.reduce((n,c)=>n+get(c),0);
    return {sessions:selected.filter(s=>s.length).length, selected:all.length,
      foregroundMeasured:fg.length, foregroundActualSlow:fg.filter(c=>c.seconds>=threshold).length,
      foregroundFailed:fg.filter(c=>c.failed).length, backgroundHandoffs:bg.length,
      backgroundNotifications:bg.filter(c=>c.endKind==='notice').length, censored:all.filter(c=>c.endKind==='censored').length,
      backgroundKnownWaiting:bg.filter(c=>c.waits.length>0).length, backgroundAmbiguousWaiting:bg.filter(c=>c.ambiguousWaits.length>0).length,
      observedDurationSeconds:sum(c=>c.seconds), foregroundInvocationSeconds:sum(c=>c.foregroundSeconds),
      attributedBackgroundWaitSeconds:sum(c=>c.waitSeconds), withoutConcurrentToolWaitSeconds:sum(c=>c.exclusiveWaitSeconds),
      ambiguousBackgroundWaitSeconds:sum(c=>c.ambiguousSeconds), concurrentNonWaitToolSeconds:sum(c=>c.overlapSeconds)};
  };
  return Object.fromEntries(thresholds.map((t,i)=>[String(t),{
    exact:by(t,c=>c.selectedExact[i]!),family:by(t,c=>c.selectedFamily[i]!),
    observedSlowForeground:by(t,c=>c.endKind==='foreground'&&c.seconds>=t),
    observedSlowNotification:by(t,c=>c.endKind==='notice'&&c.seconds>=t),
  }]));
}

if(import.meta.main){
  const root=resolve(process.argv[2]??'.'), out=process.argv[3];
  if(!process.argv[2]||!out)throw Error('Usage: bun eval/slow-checks/historical.ts FROZEN_MILESTONE_DIR NEW_RESULT_JSON');
  const entries=['fold0','fold1','fold2','test'].map(n=>JSON.parse(readFileSync(join(root,n,'entries.json'),'utf8')) as Entry[]);
  const dev=entries.slice(0,3).flat(), test=entries[3]!;
  const setHash=(es:Entry[])=>sha(es.map(e=>e.digest).sort().join('\n'));
  if(setHash(dev)!==reference.provenance.developmentContentSetSha256 || setHash(test)!==reference.provenance.holdoutContentSetSha256 || new Set(entries.flat().map(e=>e.id)).size!==entries.flat().length)throw Error('Unexpected frozen split');
  function* load(es:Entry[]) {for(const e of es){const trace=verificationTrace(frozenText(e),e.path);if(trace.id!==e.id)throw Error('Unexpected session identity');yield trace;}}
  const method='Exact/family selected at dispatch by a prior successful measured slow foreground invocation in this session. ObservedSlow groups are hindsight denominators, not predictors. Time follows the capped active replay clock. Foreground invocation does not prove exclusive waiting. A background end is a notification, not proof of process exit or success. A wait dispatch returning one native job receipt with no known competing job is attributable to that receipt, not necessarily to that process for the entire interval. Concurrent non-wait and ambiguous intervals can overlap; silence is unknown. Seconds sum per call, not per session.';
  const cohort=(es:Entry[])=>({sessions:es.length,contentSetSha256:setHash(es),thresholds:aggregate(load(es))});
  const result={method,development:cohort(dev),test:cohort(test)};
  writeFileSync(out,JSON.stringify(result,null,2)+'\n',{flag:'wx',mode:0o600});
}
