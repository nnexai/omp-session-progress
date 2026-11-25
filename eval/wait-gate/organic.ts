import { readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { activeSegments, type Trace } from '../extract';
import { frozenText, sha } from '../frozen-session';
import type { Entry } from '../tool-timing/cohort';
import type { Trial } from '../milestones/types';
import reference from '../milestones/result.json';
import { privateProvenance } from '../provenance';
import { verificationTrace, verificationIdentity, verificationHandoff, verificationNotices, verificationReceipt } from '../repeat-verification/identity';
import { forecast } from './forecast';

type Event = Trace['events'][number]['event'];
type Call = { id:number; start:number; exact:number; prior:number|null };
type Tool = { name:string; call?:Call; waitJob?:number; valid?:boolean };
export type Frame = { elapsedS:number; prior:number|null; age:number; call:number|null; wait:boolean; identifiable:boolean; strict:boolean };

// Only prefix state is used. Native wait has no authenticated target at dispatch;
// a later receipt closes a job but cannot prove what an earlier wait targeted.
export function organicReplay(trace:Trace):Frame[][] {
  const history=new Map<number,number>(), jobs=new Map<number,Call|null>(), seen=new Set<number>();
  let unknown=false, next=0, clock=0;
  const output:Frame[][]=[];
  for(const prompt of trace.prompts){
    const segments=activeSegments(trace.events,prompt), open=new Map<string,Tool>();
    const total=segments.reduce((s,x)=>s+x.ms,0)/1000;
    const notices=verificationNotices(trace).filter(n=>n.at>=trace.events[prompt.startIndex]!.t && n.at<=trace.events[prompt.endIndex]!.t);
    let ni=0, cursor=prompt.startIndex;
    const invalidate=()=>{for(const t of open.values())if(t.name==='wait')t.valid=false;};
    const finishJob=(id:number)=>{if(jobs.has(id)){invalidate();jobs.delete(id);}};
    const at=(wall:number)=>{let elapsed=0;for(const s of segments){if(wall<s.t)break;elapsed+=Math.min(s.ms,Math.max(0,wall-s.t))/1000;}return clock+elapsed;};
    const process=(event:Event, time:number)=>{
      const key=String(event.toolCallId);
      if(event.type==='tool_execution_start'){
        if(open.has(key)){unknown=true;open.delete(key);invalidate();}
        const name=String(event.toolName);
        if(open.size)invalidate();
        if(name==='task'){unknown=true;invalidate();}
        const identity=verificationIdentity(event);
        const call=identity?{id:next++,exact:identity.exact,start:time,prior:history.get(identity.exact)??null}:undefined;
        const sole=jobs.size===1?[...jobs.entries()][0]:undefined;
        const valid=name==='wait'&&!unknown&&open.size===0&&!!sole?.[1];
        open.set(key,{name,call,...(name==='wait'?{valid,waitJob:valid?sole![0]:undefined}:{})});
      } else if(event.type==='tool_execution_end'){
        const tool=open.get(key), receipt=verificationReceipt(event);
        if(receipt!==undefined)finishJob(receipt);
        if(!tool){if(event.toolName!=='wait')unknown=true;return;}
        open.delete(key);
        if(tool.name!=='wait')invalidate();
        const d=event.details&&typeof event.details==='object'?event.details as Record<string,unknown>:{};
        const job=verificationHandoff(event);
        const handed=job!==undefined||d.backgrounded===true||d.running===true||d.status==='running'||d.status==='backgrounded';
        if(handed){
          invalidate();
          if(job===undefined||seen.has(job)){unknown=true;if(job!==undefined)jobs.delete(job);}
          else {seen.add(job);jobs.set(job,tool.call??null);}
        } else if(tool.call && !event.isError && !(typeof d.exitCode==='number'&&d.exitCode!==0)) {
          history.set(tool.call.exact,Math.max(0,time-tool.call.start));
        }
      }
    };
    const advance=(wall:number)=>{
      while(true){
        const eventTime=cursor<=prompt.endIndex?trace.events[cursor]!.t:Infinity;
        const noticeTime=notices[ni]?.at??Infinity;
        if(Math.min(eventTime,noticeTime)>wall)break;
        if(eventTime<=noticeTime){const item=trace.events[cursor++]!;process(item.event,at(item.t));}
        else finishJob(notices[ni++]!.job);
      }
    };
    const rows:Frame[]=[];
    if(prompt.outcome==='completed'){
      let elapsed=0, segment=0;
      for(let target=5000;target<total*1000;target=target===5000?15000:target+15000){
        while(elapsed+segments[segment]!.ms<target)elapsed+=segments[segment++]!.ms;
        advance(segments[segment]!.t+target-elapsed);
        const candidates=[...open.values()].flatMap(t=>t.call?[t.call]:[]).concat([...jobs.values()].filter((c):c is Call=>c!==null));
        candidates.sort((a,b)=>a.start-b.start||a.id-b.id);
        const candidate=candidates[0], waits=[...open.values()].filter(t=>t.name==='wait');
        const identifiable=waits.length===1 && waits[0]!.valid===true && waits[0]!.waitJob!==undefined && jobs.get(waits[0]!.waitJob!)!=null;
        // For the optimistic arm use its sole job, not an unrelated older call.
        const selected=identifiable?jobs.get(waits[0]!.waitJob!)!:candidate;
        rows.push({elapsedS:target/1000,prior:selected?.prior??null,age:selected?Math.max(0,clock+target/1000-selected.start):0,call:selected?.id??null,wait:waits.length>0,identifiable,strict:false});
      }
      output.push(rows);
    }
    advance(trace.events[prompt.endIndex]!.t);
    if(open.size){unknown=true;open.clear();}
    if(prompt.outcome!=='completed'){unknown=true;invalidate();}
    clock+=total;
  }
  return output;
}

type Point={session:string;prompt:number;duration:number;slow:boolean;gold:number;frame:Frame;local:number};
const arms=['local','ungated','gated','optimistic'] as const;
type Arm=typeof arms[number];
function values(p:Point, threshold:number):Record<Arm,number>{
  const prior=p.frame.prior!==null&&p.frame.prior>=threshold?p.frame.prior:null;
  const strict=forecast(p.local,prior,p.frame.age,p.frame.strict);
  return {local:strict.local,ungated:strict.ungated,gated:strict.gated,optimistic:forecast(p.local,prior,p.frame.age,p.frame.identifiable).gated};
}
function summarize(points:Point[],threshold:number){
  const score=(rows:Point[])=>{
    const grouped=new Map<number,Point[]>();for(const r of rows){const g=grouped.get(r.prompt)??[];g.push(r);grouped.set(r.prompt,g);}
    const groups=[...grouped.values()];
    const means=(fn:(v:number,p:Point)=>number)=>Object.fromEntries(arms.map(a=>[a,groups.length?groups.reduce((s,g)=>s+g.reduce((q,p)=>q+fn(values(p,threshold)[a],p),0)/g.length,0)/groups.length:null]));
    return {sessions:new Set(rows.map(p=>p.session)).size,prompts:groups.length,calls:new Set(rows.filter(p=>p.frame.call!==null).map(p=>`${p.session}:${p.frame.call}`)).size,readings:rows.length,
      maeSeconds:means((v,p)=>Math.abs(v-p.gold)),miss900:means((v,p)=>Number(Math.abs(v-p.gold)>=900)),forecast900:means(v=>Number(v>=900))};
  };
  const hasPrior=(p:Point)=>p.frame.prior!==null&&p.frame.prior>=threshold;
  const filters:Record<string,(p:Point)=>boolean>={all:()=>true,matched:hasPrior,medium:p=>p.duration>=180&&p.duration<900,long:p=>p.duration>=900,'30+':p=>p.duration>=1800,slow:p=>p.slow,short:p=>p.duration<60,
    matchedMedium:p=>hasPrior(p)&&p.duration>=180&&p.duration<900,matchedLong:p=>hasPrior(p)&&p.duration>=900,matched30Plus:p=>hasPrior(p)&&p.duration>=1800,matchedSlow:p=>hasPrior(p)&&p.slow,matchedShort:p=>hasPrior(p)&&p.duration<60,
    wait:p=>p.frame.wait,waitIdentifiable:p=>p.frame.wait&&p.frame.identifiable,waitAmbiguous:p=>p.frame.wait&&!p.frame.identifiable,waitStrict:p=>p.frame.strict,waitWithPrior:p=>p.frame.wait&&hasPrior(p),waitIdentifiableWithPrior:p=>p.frame.identifiable&&hasPrior(p),
    ungatedChanged:p=>values(p,threshold).ungated>p.local,ungatedUnchanged:p=>values(p,threshold).ungated===p.local,strictChanged:p=>values(p,threshold).gated>p.local,strictUnchanged:p=>values(p,threshold).gated===p.local,optimisticChanged:p=>values(p,threshold).optimistic>p.local,optimisticUnchanged:p=>values(p,threshold).optimistic===p.local};
  return Object.fromEntries(Object.entries(filters).map(([k,f])=>[k,score(points.filter(f))]));
}

export function run(root:string,out:string){
  const names=['fold0','fold1','fold2','test'];
  const entries=names.map(n=>JSON.parse(readFileSync(join(root,n,'entries.json'),'utf8')) as Entry[]);
  const dev=entries.slice(0,3).flat(),held=entries[3]!;
  const setHash=(es:Entry[])=>sha(es.map(e=>e.digest).sort().join('\n'));
  const provenance=privateProvenance();
  if(setHash(dev)!==provenance.developmentContentSetSha256||setHash(held)!==provenance.holdoutContentSetSha256||sha(readFileSync(join(root,'test/src/model-data.ts')))!==provenance.refitWeightsSha256||new Set(entries.flat().map(e=>e.id)).size!==entries.flat().length)throw Error('Unexpected frozen split or refit');
  let promptId=0;
  const load=(index:number)=>{
    const cached=readFileSync(join(root,names[index]!,'milestones.jsonl'),'utf8').trimEnd().split('\n').filter(Boolean).map(l=>JSON.parse(l) as Trial);
    const points:Point[]=[];let cursor=0;
    for(const entry of entries[index]!){
      const trace=verificationTrace(frozenText(entry),entry.path);if(trace.id!==entry.id)throw Error('Session identity mismatch');
      for(const frames of organicReplay(trace)){
        const trial=cached[cursor++];if(!trial||trial.session!==entry.id||trial.missing||trial.rows.length!==frames.length)throw Error('Prompt mismatch');
        const id=promptId++;
        for(let i=0;i<frames.length;i++){
          const frame=frames[i]!,row=trial.rows[i]!,e=row.prefix.evidence;
          if(Math.abs(e.elapsedS-frame.elapsedS)>1e-6||Math.abs(frame.elapsedS/60+row.gold.remainingMin-trial.duration)>1e-6)throw Error('Checkpoint mismatch');
          const local=e.localMin*60,gold=row.gold.remainingMin*60;
          if(!Number.isFinite(local)||local<0||!Number.isFinite(gold)||gold<0)throw Error('Invalid score');
          points.push({session:entry.id,prompt:id,duration:trial.duration*60,slow:e.slowTool,gold,frame,local});
        }
      }
    }
    if(cursor!==cached.length)throw Error('Omitted prompts');return {points,prompts:cursor};
  };
  const development=[0,1,2].map(load),test=load(3);
  const check=summarize(test.points,0),medium=check.medium!,long=check.long!,slow=check.slow!;
  if(test.prompts!==reference.completed||check.all!.readings!==reference.stage.slices.all.readings||medium.maeSeconds.local===null||long.maeSeconds.local===null||slow.maeSeconds.local===null||Math.abs((medium.maeSeconds.local+long.maeSeconds.local)/120-reference.stage.balanced.local)>1e-8||Math.abs(slow.maeSeconds.local/60-reference.stage.slices.slow.mae.local)>1e-8)throw Error('Frozen local control mismatch');
  for(const name of ['all','medium','long','30+','slow','short'] as const){
    const actual=check[name]!,expected=reference.stage.slices[name];
    if(actual.sessions!==expected.sessions||actual.prompts!==expected.prompts||actual.readings!==expected.readings||
      actual.maeSeconds.local===null||Math.abs(actual.maeSeconds.local/60-expected.mae.local)>1e-8||
      actual.miss900.local===null||Math.abs(actual.miss900.local-expected.miss15.local)>1e-8||
      actual.forecast900.local===null||Math.abs(actual.forecast900.local-expected.forecast15.local)>1e-8)throw Error('Frozen slice control mismatch');
  }
  const report=(points:Point[])=>({anyPrior:summarize(points,0),priorAtLeast60Seconds:summarize(points,60)});
  writeFileSync(out,JSON.stringify({method:'Seconds; prompt-weighted matched whole-prompt errors. Prior is last successful measured foreground exact command and cwd frozen at launch. Strict targeting is unavailable: ordinary wait is untargeted. Optimistic sole-known-pending wait is not proof of targeting. Handoffs/notices never supply durations. Unknown competitors persist conservatively. All checkpoint evidence is prefix-only.',provenance:{refitWeightsSha256:provenance.refitWeightsSha256},development:report(development.flatMap(x=>x.points)),holdout:report(test.points)},null,2)+'\n',{flag:'wx',mode:0o600});
}
if(import.meta.main){if(process.argv.length!==4)throw Error('Usage: bun eval/wait-gate/organic.ts FROZEN_MILESTONE_DIR NEW_AGGREGATE');run(resolve(process.argv[2]!),resolve(process.argv[3]!));}
