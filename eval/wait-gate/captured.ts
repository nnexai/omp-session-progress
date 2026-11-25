import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { extractSessionText, activeSegments } from '../extract';
import { collectTrace } from '../tool-timing/replay';
import { forecast } from './forecast';

type Native = { start:number; end:number; kind:string; text:string; async:boolean };
type Check = { number:number; start:number; end:number; prior:number|null; key:string; code:number };
type Span = { kind:string; check?:number; start:number; end:number };
const seconds = (ns:number) => ns / 1e9;
function requireValue(ok:unknown, message:string): asserts ok { if (!ok) throw Error(message); }

/** Unique independent native call enclosing a private monotonic interval. */
export function aligned(calls:Native[], start:number, end:number, predicate:(c:Native)=>boolean):boolean {
  return calls.filter(c => predicate(c) && c.start <= start + 0.1 && c.end >= end - 0.1 &&
    start - c.start <= 2 && c.end - end <= 2).length === 1;
}

function clockOffset():number {
  // Bun's hrtime can be process-relative. Sample the collector's Linux
  // CLOCK_MONOTONIC and CLOCK_REALTIME together, in its own clock domain.
  const p=Bun.spawnSync(['python3','-B','-c','import time; a=time.monotonic_ns(); w=time.time_ns(); b=time.monotonic_ns(); print(a,w,b)'],{stderr:'ignore'});
  const [a,w,b]=p.stdout.toString().trim().split(' ').map(BigInt);
  requireValue(p.exitCode===0 && a!==undefined && w!==undefined && b!==undefined && b>=a && b-a<100_000_000n,'Unstable clock sample');
  return Number(w-(a+b)/2n)/1e9;
}
function json(path:string):any { return JSON.parse(readFileSync(path, 'utf8')); }
function capture(root:string, last:number) {
  const meta = json(join(root,'capture'));
  requireValue(meta.version === 1 && meta.boot === readFileSync('/proc/sys/kernel/random/boot_id','utf8').trim(), 'Capture is not from this boot');
  const checks:Check[]=[];
  for(let n=1;n<=last;n++) {
    const a=json(join(root,`check-${n}`,'start')), b=json(join(root,`check-${n}`,'end'));
    const priorNs=a.prior_ns ?? null;
    requireValue(Number.isSafeInteger(a.ns) && Number.isSafeInteger(b.ns) && b.ns>a.ns && b.ns-a.ns<86400e9 &&
      typeof a.key==='string' && (priorNs===null || Number.isSafeInteger(priorNs) && priorNs>0) && Number.isInteger(b.code), 'Invalid capture check');
    const prior=checks.filter(c=>c.key===a.key && c.code===0 && c.end<=seconds(a.ns)).sort((x,y)=>y.end-x.end)[0];
    requireValue((priorNs===null && !prior) || (!!prior && priorNs!==null && Math.abs(seconds(priorNs)-(prior.end-prior.start))<0.001), 'Captured prior cannot be validated within selected checks');
    checks.push({number:n,start:seconds(a.ns),end:seconds(b.ns),prior:priorNs===null?null:seconds(priorNs),key:a.key,code:b.code});
  }
  requireValue(checks.every((c,i)=>i===0 || c.start>checks[i-1]!.start), 'Checks out of order');
  const spans:Span[]=readdirSync(root).filter(n=>n.startsWith('span-')).map(n=>json(join(root,n))).filter(s=>s.kind==='join' && Number.isInteger(s.check) && s.check<=last).map(s=>({kind:s.kind,check:s.check,start:seconds(s.start),end:seconds(s.end)}));
  requireValue(spans.every(s=>Number.isFinite(s.start) && Number.isFinite(s.end) && s.start<=s.end && s.end-s.start<86400), 'Invalid captured span');
  return {checks,spans};
}

function snapshot(path:string, afterMs:number):string {
  const lines=readFileSync(path,'utf8').split('\n');
  let ended=false; const prefix:string[]=[];
  for(const line of lines) {
    if(!line.trim()) continue;
    let e:any; try {e=JSON.parse(line);} catch {break;}
    prefix.push(line);
    if(e.type==='message' && e.message?.role==='assistant' && e.message.stopReason==='stop' && Date.parse(e.timestamp)>afterMs) { ended=true; break; }
  }
  requireValue(ended, 'No completed prompt after the captured checks');
  return prefix.join('\n')+'\n';
}

function nativeCalls(text:string):Native[] {
  const entries=text.trim().split('\n').map(line=>JSON.parse(line));
  const starts=new Map<string,number>(), ends=new Map<string,number>();
  for(const e of entries) {
    if(e.type==='custom' && e.customType==='tool_execution_start') starts.set(String(e.data?.toolCallId),Date.parse(e.data?.startedAt)/1000);
    if(e.type==='message' && e.message?.role==='toolResult') ends.set(String(e.message.toolCallId),Date.parse(e.timestamp)/1000);
  }
  const calls:Native[]=[];
  for(const e of entries) if(e.type==='message' && e.message?.role==='assistant' && Array.isArray(e.message.content)) {
    for(const b of e.message.content) if(b.type==='toolCall') {
      const args=b.arguments ?? {}, start=starts.get(String(b.id)), end=ends.get(String(b.id));
      if(start!==undefined && end!==undefined && Number.isFinite(start) && Number.isFinite(end) && end>=start)
        calls.push({start,end,kind:String(b.name),text:typeof args.command==='string'?args.command:typeof args.code==='string'?args.code:'',async:args.async===true || args.background===true || args.timeout===0});
    }
  }
  return calls;
}

function summarize(rows:{local:number;gated:number;ungated:number;checkOnly:number;truth:number;eligible:boolean;waiting:boolean}[]) {
  const mae=(arm:'local'|'gated'|'ungated'|'checkOnly')=>rows.length?rows.reduce((s,r)=>s+Math.abs(r[arm]-r.truth),0)/rows.length:null;
  return {points:rows.length, eligible:rows.filter(r=>r.eligible).length, waiting:rows.filter(r=>r.eligible&&r.waiting).length,
    movedGated:rows.filter(r=>r.gated>r.local).length, movedUngated:rows.filter(r=>r.ungated>r.local).length,
    maeSeconds:{local:mae('local'),gated:mae('gated'),ungated:mae('ungated'),checkOnlyWhileWaiting:mae('checkOnly')}};
}

export function run(root:string, sessionFile:string, last=3) {
  requireValue(Number.isInteger(last)&&last>=1&&last<=100, 'Invalid last check number');
  root=resolve(root);
  const captured=capture(root,last), checks=captured.checks, offset=clockOffset();
  const text=snapshot(sessionFile,(checks.at(-1)!.end+offset)*1000), trace=extractSessionText(text,sessionFile), native=nativeCalls(text);
  const starts=checks.map(c=>(c.start+offset)*1000), ends=checks.map(c=>(c.end+offset)*1000);
  const matches=trace.prompts.filter(p=>p.outcome==='completed' && trace.events[p.startIndex]!.t<=starts[0]! && trace.events[p.endIndex]!.t>=ends.at(-1)!);
  requireValue(matches.length===1, 'Capture does not align to exactly one completed prompt');
  const p=matches[0]!, startMs=trace.events[p.startIndex]!.t, endMs=trace.events[p.endIndex]!.t;
  const spans=captured.spans.filter(s=>(s.start+offset)*1000>=startMs && (s.start+offset)*1000<=endMs);
  for(const c of checks) {
    const at=c.start+offset;
    const escaped=root.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
    const pattern=new RegExp(`capture\\.py\\s+run\\s+['"]?${escaped}['"]?\\s+${c.number}(?:\\s|$)`);
    requireValue(native.filter(n=>n.kind==='bash' && n.async && pattern.test(n.text) && n.start<=at+0.1 && at-n.start<=2).length===1, 'Native dispatch or clock alignment mismatch');
  }
  for(const s of spans) {
    const c=checks.find(c=>c.number===s.check);
    const target=new RegExp(`\\bjoin\\s+\\S+\\s+${s.check}(?!\\d)`);
    requireValue(c && s.start>=c.start && (s.end+offset)*1000<=endMs && aligned(native,s.start+offset,s.end+offset,n=>n.kind==='eval' && !n.async && /\bawait sh\(/.test(n.text) && target.test(n.text)), 'Native targeted join or clock alignment mismatch');
  }
  requireValue(checks.every(c=>spans.some(s=>s.check===c.number)), 'Missing captured join');
  const collected=collectTrace(trace);
  const completed=trace.prompts.filter(q=>q.outcome==='completed');
  const index=completed.indexOf(p), prompt=collected.prompts[index];
  requireValue(prompt, 'Completed prompt replay missing');
  const total=activeSegments(trace.events,p).reduce((s,x)=>s+x.ms,0)/1000;
  const score=(minimum:number)=>{
    const rows=prompt.points.map((point,i)=>{
      const now=point.atMs/1000-offset;
      const running=checks.filter(c=>c.start<=now && now<c.end);
      requireValue(running.length<=1, 'Overlapping captured checks are ambiguous');
      const c=running[0], prior=c && c.prior!==null && c.prior>=minimum?c.prior:null;
      const waiting=!!c && spans.some(s=>s.check===c.number && s.start<=now && now<s.end);
      const f=forecast(point.evidence.localMin*60,prior,c?now-c.start:0,waiting);
      // Diagnostic only: a blocking check need not be the last piece of work.
      return {...f,checkOnly:waiting&&f.left!==null?f.left:f.local,truth:prompt.truth[i]!.remainingS,eligible:prior!==null,waiting,at:now};
    });
    return {all:summarize(rows),eligibleWaiting:summarize(rows.filter(r=>r.eligible&&r.waiting)),
      eligibleNotWaiting:summarize(rows.filter(r=>r.eligible&&!r.waiting)),afterLastExit:summarize(rows.filter(r=>r.at>=checks.at(-1)!.end))};
  };
  return {kind:'One completed real prompt, deliberately scheduled, reconstructed join-entry/return events; check-only arm is a diagnostic, not a proposed prompt ETA.',clock:'same-boot monotonic aligned to native call boundaries; active prompt labels and production observer cadence',
    promptSeconds:total,postLastCheckActiveSeconds:activeSegments(trace.events,p).reduce((s,x)=>s+Math.max(0,Math.min(x.t+x.ms,endMs)-Math.max(x.t,ends.at(-1)!)),0)/1000,
    checks:checks.map(c=>({durationSeconds:c.end-c.start,priorSeconds:c.prior,joins:spans.filter(s=>s.check===c.number).length})),
    missingPoints:prompt.missingPoints,primaryPriorAtLeast60Seconds:score(60),anyPriorDuration:score(0)};
}

if(import.meta.main) {
  const [root,out,last]=process.argv.slice(2), session=process.env.PI_SESSION_FILE;
  requireValue(root && out && session, 'Usage: PI_SESSION_FILE=... bun eval/wait-gate/captured.ts CAPTURE_DIRECTORY NEW_AGGREGATE [LAST_CHECK]');
  const result=run(root,session,last===undefined?3:Number(last));
  writeFileSync(out,JSON.stringify(result,null,2)+'\n',{flag:'wx',mode:0o600});
}
