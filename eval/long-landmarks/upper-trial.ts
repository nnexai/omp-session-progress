// One preregistered, research-only 10-active-minute landmark trial.
// Usage: bun eval/long-landmarks/upper-trial.ts PRIVATE_SCALE_ROOT PRIVATE_REFIT_ROOT PRIVATE_WORK_ROOT OUTPUT_JSON
// Before holdout replay: fit three whole-session OOF copies of frozen v4 (0bada07).
// At 600 s, define high score as >= the development OOF 75th percentile of
// the *displayed point*; fit its 75th percentile of remaining minutes.
// Keep the actual lower bound and raise, but never shrink, the upper bound to
// that quantile. The smoothed point and all other readings remain unchanged.
// The display metric runs through the actual formatter, treating point fallback
// as a degenerate interval. No holdout-derived threshold or iteration.
// Acceptance, separately in both holdouts: high-score paired score improves with
// bootstrap 95% upper bound < 0; all-risk score improves; all-risk coverage is
// >= min(0.45, baseline); shorter-than-15m score does not worsen. Report only;
// this criterion does not authorize deployment.
import {existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync} from 'node:fs';
import {join, resolve} from 'node:path';
import {spawnSync} from 'node:child_process';
import {frozenText} from '../frozen-session';
import {extractSessionText, activeSegments} from '../extract';
import {ReplayClock} from '../tool-timing/replay';
import {privateDir} from '../horizon-study/data';
import type {Split} from '../scale-study/rows';
import type {Observer} from '../../src/observer';
import {formatRemaining} from '../../src/widget';
const [scaleArg, refitArg, workArg, out] = process.argv.slice(2);
if (!scaleArg || !refitArg || !workArg || !out) throw Error('Expected four arguments');
const scale=resolve(scaleArg), refitRoot=resolve(refitArg), work=resolve(workArg); [scale,refitRoot,work].forEach(privateDir);
const cohort:Split=JSON.parse(readFileSync(join(scale,'cohort.json'),'utf8'));
const groups=[...cohort.dev,cohort.official,cohort.enriched,cohort.fresh];
const ids=new Set<string>(); for(const e of groups.flat()){if(ids.has(e.id))throw Error('Non-disjoint cohorts');ids.add(e.id);}
const archive=spawnSync('git',['archive','refs/tags/replay/0bada07','src','eval'],{maxBuffer:100*1024*1024}); if(archive.status!==0)throw Error('Archive failed');
const excluded=[...cohort.official,...cohort.enriched,...cohort.fresh];
const dirs:string[]=[]; const jobs:Promise<void>[]=[];
for(let fold=0;fold<3;fold++){
 const found=readdirSync(work).filter(s=>s.startsWith(`upper-fold${fold}-`));
 if(found.length>1)throw Error('Ambiguous fold fit');
 if(found.length){const dir=join(work,found[0]!);if(existsSync(join(dir,'home')))throw Error('Incomplete fold fit');const held:string[]=JSON.parse(readFileSync(join(dir,'.omp/ratchet/progress-time/_state.json'),'utf8')).test_ids;if(![...cohort.dev[fold]!,...excluded].every(e=>held.includes(e.id)))throw Error('Fold exclusion mismatch');dirs.push(dir);continue;}
 const dir=mkdtempSync(join(work,`upper-fold${fold}-`));dirs.push(dir);
 if(spawnSync('tar',['-x','-C',dir],{input:archive.stdout}).status!==0)throw Error('Unpack failed');
 const sessions=join(dir,'home/.omp/agent/sessions/frozen'),state=join(dir,'.omp/ratchet/progress-time');mkdirSync(sessions,{recursive:true,mode:0o700});mkdirSync(state,{recursive:true,mode:0o700});
 const train=cohort.dev.flatMap((g,i)=>i===fold?[]:g); for(const e of train)writeFileSync(join(sessions,e.id+'.jsonl'),frozenText(e),{flag:'wx',mode:0o600});
 writeFileSync(join(state,'_state.json'),JSON.stringify({test_ids:[...cohort.dev[fold]!,...excluded].map(e=>e.id)}),{mode:0o600});
 const child=Bun.spawn([process.execPath,join(dir,'eval/train.ts')],{env:{...process.env,HOME:join(dir,'home')},stdout:'ignore',stderr:'ignore'});
 jobs.push((async()=>{if(await child.exited!==0)throw Error(`Training fold ${fold} failed`);rmSync(join(dir,'home'),{recursive:true});})());
}
await Promise.all(jobs);
type Row={sid:number; y:number; total:number; point:number|null; lo:number|null; hi:number|null};
type C=typeof Observer;
async function replay(entries:Split['official'], Ctor:C):Promise<Row[]>{
 const rows:Row[]=[];
 for(const [sid,entry] of entries.entries()){
  const trace=extractSessionText(frozenText(entry),entry.path);if(trace.id!==entry.id)throw Error('Frozen identity mismatch');
  const clock=new ReplayClock(),obs=new Ctor(clock);obs.enable('eval',()=>{});let cursor=0;
  for(const p of trace.prompts){if(p.outcome!=='completed')continue;const seg=activeSegments(trace.events,p),total=seg.reduce((s,v)=>s+v.ms,0);let j=0,sum=0;
   for(let t=5000;t<total&&t<=600000;t=t===5000?15000:t+15000){while(sum+seg[j]!.ms<t)sum+=seg[j++]!.ms;const at=seg[j]!.t+t-sum;
    while(cursor<=p.endIndex&&trace.events[cursor]!.t<=at){const {t:ts,event}=trace.events[cursor++]!;clock.advance(ts,obs);obs.observe('eval',event);}clock.advance(at,obs);
    if(t===600000){const {facts,reading}=obs.report(),est=reading?.estimate;const valid=facts.running&&!facts.asking.size&&est?.progress!=null&&est.remainingMinutes!=null;
     rows.push({sid,y:(total-t)/60000,total:total/60000,point:valid?est!.remainingMinutes:null,lo:valid?est!.lowMinutes:null,hi:valid?est!.highMinutes:null});}
   }
   while(cursor<=p.endIndex){const {t,event}=trace.events[cursor++]!;clock.advance(t,obs);obs.observe('eval',event);}
  }obs.disable();
 }return rows;
}
function pointValue(v:number){const s=formatRemaining(v,null,null,true);if(s==='<30s')return .25;const m=/^~(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+)s)?$/.exec(s);if(!m)throw Error('Point format');return +(m[1]??0)*60+ +(m[2]??0)+ +(m[3]??0)/60;}
function quantile(a:number[],p:number){if(!a.length)throw Error('Empty dev data');a.sort((x,y)=>x-y);return a[Math.floor((a.length-1)*p)]!;}
let dev:Row[]=[];for(let i=0;i<3;i++){const Ctor=(await import(join(dirs[i]!,'src/observer.ts'))).Observer as C;const r=await replay(cohort.dev[i]!,Ctor);for(const x of r)x.sid=x.sid*3+i;dev.push(...r);}
const usable=dev.filter(r=>r.point!==null), cutoff=quantile(usable.map(r=>pointValue(r.point!)),.75),high=usable.filter(r=>pointValue(r.point!)>=cutoff);
const upper=quantile(high.map(r=>r.y),.75);
// Train all cutoffs and bounds above before importing or scoring either holdout.
const officialDir=mkdtempSync(join(work,'upper-official-'));if(spawnSync('tar',['-x','-C',officialDir],{input:archive.stdout}).status!==0)throw Error('Unpack failed');
const clean=join(readFileSync(join(refitRoot,'refit-root.txt'),'utf8').trim(),'0bada07');
const protectedIds:string[]=JSON.parse(readFileSync(join(clean,'.omp/ratchet/progress-time/_state.json'),'utf8')).test_ids;
if(!excluded.every(e=>protectedIds.includes(e.id)))throw Error('Refit exclusion mismatch');
const result:any={design:{revision:'0bada07',landmarkActiveMinutes:10,training:'three whole-session OOF frozen v4 fits; high-score threshold and upper quantile from OOF dev only',timers:'ReplayClock, Observer ticks/events and 5s then 15s observation cadence',target:'displayed point >= OOF dev 75th percentile; raise only existing upper bound to the conditional remaining P75; retain lower bound and smoothed point',acceptance:'Both cohorts: high-score displayed proper score decreases with session-bootstrap 95% upper bound below zero; all-risk displayed score decreases and coverage >= min(0.45, baseline); shorter-than-15-minute score does not increase',limitations:'Landmark only; completed prompts surviving 10 active minutes. Not a full-trajectory trial; holdouts previously studied; no deployment.'},development:{eligible:dev.length,usable:usable.length,high:high.length,cutoffDisplayedMinutes:cutoff,upperMinutes:upper},cohorts:{}};
function displayed(r:Row,changed:boolean):[number,number,boolean]{const p=r.point!;const l=r.lo;let h=r.hi;if(changed&&pointValue(p)>=cutoff&&l!==null&&h!==null)h=Math.max(h,upper);const s=formatRemaining(p,l,h,false),m=/^~(\d+)–(\d+)m/.exec(s);if(m)return [+m[1]!,+m[2]!,true];const v=pointValue(p);return [v,v,false];}
function summarize(rows:Row[]){const n=rows.length;if(!n)return {n:0};const v=rows.map(r=>{const a=displayed(r,false),b=displayed(r,true);const calc=(q:[number,number,boolean])=>({score:q[1]-q[0]+4*Math.max(0,q[0]-r.y,r.y-q[1]),width:q[1]-q[0],coverage:+(q[0]<=r.y&&r.y<=q[1]),range:+q[2],alert:+(q[1]>=20)});return {sid:r.sid,a:calc(a),b:calc(b)};});
 const mean=(arm:'a'|'b',key:'score'|'width'|'coverage'|'range'|'alert')=>v.reduce((s,x)=>s+x[arm][key],0)/n;
 const cluster=[...new Set(v.map(x=>x.sid))].map(s=>v.filter(x=>x.sid===s));let seed=731;const diffs:number[]=[];for(let b=0;b<4000;b++){let sum=0,k=0;for(let i=0;i<cluster.length;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;for(const x of cluster[Math.floor(seed/4294967296*cluster.length)]!){sum+=x.b.score-x.a.score;k++;}}diffs.push(sum/k);}diffs.sort((a,b)=>a-b);
 return {n,sessions:cluster.length,baseline:{score:mean('a','score'),width:mean('a','width'),coverage:mean('a','coverage'),rangeRate:mean('a','range'),upperAtLeast20Rate:mean('a','alert')},candidate:{score:mean('b','score'),width:mean('b','width'),coverage:mean('b','coverage'),rangeRate:mean('b','range'),upperAtLeast20Rate:mean('b','alert')},pairedScoreDelta95:[diffs[100],diffs[3900]]};}
for(const [name,dir] of [['official',officialDir],['enriched',clean]] as const){const Ctor=(await import(join(dir,'src/observer.ts'))).Observer as C;const rows=await replay(cohort[name],Ctor),valid=rows.filter(r=>r.point!==null);const all=summarize(valid),scored=summarize(valid.filter(r=>pointValue(r.point!)>=cutoff)),long=summarize(valid.filter(r=>r.total>=30)),short=summarize(valid.filter(r=>r.total<15));
 const pass=!!(scored.n&&all.n&&short.n&&scored.candidate!.score<scored.baseline!.score&&scored.pairedScoreDelta95![1]!<0&&all.candidate!.score<all.baseline!.score&&all.candidate!.coverage>=Math.min(.45,all.baseline!.coverage)&&short.candidate!.score<=short.baseline!.score);
 result.cohorts[name]={sessions:cohort[name].length,riskSet:rows.length,eligible:valid.length,abstained:rows.length-valid.length,all,highScored:scored,trueLong30Plus:long,shorterThan15:short,pass};}
writeFileSync(out,JSON.stringify(result,null,2)+'\n',{flag:'wx',mode:0o600});
