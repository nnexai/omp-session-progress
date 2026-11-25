// Compare actual independently timed Observers, including live smoothing, ranges and abstentions.
import {cpSync,mkdtempSync,readFileSync,writeFileSync} from 'node:fs';import {join,resolve} from 'node:path';
import {extractSessionText,activeSegments} from '../extract';import {frozenText} from '../frozen-session';import {ReplayClock} from '../tool-timing/replay';import {privateDir,refit} from '../horizon-study/data';import {priority,score,type Observation} from '../horizon-study/score';import {intervals} from '../horizon-study/interval-score';import type {Observer} from '../../src/observer';import type {Split} from './rows';
import {handoffStage} from '../../src/handoff';import {activeMs} from '../../src/facts';
const [root,mode,baseline,candidate,trial='scale']=process.argv.slice(2);if(!root)throw Error('Expected private study root');privateDir(root);
const json=(p:string)=>JSON.parse(readFileSync(p,'utf8'));const cohort:Split=json(join(root,'cohort.json'));
if(mode==='control'){
  const control=await refit(root,cohort.dev.flat().filter(e=>cohort.legacy.flat().some(v=>v.id===e.id)),[...cohort.official,...cohort.enriched,...cohort.fresh]);
  writeFileSync(join(root,'enriched-control.json'),JSON.stringify(control),{mode:0o600,flag:'wx'});
}else{
  if(!baseline||!candidate||!mode||!['enriched','official','pi','omp','managed'].includes(mode))throw Error('Expected ROOT official|enriched|pi|omp|managed BASELINE_DIR CANDIDATE_DIR [trial]');
  const entries:Split['official']=mode==='official'||mode==='enriched'?cohort[mode]:json(join(root,'extra',mode+'-entries.json'));if(!entries.length)throw Error('Empty holdout');
  const include=mode==='managed'?new Set<number>(json(join(root,'extra/managed-pids.json'))):null;
  async function load(dir:string){const dest=mkdtempSync(join(root!,'replay-'));cpSync(join(resolve(dir),'src'),join(dest,'src'),{recursive:true});if(mode==='enriched')writeFileSync(join(dest,'src/model-data.ts'),'export const MODEL_DATA='+JSON.stringify(json(join(root!,'enriched-control.json')))+';');return (await import(join(dest,'src/observer.ts'))).Observer as typeof Observer;}
  const constructors=await Promise.all([load(baseline),load(candidate)]);const readings:[Observation[],Observation[]]=[[],[]];
  const onsets:[Observation[],Observation[]]=[[],[]];
  for(const [sid,entry] of entries.entries()){
    const trace=extractSessionText(frozenText(entry),entry.path);if(trace.id!==entry.id)throw Error('Frozen session mismatch');
    for(const [which,C] of constructors.entries()){
      const clock=new ReplayClock(),obs=new C(clock);obs.enable('eval',()=>{});let cursor=0;
      for(const [pid,p] of trace.prompts.entries()){
        if(p.outcome!=='completed'||(include&&!include.has(sid*100000+pid)))continue;const seg=activeSegments(trace.events,p),total=seg.reduce((s,v)=>s+v.ms,0);let j=0,sum=0;
        const offsets=new Map<number,number>();let accumulated=0,seen=false;for(const s of seg){offsets.set(s.index,accumulated);accumulated+=s.ms;}
        const consume=()=>{const index=cursor++,{t:ts,event}=trace.events[index]!;clock.advance(ts,obs);obs.observe('eval',event);
          if(seen||!offsets.has(index)||trial!=='handoff')return;
          const {facts,reading}=obs.report();if(handoffStage(facts,activeMs(facts,ts)/1000)<0)return;seen=true;
          const elapsed=offsets.get(index)!/1000,est=reading?.estimate,s=(v:number|null|undefined)=>v==null?null:60*v;
          onsets[which]!.push({sid,pid,elapsed,y:total/1000-elapsed,slow:false,v:s(est?.remainingMinutes),lo:s(est?.lowMinutes),hi:s(est?.highMinutes)});
        };
        for(let t=5000;t<total;t=t===5000?15000:t+15000){while(sum+seg[j]!.ms<t)sum+=seg[j++]!.ms;const at=seg[j]!.t+t-sum;
          while(cursor<=p.endIndex&&trace.events[cursor]!.t<=at)consume();clock.advance(at,obs);
          const {facts,reading}=obs.report(),est=reading?.estimate;const secs=(v:number|null|undefined)=>v==null?null:60*v;
          readings[which]!.push({sid,pid,elapsed:t/1000,y:(total-t)/1000,slow:[...facts.underway.values()].some(v=>v.name!=='ask'&&t/1000-v.startedAtS>60),v:secs(est?.remainingMinutes),lo:secs(est?.lowMinutes),hi:secs(est?.highMinutes)});
        }
        while(cursor<=p.endIndex)consume();
      }obs.disable();
    }
  }
  const [a,b]=readings,ctrl=score(a),model=score(b),delta=priority(model)-priority(ctrl);
  // Paired cluster bootstrap: whole sessions, with equal weight per prompt inside each stratum.
  const clusters=entries.map((_,i)=>[score(a.filter(r=>r.sid===i)),score(b.filter(r=>r.sid===i))] as const);
  let seed=731;const draw=()=>Math.floor(((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/4294967296)*entries.length);
  const ci:Record<string,number[]>={};for(const k of ['medium','long','halfHour'] as const){const samples:number[]=[];for(let r=0;r<4000;r++){let n=0,sum=0;for(let i=0;i<entries.length;i++){const [before,after]=clusters[draw()]!,w=before[k].prompts;n+=w;sum+=w*(after[k].maeMin-before[k].maeMin);}if(n)samples.push(sum/n);}samples.sort((a,b)=>a-b);ci[k]=[samples[Math.floor(.025*samples.length)]!,samples[Math.floor(.975*samples.length)]!];}
  const [c,d]=onsets;if(c.length!==d.length||c.some((v,i)=>v.sid!==d[i]!.sid||v.pid!==d[i]!.pid||v.y!==d[i]!.y))throw Error('Handoff event mismatch');
  const result={sessions:entries.length,ctrl,model,priorityDelta:delta,delta95:ci,controlIntervals:intervals(a),modelIntervals:intervals(b),
    onset:{control:score(c),candidate:score(d),controlIntervals:intervals(c),candidateIntervals:intervals(d)}};
  writeFileSync(join(root,`${mode}-${trial}.json`),JSON.stringify(result),{mode:0o600,flag:'wx'});console.log(mode,'full priority delta',delta);
  for(const k of ['short','medium','long','halfHour','all'] as const)console.log(k,'full',ctrl[k].prompts,ctrl[k].maeMin,model[k].maeMin,'at closeout',result.onset.control[k].prompts,result.onset.control[k].maeMin,result.onset.candidate[k].maeMin);
}
