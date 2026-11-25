// One final comparison: the fixed point and range calibrations see only a
// production Observer reading (including its timers, abstentions and smoothing).
import {cpSync,mkdirSync,readFileSync,writeFileSync} from 'node:fs';import {join,resolve} from 'node:path';
import {entries,privateDir,refit} from './data';import {frozenText} from '../frozen-session';import {extractSessionText,activeSegments} from '../extract';import {ReplayClock} from '../tool-timing/replay';import type {Observer as LiveObserver} from '../../src/observer';import {activeMs} from '../../src/facts';
import {calibratedRange as adjust,type RangeModel} from '../../src/range-calibration';import {predict} from './calibration';import {intervalPass,intervals} from './interval-score';import {priority,score,type Observation} from './score';
const [root,corpus,baselineRoot,officialDir,split]=process.argv.slice(2);if(!root||!corpus||!baselineRoot||!officialDir||!['official','enriched'].includes(split!))throw Error('Expected PRIVATE_ROOT FROZEN_FOLDS BASELINE_SOURCE FROZEN_OFFICIAL_INDEX_DIR official|enriched');privateDir(root);
const json=(file:string)=>JSON.parse(readFileSync(file,'utf8'));
const pt=json(join(root,'calibration-model.json')) as number[],range=json(join(root,'range-model.json')) as RangeModel;
let source=resolve(baselineRoot);let load:(i:number)=>ReturnType<typeof extractSessionText>;let n:number;
if(split==='enriched'){
  const test=entries(corpus,'test');n=test.length;load=i=>{const e=test[i]!;return extractSessionText(frozenText(e),e.path);};
  const cached=join(root,'enriched-model.json');let model;
  if(await Bun.file(cached).exists())model=json(cached);else{model=await refit(root,[0,1,2].flatMap(i=>entries(corpus,'fold'+i)),test);writeFileSync(cached,JSON.stringify(model),{mode:0o600,flag:'wx'});}
  const dir=join(root,'replication');mkdirSync(dir);cpSync(join(source,'src'),join(dir,'src'),{recursive:true});source=dir;
  writeFileSync(join(dir,'src/model-data.ts'),`export const MODEL_DATA=${JSON.stringify(model)};`);
}else{
  const index=json(join(officialDir,'index.json')) as Record<string,string>;
  const ids=(json(resolve(import.meta.dir,'../../.omp/ratchet/progress-time/_state.json')).test_ids as string[]).map(s=>s.replace(/-p\d+$/,''));if(new Set(ids).size!==40)throw Error('Official split mismatch');
  n=ids.length;load=i=>{const id=ids[i]!,path=index[id]!;const trace=extractSessionText(readFileSync(path,'utf8'),path);if(trace.id!==id)throw Error('Session mismatch');return trace;};
}
const {Observer}=await import(join(source,'src/observer.ts')) as {Observer:typeof LiveObserver};
const base:Observation[]=[],point:Observation[]=[],ranges:Observation[]=[];
for(let sid=0;sid<n;sid++){
  const trace=load(sid),clock=new ReplayClock(),observer=new Observer(clock);observer.enable('eval',()=>{});let cursor=0;
  for(const [pid,p] of trace.prompts.entries()){
    if(p.outcome!=='completed')continue;const ss=activeSegments(trace.events,p),total=ss.reduce((a,s)=>a+s.ms,0);let j=0,sum=0;
    for(let t=5000;t<total;t=t===5000?15000:t+15000){
      while(sum+ss[j]!.ms<t)sum+=ss[j++]!.ms;const at=ss[j]!.t+t-sum;
      while(cursor<trace.events.length&&trace.events[cursor]!.t<=at){const item=trace.events[cursor++]!;clock.advance(item.t,observer);observer.observe('eval',item.event);}clock.advance(at,observer);
      const {reading,facts}=observer.report(),est=reading?.estimate,v=est?.remainingMinutes==null?null:est.remainingMinutes*60,lo=est?.lowMinutes==null?null:est.lowMinutes*60,hi=est?.highMinutes==null?null:est.highMinutes*60;
      const e=activeMs(facts,reading?.at??at)/1000,o={sid,pid,elapsed:t/1000,y:(total-t)/1000,slow:[...facts.underway.values()].some(x=>x.name!=='ask'&&t/1000-x.startedAtS>60),v,lo,hi};base.push(o);
      point.push({...o,v:v===null?null:predict(pt,e,v)});
      if(v===null||lo===null||hi===null)ranges.push(o);else{const [l,h]=adjust(range,e,v,lo,hi);ranges.push({...o,lo:l,hi:h});}
    }
  }
  observer.disable();
}
const bp=score(base),pc=score(point),br=intervals(base),cr=intervals(ranges);
// Cluster bootstrap sums of prompt scores within sessions (never resample checkpoints).
function paired(){const per=Array.from({length:n},(_,sid)=>{const a=intervals(base.filter(r=>r.sid===sid)),b=intervals(ranges.filter(r=>r.sid===sid));return [a,b] as const;});let state=731;const sample=()=>{state=(Math.imul(state,1664525)+1013904223)>>>0;return Math.floor(state*n/4294967296);};const out:Record<string,number[]>={};for(const k of ['medium','long','halfHour'] as const){const diffs:number[]=[];for(let b=0;b<3000;b++){let sum=0,den=0;for(let i=0;i<n;i++){const [old,fresh]=per[sample()]!;const count=old[k].prompts;if(count){sum+=count*(fresh[k].scoreMin-old[k].scoreMin);den+=count;}}if(den)diffs.push(sum/den);}diffs.sort((a,b)=>a-b);out[k]=[diffs[Math.floor(diffs.length*.025)]!,diffs[Math.floor(diffs.length*.975)]!];}return out;}
const result={sessions:n,basePoint:bp,point:pc,pointPriorityDelta:priority(pc)-priority(bp),baseline:br,range:cr,rangePass:intervalPass(br,cr),rangeScoreDelta95:paired()};writeFileSync(join(root,split+'-result.json'),JSON.stringify(result,null,2),{mode:0o600,flag:'wx'});
console.log(split,'range passes:',result.rangePass,'point priority delta:',result.pointPriorityDelta);for(const k of Object.keys(br) as (keyof typeof br)[])console.log(k,br[k].scoreMin,cr[k].scoreMin,br[k].coverage,cr[k].coverage,'width',br[k].widthMin,cr[k].widthMin);console.log(result.rangeScoreDelta95);
