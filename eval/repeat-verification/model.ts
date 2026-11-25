import { survivalMedian, type Duration } from '../tool-timing/model';
import type { Span, TimingPrefix } from './types';

// Each row is one call, not one display tick. Populate a same-runner curve and
// a cross-runner fallback; thresholds/censor handling are shared with the prior
// survival study. No command identity or session speed is pooled across users.
type Curve = Map<string, Duration[]>;
type Source = 'population'|'exact'|'family';
const groups=(kind:string)=>kind.includes('#')?[kind,kind.split('#')[0]!,'*']:[kind,'*'];
export type Model = Record<Source, Curve>;
function add(curve:Curve,kind:string,row:Duration){
  for(const key of groups(kind)){let rows=curve.get(key);if(!rows)curve.set(key,rows=[]);rows.push(row);}
}
export function fit(spans:Span[][]):Model {
  const model:Model={population:new Map(),exact:new Map(),family:new Map()};
  spans.forEach((calls,session)=>{for(const c of calls){
    if(!Number.isFinite(c.seconds)||c.seconds<0||c.kind==='*')throw Error('Invalid verification span');
    const sample={durationS:c.seconds,completed:c.completed,session};add(model.population,c.kind,sample);
    for(const [source,anchor] of [['exact',c.exactAnchorS],['family',c.familyAnchorS]] as const){
      if(anchor===null)continue;
      // One second of regularization handles subsecond and zero-length prior calls.
      add(model[source],c.kind,{...sample,durationS:c.seconds/(1+anchor)});
    }
  }});
  for(const curve of Object.values(model))for(const rows of curve.values())rows.sort((a,b)=>a.durationS-b.durationS);
  return model;
}
function left(curve:Curve,kind:string,ageS:number,scale:number):number|null {
  for(const key of groups(kind)){const rows=curve.get(key);if(!rows)continue;
    const result=survivalMedian(rows,ageS/scale);
    if(result.enough)return result.leftS===null?null:scale*result.leftS/60;
  }
  return null;
}
export function median(xs:number[]):number|null {
  if(!xs.length)return null;
  const a=[...xs].sort((x,y)=>x-y),m=a.length>>1;
  return a.length%2?a[m]!:(a[m-1]!+a[m]!)/2;
}
export interface Timer {population:number;exact:number;pooled:number;used:'exact'|'family'|null}
/** Positive age conditions both the reference and repeat timer on still running. */
export function forecast(model:Model,kind:string,ageS:number,exactAnchor:number|null,familyAnchor:number|null):Timer|null {
  const population=left(model.population,kind,ageS,1);if(population===null)return null;
  const exact=exactAnchor===null?null:left(model.exact,kind,ageS,1+exactAnchor);
  if(exact!==null)return {population,exact,pooled:exact,used:'exact'};
  const family=familyAnchor===null?null:left(model.family,kind,ageS,1+familyAnchor);
  return {population,exact:population,pooled:family??population,used:family===null?null:'family'};
}
export function predict(model:Model,local:number,p:TimingPrefix|null){
  const tool=p?forecast(model,p.kind,p.ageS,median(p.exact),median(p.family)):null;
  // The displayed local ETA already includes the foreground call. Transfer only
  // the change in its predicted remainder; do not add its entire duration again.
  // Foreground work is serial. Background work can overlap everything else:
  // extend only for newly predicted time beyond both the old timer and the
  // whole prompt ETA, never shorten the prompt because a background job is fast.
  const replace=(minutes:number)=>p?.background
    ? local+Math.max(0,minutes-Math.max(local,tool!.population))
    : Math.max(0,local+(minutes-tool!.population));
  return {tool,exact:tool?.used==='exact'?replace(tool.exact):local,pooled:tool?.used?replace(tool.pooled):local};
}
