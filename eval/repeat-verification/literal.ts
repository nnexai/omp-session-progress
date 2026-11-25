import { forecast,type Model,type Timer } from './model';
import type { ScoredPrompt,TimingPrefix } from './types';
import { summarize } from './metrics';

// Sensitivity analysis, not a replacement chosen using holdout scores. Look up
// the LAST comparable completed call, even if we have only one example and no
// learned curve at the current age. Do not shorten the whole prompt just because
// a prior run has already been outlived: only raise an implausibly short ETA.
export function literal(model:Model,local:number,p:TimingPrefix|null){
  if(!p)return {tool:null,exact:local,pooled:local};
  const remainder=(seconds:number)=>Math.max(0,seconds-p.ageS)/60;
  const x=p.exact.length?remainder(p.exact[p.exact.length-1]!):null;
  const f=p.family.length?remainder(p.family[p.family.length-1]!):null;
  const atStart=forecast(model,p.kind,0,null,null);
  let tool:Timer|null=null;
  if(atStart){const population=remainder(atStart.population*60);
    tool={population,exact:x??population,pooled:x??f??population,used:x!==null?'exact':f!==null?'family':null};}
  return {tool, exact:Math.max(local,x??0),pooled:Math.max(local,x??f??0)};
}
export function literalReport(prompts:ScoredPrompt[]){
  const r=summarize(prompts),changed=prompts.flatMap(p=>p.points.filter(v=>v.pooled!==v.local));
  return {rule:'Post-hoc sensitivity: most recent exact run, else same runner/action/scope. Floor the local ETA at max(0, prior duration minus age), regardless of population support.',
    changedReadings:changed.length,balanced:r.balancedMediumLong,halfHour:r.slices['30+'].mae,slow:r.slices.slow.mae,shortFalse15:r.slices.short.forecast15,
    exactHistory:r.slices.exactMatch,runnerHistoryOnly:r.slices.familyOnly,longWithHistory:r.slices.longMatched,
    toolOnObservedMatches:{calls:r.tool.anySupport.calls,readings:r.tool.anySupport.readings,sessions:r.tool.anySupport.sessions,
      populationMinusAge:r.tool.anySupport.callWeighted.mae.local,latestMatchMinusAge:r.tool.anySupport.callWeighted.mae.pooled,
      contrast:r.tool.anySupport.callWeighted.contrasts.pooledVsLocal}};
}
