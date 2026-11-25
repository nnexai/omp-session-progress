import { readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { frozenText, sha } from '../frozen-session';
import type { Entry } from '../tool-timing/cohort';
import type { Trial } from '../milestones/types';
import reference from '../milestones/result.json';
import { verificationTrace, replayVerifications } from './replay';
import { fit, predict, type Model } from './model';
import { dispatchReport } from './dispatch';
import { summarize } from './metrics';
import type { Replay, ScoredPrompt, Span } from './types';
import { literal, literalReport } from './literal';

const root=resolve(process.argv[2]??'.'), out=process.argv[3];
if(!out)throw Error('Usage: bun eval/repeat-verification/study.ts FROZEN_MILESTONE_DIRECTORY NEW_AGGREGATE_JSON');
const names=['fold0','fold1','fold2','test'];
const entries=names.map(n=>JSON.parse(readFileSync(join(root,n,'entries.json'),'utf8')) as Entry[]);
const development=entries.slice(0,3).flat(),testEntries=entries[3]!;
const setHash=(es:Entry[])=>sha(es.map(e=>e.digest).sort().join('\n'));
const {provenance}=reference;
if(setHash(development)!==provenance.developmentContentSetSha256 || setHash(testEntries)!==provenance.holdoutContentSetSha256 ||
    sha(readFileSync(join(root,'test/src/model-data.ts')))!==provenance.refitWeightsSha256 ||
    new Set(entries.flat().map(e=>e.id)).size!==entries.flat().length)throw Error('Unexpected session split or refit control');

interface Session {id:string; replay:Replay; trials:Trial[]}
function load(index:number):Session[]{
  const cached:Trial[]=readFileSync(join(root,names[index]!,'milestones.jsonl'),'utf8').trim().split('\n').map(l=>JSON.parse(l));
  let cursor=0;
  const sessions=entries[index]!.map(entry=>{
    const trace=verificationTrace(frozenText(entry),entry.path),replay=replayVerifications(trace);
    if(trace.id!==entry.id)throw Error('Unexpected session identity');
    const trials=replay.prompts.map(frames=>{const t=cached[cursor++];
      if(!t||t.session!==trace.id||t.missing||frames.length!==t.rows.length)throw Error('Completed prompt or reading mismatch');
      for(let i=0;i<frames.length;i++){const f=frames[i]!,r=t.rows[i]!;
        if(Math.abs(r.prefix.evidence.elapsedS-f.elapsedS)>1e-6||Math.abs(f.elapsedS/60+r.gold.remainingMin-t.duration)>1e-6)throw Error('Checkpoint changed');
        if(f.toolLeftS!==null&&(!Number.isFinite(f.toolLeftS)||f.toolLeftS<0||f.toolLeftS>r.gold.remainingMin*60+1e-6))throw Error('Invalid tool label');
      }
      return t;
    });return {id:entry.id,replay,trials};
  });
  if(cursor!==cached.length)throw Error('Omitted reference prompts');return sessions;
}
const spans=(ss:Session[]):Span[][]=>ss.map(s=>s.replay.spans);
function score(ss:Session[],model:Model,timer:typeof predict=predict):ScoredPrompt[]{return ss.flatMap(s=>s.trials.map((t,i)=>({session:s.id,duration:t.duration*60,points:t.rows.map((r,j)=>{
  const e=r.prefix.evidence,f=s.replay.prompts[i]![j]!;
  // Neither actual prompt remainder nor actual call duration enters predict().
  const {tool,exact,pooled}=timer(model,e.localMin,f.timing);
  if(![exact,pooled].every(v=>Number.isFinite(v)&&v>=0))throw Error('Invalid whole-prompt forecast');
  return {local:e.localMin,exact,pooled,slow:e.slowTool,remaining:r.gold.remainingMin,timing:f.timing,tool:tool&&{...tool,gold:f.toolLeftS===null?null:f.toolLeftS/60}};
})})));}

const dev=[0,1,2].map(load);
const model=fit(spans(dev.flat()));
// A second, development-only readout uses session-OOF duration curves AND the
// cached session-OOF local estimates. No parameters are selected on either readout.
const cv=summarize(dev.flatMap((ss,i)=>score(ss,fit(spans(dev.filter((_,j)=>j!==i).flat())))));
console.log(`Froze tool-duration curves on ${development.length} development sessions.`);
const held=load(3),points=score(held,model),report=summarize(points);
if(report.completedPrompts!==reference.completed||report.slices.all.readings!==reference.stage.slices.all.readings||
    Math.abs(report.balancedMediumLong.mae.local!-reference.stage.balanced.local)>1e-8||
    Math.abs(report.slices.slow.mae.local!-reference.stage.slices.slow.mae.local)>1e-8)throw Error('Displayed control differs from frozen reference');
const devSpans=spans(dev.flat()).flat();
const result={provenance,development:{sessions:development.length,trainingCalls:devSpans.length,completed:devSpans.filter(s=>s.completed).length,background:devSpans.filter(s=>s.background).length,
  terminalNotices:devSpans.filter(s=>s.noticeOnly).length,
  crossValidated:{balancedMediumLong:cv.balancedMediumLong,halfHour:cv.slices['30+'],coverage:cv.coverage,exactTool:cv.tool.exactSupport.callWeighted,anyTool:cv.tool.anySupport.callWeighted}},
  heldoutSessions:held.length,atDispatch:dispatchReport(model,spans(held)),...report,
  literalSensitivity:literalReport(score(held,model,literal))};
writeFileSync(out,JSON.stringify(result,null,2)+'\n',{flag:'wx',mode:0o600});
console.log('Reproduced the displayed local control; wrote aggregate-only results.');
