#!/usr/bin/env bun
// Run inside the private refit source snapshot, not against the shipped weights.
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, realpathSync, statSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { verifyEntry, type Cohort, type Entry } from './cohort';
import { collectSession } from './replay';
import { fitTiming, timing } from './model';
import type { LunaAnswer, LunaInput, Point } from './types';
import { report, type ScoredPrompt, type ScoredPoint } from './summary';

const root=realpathSync(process.argv[2]??'');
if ((statSync(root).mode&0o077)!==0 || realpathSync(resolve(import.meta.dir,'../..'))!==join(root,'source')) throw Error('Use the private refit snapshot');
const expectedModel=process.env.OMP_TOOL_TIMING_EXPECTED_MODEL;
if(!expectedModel||expectedModel.trim()!==expectedModel)throw Error('Set OMP_TOOL_TIMING_EXPECTED_MODEL to the exact expected model identifier');
const split:Cohort=JSON.parse(readFileSync(join(root,'cohort.json'),'utf8'));
const ids=new Set(split.train.map(s=>s.id));
if(split.test.some(s=>ids.has(s.id))||new Set([...split.train,...split.test].map(s=>s.id)).size!==split.train.length+split.test.length)throw Error('Session overlap');
const sessions=(es:Entry[])=>es.map(e=>{verifyEntry(e);const s=collectSession(e.path);if(s.id!==e.id)throw Error('Session identity mismatch');return s;});
const train=sessions(split.train);
const model=fitTiming(train);
// The naive timer knows only the typical completed duration, not survival past the current age.
const times=new Map<string,number[]>();let calls=0,censored=0;
for(const s of train)for(const span of s.spans){calls++;if(!span.completed){censored++;continue;}const rows=times.get(span.family)??[];rows.push(span.durationS);times.set(span.family,rows);}
const medians=new Map([...times].map(([k,r])=>[k,r.sort((a,b)=>a-b)[Math.floor(r.length/2)]!]));
console.log(`Fit on ${train.length} sessions, ${calls} observed tool starts (${censored} right-censored).`);
train.length=0;

function packet(p:Point,postMin:number):LunaInput {
  const e=p.evidence;
  return {phase:e.phase,request:e.request,hasCurrentPlan:e.hasCurrentPlan,open:e.open,done:e.done,openKinds:e.openKinds,currentKinds:e.currentKinds,recent:e.recent,cleanExecAfterEdit:e.cleanExecAfterEdit,errors:e.errors,wrapSignal:e.wrapSignal,family:p.tool!.family,localPostMin:Math.round(postMin*10)/10};
}
interface Issued { index:number; atMs:number; elapsedS:number }
const packets:LunaInput[]=[];const prompts:ScoredPrompt[]=[];
for(const entry of split.test){
  const [s]=sessions([entry]);
  for(const pr of s!.prompts){
    const points:ScoredPoint[]=[];const issued=new Map<number,Issued>();let lastLaunch=-Infinity;
    for(let j=0;j<pr.points.length;j++){
      const p=pr.points[j]!;const base=p.evidence.localMin;const candidate=timing(model,p);
      const treated=candidate!==null && p.tool!==null;
      if(treated && [candidate.toolMin,candidate.postMin,candidate.totalMin].some(n=>!Number.isFinite(n)||n<0))throw Error('Invalid timing prediction');
      // Both the decision to query and what is sent are independent of downstream gold.
      if(treated && p.tool!.ageS>=60 && !issued.has(p.tool!.id) && issued.size<2 && p.elapsedS-lastLaunch>=90){
        issued.set(p.tool!.id,{index:packets.length,atMs:p.atMs,elapsedS:p.elapsedS});lastLaunch=p.elapsedS;
        packets.push(packet(p,candidate.postMin));
      }
      const opinion=treated?issued.get(p.tool!.id):undefined;
      const ref=opinion && p.elapsedS-opinion.elapsedS<=300 ? {index:opinion.index,delayMs:p.atMs-opinion.atMs}:null;
      const {remainingS,toolLeftS}=pr.truth[j]!;
      if(!Number.isFinite(remainingS)||remainingS<0||toolLeftS!==null&&(toolLeftS<0||toolLeftS>remainingS+1e-4))throw Error('Invalid offline label');
      const naive=p.tool&&medians.has(p.tool.family)?Math.max(0,medians.get(p.tool.family)!-p.tool.ageS)/60:null;
      points.push({truth:remainingS/60,local:base,timed:candidate?.totalMin??base,slow:(p.tool?.ageS??0)>60,eligible:(p.tool?.ageS??0)>=30,treated,tool:candidate?.toolMin??null,post:candidate?.postMin??null,goldTool:toolLeftS===null?null:toolLeftS/60,naive,ref});
    }
    const totalMin=points.length?(pr.points[0]!.elapsedS+pr.truth[0]!.remainingS)/60:0;
    prompts.push({session:s!.id,totalMin,missing:pr.missingPoints??0,points});
  }
}
const io=join(root,'luna');mkdirSync(io,{recursive:true,mode:0o700});
const req=join(io,'requests.json'),res=join(io,'answers.json');writeFileSync(req,JSON.stringify(packets),{mode:0o600});
console.log(`Fixed comparison: ${prompts.length} completed holdout prompts; ${packets.length} eligible post-tool queries.`);
if(packets.length===0)throw Error('No eligible Luna comparisons');
// No case identifiers, truth, per-checkpoint outcomes or transcript paths are passed to the provider worker.
const proc=Bun.spawn([process.execPath,join(import.meta.dir,'luna.ts'),req,res],{stdout:'inherit',stderr:'inherit'});
if(await proc.exited!==0)throw Error('Incomplete Luna run; no comparison published');
const ans:LunaAnswer[]=JSON.parse(readFileSync(res,'utf8'));
if(ans.length!==packets.length||ans.some(a=>a.model!==expectedModel||!Number.isFinite(a.latencyMs)||a.latencyMs<0||a.postMin!==null&&(!Number.isFinite(a.postMin)||a.postMin<0||a.postMin>240)))throw Error('Invalid model reply/provenance');
const aggregate=report(prompts,ans);
const result={developmentSessions:split.train.length,holdoutSessions:split.test.length,trainingCalls:calls,censoredCalls:censored,queries:ans.length,abstentions:ans.filter(a=>a.postMin===null).length,reportedCost:ans.reduce((n,a)=>n+a.cost,0),...aggregate};
const sha=(data:string|Buffer)=>createHash('sha256').update(data).digest('hex');
const provenance={expectedModel,developmentContentSetSha256:sha(split.train.map(e=>e.digest).sort().join('\n')),holdoutContentSetSha256:sha(split.test.map(e=>e.digest).sort().join('\n')),refitWeightsSha256:sha(readFileSync(join(root,'source','src','model-data.ts')))};
writeFileSync(join(root,'provenance.json'),JSON.stringify(provenance,null,2)+'\n',{mode:0o600});
writeFileSync(join(root,'aggregate.json'),JSON.stringify(result,null,2)+'\n',{mode:0o600});console.log(JSON.stringify(result,null,2));
