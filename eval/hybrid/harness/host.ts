// Disposable RPC extension used only by run.ts. No historical text is logged.
import { completeSimple } from '@oh-my-pi/pi-ai';
import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { trial } from './trial';
import type { Ask } from '../../../src/hybrid/runtime/forecast';

interface Case { id:string; tag:string; short:number; commitment:string }
interface Job { cases:Case[]; train:string[]; reps:number; concurrency:number; out:string; variant:string }
type Model = Parameters<typeof completeSimple>[0];
interface Context {
 ui:{notify(message:string, type?:'info'|'warning'|'error'):void};
 models:{resolve(spec:string):Model|undefined};
 modelRegistry:{getApiKey(model:Model, sessionId?:string, options?:{signal?:AbortSignal}):Promise<string|undefined>};
}
interface Api { registerCommand(name:string, options:{description:string;handler(args:string, ctx:Context):Promise<void>}):void }
const CASE_MS = 240_000, CALL_MS = 30_000, ATTEMPTS = 3;
class Failure extends Error { constructor(readonly kind:string) { super(kind); } }
const categories = new Set(['investigation','implementation','verification','endToEnd','release','review','deliver']);
const families = new Set(['execute','inspect','edit','delegate','plan','ask','other']);
const phases = new Set(['idle','watching','planning','exploring','editing','verifying','delegating','waiting','retrying','wrapping up','done']);
function numeric(x:unknown): x is number { return typeof x === 'number' && Number.isFinite(x) && x >= 0; }
function enumArray(x:unknown, allowed:Set<string>): boolean { return Array.isArray(x) && x.every(v => typeof v === 'string' && allowed.has(v)); }
// Reconstruct the allowlisted schema rather than persisting arbitrary callback text.
function diagnostic(packet:string, diagnosis:string) {
 const e = JSON.parse(packet) as Record<string, unknown>, d = JSON.parse(diagnosis) as Record<string, unknown>;
 const nums = ['elapsedS','localMin','requestedItems','open','done','calls','errors','cleanExecAfterEdit','edits','delegates'];
 const bools = ['continuation','hasCurrentPlan','slowTool','wrapSignal'];
 if (!e || typeof e !== 'object' || !nums.every(k=>numeric(e[k])) || !bools.every(k=>typeof e[k] === 'boolean') || typeof e.phase !== 'string' || !phases.has(e.phase) || !['lowMin','highMin'].every(k=>e[k] === null || numeric(e[k])) || !['request','openKinds','currentKinds'].every(k=>enumArray(e[k],categories)) || !enumArray(e.recent,families) || !Array.isArray(e.running) || !e.running.every((r:unknown)=>{ if (!r || typeof r !== 'object' || !('kind' in r) || !('ageS' in r) || !('intentKinds' in r)) return false; return typeof r.kind === 'string' && families.has(r.kind) && numeric(r.ageS) && enumArray(r.intentKinds,categories); })) throw new Failure('diagnostic_schema');
 if (!d || !['local_min','hybrid_min','true_min'].every(k=>numeric(d[k]) && Number.isInteger(d[k]))) throw new Failure('diagnostic_schema');
 const safe:Record<string,unknown> = {};
 for (const k of [...nums,...bools,'phase','lowMin','highMin','request','openKinds','currentKinds','recent']) safe[k] = e[k];
 safe.running = (e.running as Array<{kind:string;ageS:number;intentKinds:string[]}>).map(r=>({kind:r.kind,ageS:r.ageS,intentKinds:r.intentKinds}));
 return [{role:'user',content:JSON.stringify(safe)},{role:'assistant',content:JSON.stringify({local_min:d.local_min,hybrid_min:d.hybrid_min,true_min:d.true_min})}];
}
function timeout<T>(p:Promise<T>, signal:AbortSignal):Promise<T> {
 const {promise,resolve,reject}=Promise.withResolvers<T>();
 const abort=()=>reject(new Failure('timeout'));
 if (signal.aborted) { abort(); return promise; }
 signal.addEventListener('abort',abort,{once:true});
 p.then(v=>{signal.removeEventListener('abort',abort);resolve(v);},e=>{signal.removeEventListener('abort',abort);reject(e);});
 return promise;
}
async function pause(ms:number, signal:AbortSignal):Promise<void> {
 const {promise,resolve}=Promise.withResolvers<void>();
 const timer=setTimeout(resolve,ms);
 try { await timeout(promise,signal); } finally { clearTimeout(timer); }
}
export default function hybridEval(api:Api) {
 api.registerCommand('hybrid-eval', { description:'Run private hybrid replay', handler:async (args,ctx)=>{
  try {
   const job:Job = JSON.parse(readFileSync(args.trim(),'utf8'));
   const model = ctx.models.resolve('@widget');
   // Local-only policies remain runnable even if the role cannot resolve; an actual call fails safely.
   const results = join(job.out,'results.jsonl'), errors = join(job.out,'errors.jsonl');
   const completed = new Set<string>();
   if (existsSync(results)) for (const line of readFileSync(results,'utf8').split('\n').filter(Boolean)) { const row=JSON.parse(line); completed.add(`${row.prompt_id}#${row.rep}`); }
   if (job.variant === 'error-pilot' && existsSync(errors)) for (const line of readFileSync(errors,'utf8').split('\n').filter(Boolean)) { const row=JSON.parse(line); if (row.class === 'injected_transport' && row.calls > 0) completed.add(`${row.prompt_id}#${row.rep}`); }
   const queue = job.cases.flatMap(c=>Array.from({length:job.reps},(_,rep)=>({c,rep}))).filter(({c,rep})=>!completed.has(`${c.id}#${rep}`));
   const train = new Set(job.train);
   let errorPilotFailed = false;
   const worker = async ()=>{
    for (let item=queue.shift(); item; item=queue.shift()) {
     const {c,rep}=item; const controller=new AbortController();
     const timer=setTimeout(()=>controller.abort(),CASE_MS);
     const start=Date.now(); let calls=0, retries=0, responseModel:string|undefined, hasUsage=false;
     const usage={input:0,output:0,cacheRead:0,cacheWrite:0}; let cost=0, hasCost=false;
     const trace:Array<{role:string;content:string}>=[];
     let recordedFailure:string|undefined;
     const ask:Ask = async (system,packet,signal)=>{
      calls++;
      if (job.variant === 'error-pilot') { recordedFailure='injected_transport'; throw new Failure(recordedFailure); }
      if (!model) { recordedFailure='model_unavailable'; throw new Failure(recordedFailure); }
      const price=model.cost;
      const knownHaiku=model.provider === 'anthropic' && ['claude-3-haiku-20240307','claude-3-5-haiku-20241022','claude-haiku-4-5-20251001'].includes(model.id);
      const catalogPrice = price && numeric(price.input) && numeric(price.output) && price.input > 0 && price.output > 0;
      if (!(catalogPrice ? price.input <= 2 && price.output <= 10 : knownHaiku)) { recordedFailure='model_price_unapproved'; throw new Failure(recordedFailure); }
      const combined=AbortSignal.any([controller.signal,signal]);
      for (let attempt=0; attempt<ATTEMPTS; attempt++) {
       if (combined.aborted) { recordedFailure='timeout'; throw new Failure(recordedFailure); }
       try {
        const callSignal=AbortSignal.any([combined,AbortSignal.timeout(CALL_MS)]);
        const key=`hybrid-eval-${crypto.randomUUID()}`;
        const started=Date.now();
        const apiKey=await timeout(ctx.modelRegistry.getApiKey(model,key,{signal:callSignal}),callSignal);
        if (!apiKey) throw new Failure('credentials');
        const reply=await timeout(completeSimple(model,{systemPrompt:[system],messages:[{role:'user',content:packet,timestamp:started}]},{apiKey,sessionId:key,maxTokens:240,disableReasoning:true,cacheRetention:'none',signal:callSignal}),callSignal);
        const u=reply.usage;
        if (u) {
         hasUsage=true;
         for (const k of ['input','output','cacheRead','cacheWrite'] as const) { const n=u[k]; if (numeric(n)) usage[k]+=n; }
         if (numeric(u.cost?.total)) {cost+=u.cost.total;hasCost=true;}
        }
        if (!reply.model || reply.model !== model.id || (responseModel && responseModel !== reply.model)) throw new Failure('served_model_mismatch');
        responseModel=reply.model;
        if (reply.stopReason === 'error' || reply.stopReason === 'aborted') throw new Failure('provider_error');
        const text=reply.content.filter(block=>block.type === 'text').map(block=>block.text).join('');
        if (train.has(c.id)) {
         const label=text.trim().toUpperCase();
         trace.push({role:'system',content:system},{role:'user',content:packet},{role:'assistant',content:/^(CLOSE|BOUNDED|MULTI|EXTENDED|UNKNOWN)$/.test(label)?label:'[invalid classification]'});
        }
        return {text,ms:Date.now()-started};
       } catch (e) {
        const kind=e instanceof Failure ? e.kind : 'provider_error';
        if (kind === 'served_model_mismatch' || kind === 'credentials' || combined.aborted || attempt === ATTEMPTS-1) { recordedFailure=combined.aborted ? 'timeout' : kind; throw new Failure(recordedFailure); }
        retries++;
        await pause(500 * 2 ** attempt * (0.5 + Math.random()),combined);
       }
      }
      recordedFailure='provider_error'; throw new Failure(recordedFailure);
     };
     try {
      const result=await timeout(trial(c,ask,job.variant === 'pilot' || job.variant === 'error-pilot' ? 'pilot' : 'policy',controller.signal,train.has(c.id) ? (packet,diagnosis)=>{trace.push(...diagnostic(packet,diagnosis));} : undefined),controller.signal);
      if (recordedFailure) throw new Failure(recordedFailure);
      if (retries) throw new Failure('successful_on_retry');
      if ((job.variant === 'pilot' || job.variant === 'error-pilot') && !calls) throw new Failure('pilot_no_call');
      if (!result || !result.grade || typeof result.grade !== 'object' || !Object.values(result.grade).every(v=>typeof v === 'number' && Number.isFinite(v)) || typeof result.summary !== 'string') throw new Failure('grader_error');
      if (train.has(c.id)) { mkdirSync(join(job.out,'traces'),{recursive:true,mode:0o700}); writeFileSync(join(job.out,'traces',`${c.id}_rep${rep}.json`),JSON.stringify(trace),{mode:0o600}); }
      const row={prompt_id:c.id,rep,prompt:`${c.id}: historic session`,tags:[c.tag],grade:result.grade,summary:result.summary,explanation:result.summary,status:'ok',calls,retries,latency_s:(Date.now()-start)/1000,estimated_cost_usd:hasCost ? cost : calls === 0 ? 0 : undefined,...(responseModel ? {model:responseModel} : {}),...(hasUsage ? {usage} : {}),...(job.variant === 'pilot' ? {pilot:true} : {})};
      appendFileSync(results,`${JSON.stringify(row)}\n`,{mode:0o600});
     } catch (e) {
      controller.abort();
      const kind=recordedFailure ?? (e instanceof Failure ? e.kind : 'harness_error');
      if (job.variant === 'error-pilot' && (kind !== 'injected_transport' || calls < 1)) errorPilotFailed = true;
      appendFileSync(errors,`${JSON.stringify({prompt_id:c.id,rep,class:kind,calls,retries,latency_s:(Date.now()-start)/1000,...(responseModel ? {model:responseModel} : {}),...(hasUsage ? {usage} : {}),...(hasCost ? {estimated_cost_usd:cost} : {})})}\n`,{mode:0o600});
     } finally {clearTimeout(timer); controller.abort();}
    }
   };
   await Promise.all(Array.from({length:job.concurrency},()=>worker()));
   ctx.ui.notify(errorPilotFailed ? 'hybrid-eval: failed' : 'hybrid-eval: done',errorPilotFailed ? 'error' : 'info');
  } catch { ctx.ui.notify('hybrid-eval: failed','error'); }
 }});
}
