import { extractSessionText, type Trace } from '../extract';
import type { ObservedEvent } from '../../src/facts';
import { runnerFamily } from './runner';

// IDs are interned per trace; neither commands, directories nor native job IDs
// leave these ephemeral extraction closures.
export interface VerificationIdentity { exact:number; kind:string }
export interface Notice { at:number; job:number }
const identities = new WeakMap<ObservedEvent, VerificationIdentity>();
const handoffs = new WeakMap<ObservedEvent, number>();
const notices = new WeakMap<Trace, Notice[]>();
const receipts=new WeakMap<ObservedEvent,number>();
export const verificationReceipt=(event:ObservedEvent)=>receipts.get(event);
export const verificationIdentity = (event:ObservedEvent) => identities.get(event);
export const verificationHandoff = (event:ObservedEvent) => handoffs.get(event);
export const verificationNotices = (trace:Trace) => notices.get(trace) ?? [];

function textOf(content:unknown):string {
  if(typeof content==='string') return content;
  if(!Array.isArray(content)) return '';
  return content.filter((b):b is {type:'text';text:string}=>!!b && b.type==='text' && typeof b.text==='string').map(b=>b.text).join('\n');
}

/** Only direct finite verification invocations are eligible, including async. */
export function verificationTrace(text:string,path:string):Trace {
  const intern=new Map<string,number>(), jobs=new Map<string,number>(), pending:Notice[]=[];
  const jobId=(s:string)=>{let id=jobs.get(s); if(id===undefined){id=jobs.size;jobs.set(s,id);} return id;};
  const trace=extractSessionText(text,path,{onToolStart(event,args,cwd){
    if(event.type!=='tool_execution_start' || !['bash','exec','shell'].includes(String(event.toolName))) return;
    if(args.background===true || args.timeout===0 || args.name!=null || args.ready!=null) return;
    const command=typeof args.command==='string'?args.command:typeof args.cmd==='string'?args.cmd:null;
    if(command===null) return;
    const kind=runnerFamily(command);
    if(!kind) return;
    const key=JSON.stringify([typeof args.cwd==='string'?args.cwd:cwd,command]);
    let exact=intern.get(key);
    if(exact===undefined) { exact=intern.size; intern.set(key,exact); }
    identities.set(event,{exact,kind});
  },
    onToolEnd(event,content) {
      // The native wait tool can deliver the completion instead of a custom
      // notice. Match only its first structured header, never quoted stdout.
      if(event.toolName==='wait' && !event.isError){
        const receipt=/^## Completed \(\d+\)\s*### (bg_[\w-]+) \[bash\] — completed\b/.exec(textOf(content));
        if(receipt)receipts.set(event,jobId(receipt[1]!));
        return;
      }
      if(!['bash','exec','shell'].includes(String(event.toolName))) return;
      const d=event.details && typeof event.details==='object' ? event.details as Record<string,unknown> : {};
      const structured=(d.backgrounded===true || d.status==='backgrounded') && typeof d.jobId==='string' && /^bg_[\w-]+$/.test(d.jobId) ? d.jobId : null;
      const preamble=/^Backgrounded as job (bg_[\w-]+)(?=[.\s]|$)/.exec(textOf(content));
      const job=structured ?? preamble?.[1];
      if(job) handoffs.set(event,jobId(job));
    },
    onCustomMessage(type,content,at) {
      if(type!=='async-result') return;
      const value=textOf(content);
      const single=/^<system-notice>\r?\nBackground job (bg_[\w-]+) \([^\r\n]*\) has completed\./.exec(value);
      if(single) { pending.push({at,job:jobId(single[1]!)}); return; }
      // Only the initial batch header is unambiguous; later headers could be
      // text printed by a job and are not authenticated delimiters.
      const first=/^<system-notice>\r?\n(?:\d+ background jobs have completed\.[^\r\n]*\s+)?── Job (bg_[\w-]+)(?=[\s:─(]|$)[^\r\n]*? ──/.exec(value);
      if(first) pending.push({at,job:jobId(first[1]!)});
    },
  });
  notices.set(trace,pending.sort((a,b)=>a.at-b.at));
  return trace;
}
