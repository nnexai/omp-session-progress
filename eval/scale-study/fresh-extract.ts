// Local-only discovery of independent Pi and OMP sessions and chronological OMP tails.
import {createHash} from 'node:crypto';
import {existsSync,mkdirSync,readdirSync,readFileSync,statSync,writeFileSync} from 'node:fs';
import {homedir} from 'node:os';
import {join,relative} from 'node:path';
import {activeSegments,extractSessionText,type Trace} from '../extract';
import type {Entry} from '../tool-timing/cohort';

const root=process.argv[2];if(!root)throw Error('Expected private study root');const out=join(root,'extra');
if ((statSync(root).mode&0o077)!==0 || existsSync(out)) throw Error('Expected private root and absent extra directory');
mkdirSync(out,{mode:0o700}); mkdirSync(join(out,'sessions'),{mode:0o700});
const sha=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const fitCutoff=Date.parse('2026-11-25T04:58:45+01:00'); // committed source model; deliberately later than its fit
const old=JSON.parse(readFileSync(join(root,'cohort.json'),'utf8'));
const entries:Entry[]=[...old.legacy.flat(),...old.dev.flat(),...old.official,...old.enriched,...old.fresh];
type Row={type?:string; id?:string; parentId?:string|null; timestamp?:string; version?:number; cwd?:string; title?:string; parentSession?:string; parentSessionId?:string; message?:{role?:string;content?:unknown}};
function parse(text:string):Row[]|null {try{return text.split('\n').filter(Boolean).map(x=>JSON.parse(x));}catch{return null;}}
function textOf(c:unknown):string {return typeof c==='string'?c:Array.isArray(c)?c.filter(x=>x?.type==='text').map(x=>x.text??'').join('\n'):'';}
function promptHash(c:unknown):string|null {const s=textOf(c).replace(/\s+/g,' ').trim().toLowerCase();return s ? sha(s):null;}
const headers=new Set<string>(),ids=new Set<string>(),prompts=new Set<string>(),digests=new Set<string>();
const snapshots=new Map<string,{entry:Entry;bytes:Buffer;last:number}[]>();
for(const e of entries){const b=readFileSync(e.path);if(sha(b)!==e.digest)throw Error('Frozen digest mismatch');const rows=parse(b.toString());const h=rows?.find(r=>r.type==='session');if(!h?.id)throw Error('Missing frozen identity');headers.add(h.id);digests.add(e.digest);for(const r of rows!){if(r.type!=='session'&&r.id)ids.add(r.id);if(r.type==='message'&&r.message?.role==='user'){const p=promptHash(r.message.content);if(p)prompts.add(p);}}const last=Math.max(0,...rows!.map(r=>Date.parse(r.timestamp??'')).filter(Number.isFinite));const a=snapshots.get(h.id)??[];a.push({entry:e,bytes:b,last});snapshots.set(h.id,a);}
const home=homedir(), omp=join(home,'.omp/agent/sessions');
const roots=[{dir:omp,kind:'omp'}, {dir:join(home,'.omp/agent'),kind:'recovery'}, {dir:join(home,'.pi/agent/sessions'),kind:'pi'}, {dir:join(home,'.pi/agent/backups'),kind:'pi-backup'}];
const files:{path:string;kind:string}[]=[];
function walk(dir:string,kind:string,depth=0){if(!existsSync(dir))return;for(const x of readdirSync(dir,{withFileTypes:true})){if(kind==='recovery'&&depth===0&&!x.name.startsWith('launcher-recovery.'))continue;const p=join(dir,x.name);if(x.isDirectory())walk(p,kind,depth+1);else if(x.isFile()&&x.name.endsWith('.jsonl'))files.push({path:p,kind});}}
for(const r of roots)walk(r.dir,r.kind);
const reason:Record<string,number>={}; const reject=(s:string)=>reason[s]=(reason[s]??0)+1;
const results:{pi:Entry[];omp:Entry[];tail:Entry[]}={pi:[],omp:[],tail:[]};
const tailSelection:Record<string,{promptIndices:number[];cutoff:string;prefixDigest:string;sourceDigest:string}>={};
const bands:Record<string,Record<string,number>>={pi:{},omp:{},tail:{}};
const countPrompts:Record<string,number>={pi:0,omp:0,tail:0};
const completed=(tr:Trace,indices:number[])=>indices.filter(i=>tr.prompts[i]?.outcome==='completed');
const duration=(tr:Trace,i:number)=>activeSegments(tr.events,tr.prompts[i]!).reduce((n,s)=>n+s.ms,0);
function save(kind:'pi'|'omp'|'tail',bytes:Buffer,tr:Trace,indices:number[],group:string){const digest=sha(bytes),path=join(out,'sessions',`${tr.id}-${digest.slice(0,12)}.jsonl`);if(!existsSync(path))writeFileSync(path,bytes,{mode:0o600,flag:'wx'});results[kind].push({id:tr.id,path,digest,group});countPrompts[kind]+=indices.length;for(const i of indices){const d=duration(tr,i),b=d>=1800000?'30+':d>=900000?'15-30':d>=180000?'3-15':d>5000?'5s-3m':'0-5s';bands[kind]![b]=(bands[kind]![b]??0)+1;}}
const active=new Set(readFileSync(join(root,'fresh-active-paths.txt'),'utf8').split('\n').filter(Boolean));
// Fence all direct native trainer inputs, including those outside the frozen cohorts.
for(const f of files.filter(x=>x.kind==='omp'&&relative(omp,x.path).split('/').length===2)){const st=statSync(f.path);if(st.size>64*1024*1024)continue;const rs=parse(readFileSync(f.path,'utf8'));if(!rs)continue;for(const r of rs){const t=Date.parse(r.timestamp??'');if(Number.isFinite(t)&&t>fitCutoff)continue;if(r.type!=='session'&&r.id)ids.add(r.id);if(r.type==='message'&&r.message?.role==='user'){const p=promptHash(r.message.content);if(p)prompts.add(p);}}}
for(const f of files.sort((a,b)=>a.path.localeCompare(b.path))){
 const rel=relative(omp,f.path);if(f.kind==='omp'&&(rel.startsWith('..')||rel.split('/').length!==2)){reject('omp-nested-worker');continue;}
 if(f.kind==='pi'&&relative(join(home,'.pi/agent/sessions'),f.path).split('/').length!==2){reject('pi-nested-worker');continue;}
 const st=statSync(f.path);if(active.has(f.path)||st.mtimeMs>Date.now()-10*60_000){reject('active-or-recent');continue;}if(st.size<200||st.size>64*1024*1024){reject('size');continue;}
 const bytes=readFileSync(f.path),rows=parse(bytes.toString());if(!rows){reject('malformed');continue;}const h=rows.find(r=>r.type==='session');if(!h||h.version!==3||!h.id||!h.cwd||h.parentSession||h.parentSessionId||h.parentId){reject('format-or-fork');continue;}
 const title=rows.find(r=>r.type==='title')?.title??h.title??'';const first=rows.find(r=>r.type==='message'&&r.message?.role==='user');
 if(/omp-session-progress|omp-progress-|smoke|fixture|experiment|ratchet|benchmark/i.test(f.path+' '+h.cwd+' '+title)||/worker agent|delegated task|you are a (?:subagent|worker)|⁂ role|cockpit_inbox|coord delegate/i.test(textOf(first?.message?.content))||/\b(?:worker|subagent)\b/i.test(title)){reject('scope-or-worker');continue;}
 const tr=extractSessionText(bytes.toString(),f.path);const isOmp=f.kind==='omp'||f.kind==='recovery';const prior=snapshots.get(h.id);
 if(isOmp&&prior){
  const matches=prior.filter(s=>bytes.length>s.bytes.length&&bytes.subarray(0,s.bytes.length).equals(s.bytes));if(!matches.length){reject('no-appended-exact-prefix');continue;}
  // All snapshots of this identity must match; otherwise no safe chronology is established.
  if(matches.length!==prior.length){reject('divergent-or-unappended-prefix');continue;}
  matches.sort((a,b)=>b.bytes.length-a.bytes.length);const prefix=matches[0]!,last=Math.max(...matches.map(x=>x.last)),cutoff=Math.max(last,fitCutoff);
  const tailRows=parse(bytes.subarray(prefix.bytes.length).toString());if(!tailRows){reject('malformed-tail');continue;}
  const tailStarts=new Set(tailRows.filter(r=>r.type==='message'&&r.message?.role==='user'&&Date.parse(r.timestamp??'')>cutoff).map(r=>`${Date.parse(r.timestamp!)}:${promptHash(r.message?.content)}`));
  const chosen=completed(tr,tr.prompts.map((_,i)=>i)).filter(i=>{const ev=tr.events[tr.prompts[i]!.startIndex]!;const p=promptHash(ev.event.prompt);return p&&!prompts.has(p)&&tailStarts.has(`${ev.t}:${p}`);});
  if(!chosen.length){reject('no-eligible-tail');continue;}
  for(const i of chosen){const p=promptHash(tr.events[tr.prompts[i]!.startIndex]!.event.prompt);if(p)prompts.add(p);}
  save('tail',bytes,tr,chosen,'chronological-same-session');tailSelection[tr.id]={promptIndices:chosen,cutoff:new Date(cutoff).toISOString(),prefixDigest:prefix.entry.digest,sourceDigest:sha(bytes)};continue;
 }
 if(headers.has(h.id)||digests.has(sha(bytes))||rows.some(r=>r.type!=='session'&&!!r.id&&ids.has(r.id))||rows.some(r=>r.type==='message'&&r.message?.role==='user'&&prompts.has(promptHash(r.message.content)??''))){reject('history-or-request-overlap');continue;}
 // Native trainer could have consumed any direct OMP session, regardless of cohort membership.
 if(f.kind==='omp'){reject('possible-omp-pretraining');continue;}
 const chosen=completed(tr,tr.prompts.map((_,i)=>i));if(!chosen.some(i=>duration(tr,i)>5000)){reject('no-scored-prompt');continue;}
 const kind=isOmp?'omp':'pi';save(kind,bytes,tr,chosen,kind==='pi'?'independent-pi-profile':'native-omp-recovery');headers.add(h.id);digests.add(sha(bytes));for(const r of rows){if(r.type!=='session'&&r.id)ids.add(r.id);if(r.type==='message'&&r.message?.role==='user'){const p=promptHash(r.message.content);if(p)prompts.add(p);}}
}
for(const k of ['pi','omp','tail'] as const)writeFileSync(join(out,`${k}-entries.json`),JSON.stringify(results[k]),{mode:0o600,flag:'wx'});
writeFileSync(join(out,'tail-selection.json'),JSON.stringify(tailSelection),{mode:0o600,flag:'wx'});
const summary={discovered:files.length,counts:Object.fromEntries(Object.entries(results).map(([k,v])=>[k,v.length])),completedPrompts:countPrompts,bands,rejected:reason,fitCutoff:new Date(fitCutoff).toISOString(),fitCutoffBasis:'Conservative source-model commit timestamp 1d2301b; later than fit. Exact snapshot digests and last-message timestamps used per tail.',tailWarning:'Same-session only. Replay ONLY promptIndices in tail-selection.json, never the full Entry as a new-session holdout.',piWarning:'Separate Pi profile; not native OMP.'};
writeFileSync(join(out,'summary.json'),JSON.stringify(summary,null,2),{mode:0o600,flag:'wx'});console.log(JSON.stringify(summary));
