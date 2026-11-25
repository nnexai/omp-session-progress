// Local-only cross-harness extraction. Only numeric features and labels are written.
import {createHash} from 'node:crypto';
import {existsSync,readdirSync,readFileSync,statSync,writeFileSync} from 'node:fs';
import {homedir} from 'node:os';
import {join} from 'node:path';
import {scopeBits} from './work-shape';
import {frozenText} from '../frozen-session';
import {privateDir} from '../horizon-study/data';
import type {Split} from './rows';

const root=process.argv[2]; if(!root)throw Error('Expected private study root'); privateDir(root);
const home=homedir(), protectedHashes=new Set<string>();
const normalized=(s:string)=>s.replace(/\s+/g,' ').trim().toLowerCase();
const digest=(s:string)=>createHash('sha256').update(s).digest('hex');
function protect(text:string) { for(const line of text.split('\n'))try { const r=JSON.parse(line); if(r.type!=='message'||r.message?.role!=='user')continue; const c=r.message.content; const s=normalized(typeof c==='string'?c:Array.isArray(c)?c.filter((v:any)=>v.type==='text').map((v:any)=>v.text??'').join('\n'):''); if(s.length>=40)protectedHashes.add(digest(s)); }catch{} }
const cohort:Split=JSON.parse(readFileSync(join(root,'cohort.json'),'utf8'));
for(const e of [...cohort.legacy.flat(),...cohort.dev.flat(),...cohort.official,...cohort.enriched,...cohort.fresh,...cohort.aux])protect(frozenText(e));
function walk(dir:string,out:string[]) { if(!existsSync(dir))return; for(const e of readdirSync(dir,{withFileTypes:true})) { const p=join(dir,e.name); if(e.isDirectory()) { if(!/^(blobs|node_modules|target|fixtures?|experiments?|smoke|test-results)$/i.test(e.name))walk(p,out); } else if(e.isFile()&&e.name.endsWith('.jsonl'))out.push(p); } }
const omp:string[]=[]; for(const p of ['.omp/agent/sessions','.local/state/cockpit/orchestration','.local/share/herdr-kit/coord/sessions','.local/state/cockpit/operations'])walk(join(home,p),omp);
for(const dir of [join(home,'.omp/custom-session-files'),join(home,'.omp/agent/custom-session-files')])if(existsSync(dir))for(const name of readdirSync(dir)){const p=join(dir,name);if(!statSync(p).isFile())continue;const target=readFileSync(p,'utf8').trim();if(target.endsWith('.jsonl')&&['.local/state/cockpit/','.local/share/herdr-kit/coord/sessions/','.omp/agent/sessions/'].some(prefix=>target.startsWith(join(home,prefix)+'/')||target.startsWith(join(home,prefix)))&&existsSync(target)&&statSync(target).isFile())omp.push(target);}
for(const p of omp) protect(readFileSync(p,'utf8'));
const files:string[]=[]; walk(join(home,'.codex/sessions'),files); files.sort();
let roots=0, rejected=0, invalid=0, duplicate=0, prompts=0; const data:number[][]=[];
for(const file of files) {
 let entries:any[]; try {entries=readFileSync(file,'utf8').split('\n').filter(Boolean).map(s=>JSON.parse(s));} catch {invalid++;continue;}
 const metas=entries.filter(e=>e.type==='session_meta'); const meta=metas[0]?.payload;
 if(metas.length!==1 || !meta || meta.source!=='cli' || !['codex-tui','codex_cli_rs'].includes(meta.originator) || (meta.thread_source!==undefined && !['user','User'].includes(meta.thread_source)) || meta.parent_session_id || meta.forked_from_id || typeof meta.cwd!=='string' || /omp-session-progress|fixture|\/tests?\/|smoke|herdr-kit|cockpit\/orchestration/i.test(meta.cwd)) {rejected++;continue;}
 roots++; let active:any=null; let bad=false; let prev=NaN; let seconds=0; let message=''; let userCount=0; let sid=roots;
 for(const r of entries){const p=r.payload??{},t=Date.parse(r.timestamp); if(r.type==='event_msg'&&p.type==='task_started') {active=p.turn_id;bad=false;prev=t;seconds=0;message='';userCount=0;continue;}
  if(!active)continue;
  if(Number.isFinite(t)&&Number.isFinite(prev)&&t>=prev && !(r.type==='event_msg'&&p.type==='token_count')) {seconds+=Math.min(600,(t-prev)/1000);prev=t;} else if(Number.isFinite(t)&&Number.isFinite(prev)&&t<prev)bad=true;
  if(r.type==='event_msg'&&p.type==='user_message'){userCount++;message=typeof p.message==='string'?p.message:'';}
  if(r.type==='event_msg'&&['turn_aborted','task_aborted','error'].includes(p.type))bad=true;
  if(r.type==='event_msg'&&p.type==='task_complete') {if(p.turn_id!==active||p.error||bad||userCount!==1||!message||!Number.isFinite(seconds)||seconds<=5||seconds>21600){invalid++;} else {const n=normalized(message);if(n.length>=40&&protectedHashes.has(digest(n)))duplicate++;else {data.push([sid,scopeBits(message),Math.log1p(Math.min(message.length,100000)),seconds]);prompts++;}} active=null;}
 }
 if(active)invalid++;
}
writeFileSync(join(root,'transfer-data.json'),JSON.stringify(data),{mode:0o600,flag:'wx'});
console.log(JSON.stringify({files:files.length,roots,rejected,invalid,duplicate,prompts,protectedHashes:protectedHashes.size}));
