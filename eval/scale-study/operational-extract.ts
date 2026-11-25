// Local-only native managed-actor corpus. Never report transcript content or paths.
import {createHash} from 'node:crypto';
import {existsSync, readdirSync, readFileSync, statSync, mkdirSync, writeFileSync} from 'node:fs';
import {homedir} from 'node:os';
import {join,resolve} from 'node:path';
import {activeSegments,extractSessionText} from '../extract';
import {frozenText} from '../frozen-session';
import {privateDir} from '../horizon-study/data';
import {rowsFor,type Split} from './rows';
import type {Entry} from '../tool-timing/cohort';
const root=process.argv[2]; if(!root)throw Error('private root required'); privateDir(root);
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const scope=/omp-session-progress|omp-progress-|fixture|smoke|experiment|ratchet|benchmark/i;
type Meta={h:any;rows:any[];ids:Set<string>;requests:Set<string>;family:string};
function metadata(text:string):Meta|null {
 let rows:any[];try{rows=text.split('\n').filter(Boolean).map(JSON.parse)}catch{return null}
 const h=rows.find(r=>r.type==='session');if(!h||h.version!==3||typeof h.id!=='string'||typeof h.cwd!=='string')return null;
 const ids=new Set<string>(),requests=new Set<string>();
 for(const r of rows){if(r.type!=='session'&&typeof r.id==='string')ids.add(r.id);if(r.type==='message'&&r.message?.role==='user'){const c=r.message.content;const t=(typeof c==='string'?c:Array.isArray(c)?c.filter((b:any)=>b.type==='text').map((b:any)=>b.text??'').join('\n'):'').replace(/\s+/g,' ').trim().toLowerCase();if(t)requests.add(hash(t));}}
 const cwd=h.cwd.toLowerCase();const project=cwd.match(/\/(?:dev\/prj|worktrees|projects)\/([^/]+)/)?.[1]??cwd;
 return {h,rows,ids,requests,family:hash(project)};
}
const cohort:Split=JSON.parse(readFileSync(join(root,'cohort.json'),'utf8'));const ids=new Set<string>(),requests=new Set<string>(),headers=new Set<string>();
function register(m:Meta){headers.add(m.h.id);for(const x of m.ids)ids.add(x);for(const x of m.requests)requests.add(x)}
for(const e of [...cohort.dev.flat(),...cohort.legacy.flat(),...cohort.official,...cohort.enriched,...cohort.fresh,...cohort.aux]){const m=metadata(frozenText(e));if(!m)throw Error('invalid frozen entry');register(m)}
const dirs=[join(homedir(),'.local/state/cockpit/operations'),join(homedir(),'.local/share/herdr-kit/coord/sessions')];const found=new Set<string>();
function walk(p:string){if(!existsSync(p))return;for(const e of readdirSync(p,{withFileTypes:true})){const q=join(p,e.name);if(e.isDirectory()){if(!/^(blobs|node_modules|target|fixtures?|experiments?|smoke|test-results)$/i.test(e.name))walk(q)}else if(e.isFile()&&e.name.endsWith('.jsonl'))found.add(q)}}
for(const d of dirs)walk(d);
for(const d of [join(homedir(),'.omp/custom-session-files'),join(homedir(),'.omp/agent/custom-session-files')])if(existsSync(d))for(const name of readdirSync(d)){const p=join(d,name);if(!statSync(p).isFile())continue;const q=resolve(readFileSync(p,'utf8').trim());if(dirs.some(x=>q.startsWith(x+'/'))&&q.endsWith('.jsonl')&&existsSync(q)&&statSync(q).isFile())found.add(q)}
type Candidate={text:string;id:string;family:string;rank:string;duration:number;work:Set<number>};const pool:Candidate[]=[];const rejected:Record<string,number>={};const reject=(s:string)=>{rejected[s]=(rejected[s]??0)+1};
for(const path of [...found].sort()){
 const st=statSync(path);if(st.size<200||st.size>64*1024*1024||st.mtimeMs>Date.now()-86400000){reject('size-or-recent');continue}if(scope.test(path)){reject('scope');continue}
 const text=readFileSync(path,'utf8'),after=statSync(path);if(after.size!==st.size||after.mtimeMs!==st.mtimeMs){reject('changed-during-read');continue}const m=metadata(text);if(!m){reject('format');continue}const title=m.rows.find(r=>r.type==='title')?.title??m.h.title??'';
 const first=m.rows.find(r=>r.type==='message'&&r.message?.role==='user');
 if(scope.test(m.h.cwd+' '+title+' '+JSON.stringify(first?.message?.content??''))){reject('scope');continue}if(headers.has(m.h.id)||[...m.ids].some(x=>ids.has(x))||[...m.requests].some(x=>requests.has(x))){reject('overlap');continue}
 const children=new Set<string>();let branched=false;for(const r of m.rows)if(typeof r.parentId==='string'){if(children.has(r.parentId))branched=true;children.add(r.parentId)}if(branched){reject('branched');continue}
 const args=new Map<string,string>();const trace=extractSessionText(text,path,{onToolStart:(event:any,raw:any)=>args.set(event.toolCallId,JSON.stringify(raw))});let duration=0;const work=new Set<number>();for(const [i,p] of trace.prompts.entries()){
  if(p.outcome!=='completed')continue;const firstEvent:any=trace.events[p.startIndex]!.event;if(scope.test(firstEvent.prompt??''))continue;
  const substantive=new Set<string>();for(const j of p.toolStartIndices){const e:any=trace.events[j]!.event;const a=args.get(e.toolCallId)??'';if(!/wait|sleep|inbox|poll|status|watch/i.test(e.toolName??'')&&!/\b(?:sleep|inbox|poll|watch)\b/i.test(a))substantive.add(hash((e.toolName??'')+a))}
  if(substantive.size<2)continue;const ms=activeSegments(trace.events,p).reduce((a,s)=>a+s.ms,0);if(ms<=5000)continue;duration=Math.max(duration,ms);work.add(i);
 }
 if(!work.size){reject('no-completed-substantive-work');continue}register(m);pool.push({text,id:trace.id,family:m.family,rank:hash(m.h.id),duration,work});
}
// Limit each project family to four actors, duration-stratified, so one orchestration cannot dominate.
const bands=[0,1,2,3].map(()=>[] as Candidate[]);const band=(v:number)=>v<180000?0:v<900000?1:v<1800000?2:3;for(const c of pool)bands[band(c.duration)]!.push(c);for(const b of bands)b.sort((a,b)=>a.rank.localeCompare(b.rank));
const available=bands.map(x=>x.length),selected:Candidate[]=[],families=new Map<string,number>();while(selected.length<300&&bands.some(b=>b.length))for(let j=3;j>=0;j--){const c=bands[j]!.shift();if(!c)continue;const n=families.get(c.family)??0;if(n>=4)continue;families.set(c.family,n+1);selected.push(c)}
const model=JSON.parse(readFileSync(join(root,'control0.json'),'utf8'));const dir=join(root,'operational-sessions');mkdirSync(dir,{recursive:true,mode:0o700});const rows:string[]=[];const counts=[0,0,0,0];let promptCount=0;const familyIds=new Map<string,number>();
for(const [i,c] of selected.entries()){const path=join(dir,`${i}.jsonl`);writeFileSync(path,c.text,{mode:0o600,flag:'wx'});if(!familyIds.has(c.family))familyIds.set(c.family,familyIds.size);const sid=familyIds.get(c.family)!;const e:Entry={id:c.id,path,digest:hash(c.text),group:'operational'};const seen=new Set<number>();for(const row of rowsFor(e,sid,model)){if(!c.work.has(row.pid))continue;seen.add(row.pid);row.pid=i*100000+row.pid;rows.push(JSON.stringify(row))}promptCount+=seen.size;counts[band(c.duration)]!++}
writeFileSync(join(root,'operational.jsonl'),rows.join('\n')+'\n',{mode:0o600,flag:'wx'});const stats={discovered:found.size,available,selectedBands:counts,families:families.size,sessions:selected.length,prompts:promptCount,rows:rows.length,rejected};writeFileSync(join(root,'operational-stats.json'),JSON.stringify(stats),{mode:0o600,flag:'wx'});console.log(JSON.stringify(stats));
