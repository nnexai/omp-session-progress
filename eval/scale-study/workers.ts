// Private-only auxiliary worker extraction. Output never includes transcript metadata.
import { createHash } from 'node:crypto';
import { readdirSync, readFileSync, statSync, writeFileSync, mkdirSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { activeSegments, extractSessionText, pseudonym } from '../extract';
import { frozenText } from '../frozen-session';
import { privateDir } from '../horizon-study/data';
import { rowsFor, type Split } from './rows';
import type { Entry } from '../tool-timing/cohort';
import type { EnsembleModel } from '../../src/gbdt';

const root=process.argv[2]; if(!root) throw Error('Expected private study root'); privateDir(root);
const cohort:Split=JSON.parse(readFileSync(join(root,'cohort.json'),'utf8'));
const hash=(s:string|Buffer)=>createHash('sha256').update(s).digest('hex');
const scope=/omp-session-progress|omp-progress-|smoke|fixture|experiment|ratchet|benchmark/i;
type Meta={header:any;ids:Set<string>;prompts:Set<string>;rows:any[]};
function meta(text:string):Meta|null {
  let rows:any[]; try { rows=text.split('\n').filter(Boolean).map(JSON.parse); } catch { return null; }
  const header=rows.find(r=>r.type==='session');
  if(!header||header.version!==3||typeof header.id!=='string'||typeof header.cwd!=='string')return null;
  const ids=new Set<string>(),prompts=new Set<string>();
  for(const r of rows){
    if(r.type!=='session'&&typeof r.id==='string')ids.add(r.id);
    if(r.type!=='message'||r.message?.role!=='user')continue;
    const c=r.message.content; const t=(typeof c==='string'?c:Array.isArray(c)?c.filter((b:any)=>b.type==='text').map((b:any)=>b.text??'').join('\n'):'').replace(/\s+/g,' ').trim().toLowerCase();
    if(t.length)prompts.add(hash(t));
  }
  return {header,ids,prompts,rows};
}
const occupiedIds=new Set<string>(), occupiedPrompts=new Set<string>(), occupiedHeaders=new Set<string>();
function register(m:Meta){occupiedHeaders.add(m.header.id);for(const x of m.ids)occupiedIds.add(x);for(const x of m.prompts)occupiedPrompts.add(x);}
const parents=new Map<string,{fold:number;entry:Entry;text:string}[]>();
for(const [fold,entries] of cohort.dev.entries())for(const entry of entries){const text=frozenText(entry),m=meta(text);if(!m)throw Error('Invalid manifest');const a=parents.get(m.header.id)??[];a.push({fold,entry,text});parents.set(m.header.id,a);}
for(const entry of [...cohort.dev.flat(),...cohort.legacy.flat(),...cohort.official,...cohort.enriched,...cohort.fresh,...cohort.aux]){const m=meta(frozenText(entry));if(!m)throw Error('Invalid manifest');register(m);}
const reject:Record<string,number>={};const rejected=(k:string)=>{reject[k]=(reject[k]??0)+1;};
const source=join(homedir(),'.omp/agent/sessions');
type Candidate={text:string;id:string;fold:number;parent:string;duration:number;rank:string};const candidates:Candidate[]=[];
for(const d of readdirSync(source,{withFileTypes:true}).sort((a,b)=>a.name.localeCompare(b.name))){if(!d.isDirectory())continue;const base=join(source,d.name);
 for(const sub of readdirSync(base,{withFileTypes:true}).sort((a,b)=>a.name.localeCompare(b.name))){if(!sub.isDirectory())continue;const dir=join(base,sub.name);let parentText:string;try{parentText=readFileSync(dir+'.jsonl','utf8');}catch{continue;}
  const pm=meta(parentText);if(!pm)continue;const options=parents.get(pm.header.id)??[];
  // The frozen main must be a byte-identical prefix and uniquely assigned to a dev fold.
  const matches=options.filter(p=>parentText.startsWith(p.text));if(matches.length!==1)continue;const fold=matches[0]!.fold;
  for(const file of readdirSync(dir).sort()){if(!file.endsWith('.jsonl'))continue;const path=join(dir,file);let text:string;try{const s=statSync(path);if(s.size<200||s.size>64*1024*1024||s.mtimeMs>Date.now()-86400000){rejected('size-or-recent');continue;}text=readFileSync(path,'utf8');}catch{rejected('unreadable');continue;}
   const m=meta(text);if(!m){rejected('format');continue;}const h=m.header;
   const title=m.rows.find(r=>r.type==='title')?.title??h.title??'';
   if(h.parentSession||h.parentSessionId||h.parentId||scope.test(h.cwd+' '+title)){rejected('fork-or-scope');continue;}
   // A branch in the message tree means this is not a standalone linear trace.
   const children=new Set<string>();let branched=false;for(const r of m.rows){if(typeof r.parentId==='string'){if(children.has(r.parentId))branched=true;children.add(r.parentId);}}
   if(branched){rejected('branched');continue;}
   if(occupiedHeaders.has(h.id)||[...m.ids].some(x=>occupiedIds.has(x))||[...m.prompts].some(x=>occupiedPrompts.has(x))){rejected('overlap');continue;}
   const trace=extractSessionText(text,path);let duration=0;for(const p of trace.prompts)if(p.outcome==='completed')duration=Math.max(duration,activeSegments(trace.events,p).reduce((a,s)=>a+s.ms,0));
   if(duration<=5000){rejected('no-completed-reading');continue;}
   register(m);candidates.push({text,id:trace.id,fold,parent:matches[0]!.entry.id,duration,rank:hash(h.id)});
  }
 }
}
// Balanced round-robin across duration bands, deterministic within each band.
const bands=[candidates.filter(c=>c.duration<180000),candidates.filter(c=>c.duration>=180000&&c.duration<900000),candidates.filter(c=>c.duration>=900000&&c.duration<1800000),candidates.filter(c=>c.duration>=1800000)];
for(const b of bands)b.sort((a,b)=>a.rank.localeCompare(b.rank));const available=bands.map(b=>b.length), selected:Candidate[]=[];
while(selected.length<300&&bands.some(b=>b.length))for(const band of bands){const c=band.shift();if(c&&selected.length<300)selected.push(c);}
const models:EnsembleModel[]=[0,1,2].map(i=>JSON.parse(readFileSync(join(root,`control${i}.json`),'utf8')));
const directory=join(root,'workers-sessions');mkdirSync(directory,{mode:0o700,recursive:true});const rows=[[],[],[]] as string[][];const selectedBands=[0,0,0,0],folds=[0,0,0];
const parentNumbers=new Map<string,number>();
for(const [i,c] of selected.entries()){
 const path=join(directory,`${i}.jsonl`);writeFileSync(path,c.text,{mode:0o600,flag:'wx'});
 const parentKey=`${c.fold}:${c.parent}`;if(!parentNumbers.has(parentKey))parentNumbers.set(parentKey,parentNumbers.size);const sid=parentNumbers.get(parentKey)!;
 const entry:Entry={id:c.id,path,digest:hash(c.text),group:'worker'};for(const r of rowsFor(entry,sid,models[c.fold]!)){r.pid=i*100000+r.pid;rows[c.fold]!.push(JSON.stringify(r));}
 folds[c.fold]!++;selectedBands[c.duration<180000?0:c.duration<900000?1:c.duration<1800000?2:3]!++;
}
for(let i=0;i<3;i++)writeFileSync(join(root,`workers${i}.jsonl`),rows[i]!.join('\n')+'\n',{mode:0o600,flag:'wx'});
const stats={available,selectedBands,folds,rows:rows.map(r=>r.length),rejected:reject};writeFileSync(join(root,'workers-stats.json'),JSON.stringify(stats),{mode:0o600,flag:'wx'});console.log(JSON.stringify(stats));
