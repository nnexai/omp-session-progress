import { createHash } from 'node:crypto';
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, readSync, readdirSync, statSync, symlinkSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { activeSegments, extractSession } from '../extract';
import { EVAL_DIR, SESSIONS_DIR } from '../paths';

export interface Entry { id: string; path: string; digest: string; group: string }
export interface Cohort { train: Entry[]; test: Entry[] }
const digest = (bytes: Buffer|string) => createHash('sha256').update(bytes).digest('hex');
const exclusions = ['omp-session-progress','omp-progress-smoke','omp-progress-bench','omp-progress-eval','omp-hybrid-eval'];

// Outcome-based strata enrich rare long tasks, never select by prediction error.
// Session boundaries are also the split and the uncertainty-resampling unit.
export function prepare(root:string):Cohort {
  const file=join(root,'cohort.json');
  if(existsSync(file)) throw Error('Use a fresh private directory; never redefine an existing split');
  const oldIndex = new Set(Object.keys(JSON.parse(readFileSync(join(EVAL_DIR,'index.json'),'utf8'))));
  const oldTest = new Set<string>((JSON.parse(readFileSync('.omp/ratchet/progress-time/_state.json','utf8')).test_ids as string[]).map(id=>id.replace(/-p\d+$/,'')));
  const all:Entry[]=[]; const seen=new Set<string>(); const header=Buffer.alloc(4096);
  const eligibleBefore=Date.now()-3600_000;
  for(const dir of readdirSync(SESSIONS_DIR).sort()) {
    const sub=join(SESSIONS_DIR,dir); if(!statSync(sub).isDirectory())continue;
    for(const file of readdirSync(sub).sort()) {
      if(!file.endsWith('.jsonl')) continue;
      const path=join(sub,file),st=statSync(path);
      if(st.size<200||st.size>64*1024*1024||st.mtimeMs>=eligibleBefore)continue;
      const fd=openSync(path,'r'); let head:string;
      try {head=header.toString('utf8',0,readSync(fd,header));} finally {closeSync(fd);}
      if(exclusions.some(k=>head.includes(k)))continue;
      const bytes=digest(readFileSync(path)); const trace=extractSession(path);
      if(oldTest.has(trace.id)||seen.has(trace.id))continue;
      let maxMs=0,slow=false;
      for(const prompt of trace.prompts) {
        if(prompt.outcome!=='completed')continue;
        const opened=new Map<string,number>();let elapsed=0;
        for(const s of activeSegments(trace.events,prompt)) {
          const ev=trace.events[s.index]!.event, key=String(ev.toolCallId);
          if(ev.type==='tool_execution_start'&&ev.toolName!=='ask')opened.set(key,elapsed);
          if(ev.type==='tool_execution_end')opened.delete(key);
          elapsed+=s.ms;
          if(!slow)for(const start of opened.values())if(elapsed-start>60_000){slow=true;break;}
        }
        maxMs=Math.max(maxMs,elapsed);
      }
      if(digest(readFileSync(path))!==bytes)throw Error('Session changed during selection');
      seen.add(trace.id);
      const band=maxMs>=1800_000?'30+':maxMs>=900_000?'15-30':maxMs>=180_000?'3-15':maxMs>5000?'short':'no-scored-prompt';
      all.push({id:trace.id,path,digest:bytes,group:`${band}/${slow?'slow':'other'}`});
    }
  }
  const test:Entry[]=[];
  // Fixed quota within each duration × slow-tool stratum; prior benchmark
  // sessions cannot enter test, and at least half of each stratum stays in fit.
  for(const group of [...new Set(all.map(e=>e.group))].sort()) {
    if(group.startsWith('no-scored-prompt')) continue;
    const pool=all.filter(e=>e.group===group&&!oldIndex.has(e.id)).sort((a,b)=>digest('tool-age-v1:'+a.id).localeCompare(digest('tool-age-v1:'+b.id)));
    const cap=group.startsWith('30+')?12:group.startsWith('short')?5:8;
    test.push(...pool.slice(0,Math.min(cap,Math.floor(pool.length/2))));
  }
  const ids=new Set(test.map(x=>x.id)),train=all.filter(s=>!ids.has(s.id));
  if(test.length<12||train.length<30||test.some(s=>oldIndex.has(s.id)||oldTest.has(s.id)))throw Error('Insufficient disjoint sessions');
  const home=join(root,'fit-home','.omp','agent','sessions','fit');mkdirSync(home,{recursive:true,mode:0o700});
  for(const e of train)symlinkSync(e.path,join(home,`${e.id}.jsonl`));
  writeFileSync(file,JSON.stringify({train,test}),{mode:0o600});
  return {train,test};
}
export function verifyEntry(e:Entry) {if(digest(readFileSync(e.path))!==e.digest)throw Error('Historical session changed; replay invalid');}
if(import.meta.main) {
  const root=process.argv[2]; if(!root || (statSync(root).mode & 0o077)!==0)throw Error('Expected private scratch directory');
  const {train,test}=prepare(root); console.log(JSON.stringify({developmentSessions:train.length,holdoutSessions:test.length,holdoutStrata:test.reduce((t,e)=>(t[e.group]=(t[e.group]??0)+1,t),{} as Record<string,number>)}));
}
