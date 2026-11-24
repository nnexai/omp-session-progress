// Build pointers, not transcripts. The grader alone opens the private session map.
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { activeSegments, extractSession } from '../extract';
import { EVAL_DIR } from '../paths';
const hash = (s: string | Buffer) => createHash('sha256').update(s).digest('hex');
const index: Record<string,string> = JSON.parse(readFileSync(join(EVAL_DIR,'index.json'),'utf8'));
const official = new Set<string>((JSON.parse(readFileSync('.omp/ratchet/progress-time/_state.json','utf8')).test_ids as string[]).map(s=>s.replace(/-p\d+$/,'')));
if (official.size !== 40) throw Error('Unexpected model holdout: check provenance');
const candidates = Object.entries(index).map(([session,path])=>{
 const t=extractSession(path); if(t.id!==session) throw Error('Index mismatch');
 const ps=t.prompts.flatMap((p,i)=>{if(p.outcome!=='completed')return []; const s=activeSegments(t.events,p).reduce((a,b)=>a+b.ms,0)/1000;return [{i,s}]}).sort((a,b)=>hash(`${session}:${a.i}`).localeCompare(hash(`${session}:${b.i}`)));
 return {session,path,sha:hash(readFileSync(path)),short:ps.filter(p=>p.s>5&&p.s<60).slice(0,2),medium:ps.find(p=>p.s>=180&&p.s<900),long:ps.find(p=>p.s>=900)};
}).sort((a,b)=>hash(a.session).localeCompare(hash(b.session)));
const pool=candidates.filter(c=>official.has(c.session));
const ordered=(k:'medium'|'long',other:'medium'|'long')=>pool.filter(c=>c[k]).sort((a,b)=>Number(!!a[other])-Number(!!b[other]));
const med=ordered('medium','long'),long=ordered('long','medium');
const used=new Set<string>(); const cases: {id:string,tag:string,short:number,commitment:string}[]=[];
const priv:Record<string,{path:string,sha:string,primary:number,short:number[]}>={};
function add(c:typeof pool[number],k:'medium'|'long',pilot=false){const id=(pilot?'pilot-':'case-')+hash(`hybrid-v1:${c.session}`).slice(0,10);const row={path:c.path,sha:c.sha,primary:c[k]!.i,short:c.short.map(p=>p.i)};priv[id]=row;cases.push({id,tag:pilot?'pilot':k,short:row.short.length,commitment:hash(JSON.stringify([row.sha,row.primary,row.short]))});used.add(c.session);}
for(;;){const l=long.find(c=>!used.has(c.session));const m=med.find(c=>!used.has(c.session)&&c!==l);if(!l||!m)break;add(l,'long');add(m,'medium')}
let pilots=0;for(const c of candidates){if(official.has(c.session))continue;const k=pilots%2?'medium':'long';if(!c[k])continue;add(c,k,true);if(++pilots===5)break;}
if(cases.length-pilots<15||pilots!==5)throw Error('Insufficient independent sessions');
mkdirSync(join(EVAL_DIR,'hybrid-v1'),{recursive:true,mode:0o700});writeFileSync(join(EVAL_DIR,'hybrid-v1','map.json'),JSON.stringify(priv),{mode:0o600});
writeFileSync(join(import.meta.dir,'manifest.json'),JSON.stringify(cases,null,2)+'\n');console.log(JSON.stringify({sessions:cases.length-pilots,pilots,shortCompanions:cases.filter(c=>c.tag!=='pilot').reduce((a,c)=>a+c.short,0)}));
