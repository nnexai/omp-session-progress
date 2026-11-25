// Reuse the exact cohort/weights from the previous study. Development ETA inputs
// must also be out of sample: each session gets a control trained on the other folds.
import { createHash } from 'node:crypto';
import { cpSync, existsSync, mkdirSync, readFileSync, statSync, symlinkSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { verifyEntry, type Cohort, type Entry } from '../tool-timing/cohort';
import { privateProvenance } from '../provenance';

const [priorArg, rootArg] = process.argv.slice(2);
if (!priorArg || !rootArg) throw Error('Expected frozen tool-timing directory and fresh private workspace');
const prior=resolve(priorArg), root=resolve(rootArg), project=resolve(import.meta.dir,'../..');
if (prior===root || (statSync(root).mode&0o077)!==0 || existsSync(join(root,'test'))) throw Error('Use a fresh private workspace');
const sha=(data:string|Buffer)=>createHash('sha256').update(data).digest('hex');
const setHash=(es:Entry[])=>sha(es.map(e=>e.digest).sort().join('\n'));
const data:Cohort=JSON.parse(readFileSync(join(prior,'cohort.json'),'utf8'));
const expected=privateProvenance();
const baseSrc=join(prior,'source','src'),weightFile=join(baseSrc,'model-data.ts');
if (setHash(data.train)!==expected.developmentContentSetSha256 || setHash(data.test)!==expected.holdoutContentSetSha256 || sha(readFileSync(weightFile))!==expected.refitWeightsSha256) throw Error('Cohort or control does not match the recorded comparison');
const all=[...data.train,...data.test];
if(new Set(all.map(e=>e.id)).size!==all.length || new Set(all.map(e=>e.digest)).size!==all.length)throw Error('Duplicate session/content across experiment');
for(const e of all)verifyEntry(e);
const groups=Map.groupBy(data.train,e=>e.group), folds:Entry[][]=[[],[],[]];
for(const group of groups.values()){
  group.sort((a,b)=>sha('milestones:'+a.id).localeCompare(sha('milestones:'+b.id)));
  group.forEach((e,i)=>folds[i%folds.length]!.push(e));
}
function setup(name:string,entries:Entry[]){
  const dir=join(root,name), state=join(dir,'.omp/ratchet/progress-time');
  mkdirSync(state,{recursive:true});cpSync(join(project,'.omp/ratchet/progress-time/_state.json'),join(state,'_state.json'));
  cpSync(baseSrc,join(dir,'src'),{recursive:true});cpSync(join(project,'eval'),join(dir,'eval'),{recursive:true});
  writeFileSync(join(dir,'entries.json'),JSON.stringify(entries),{mode:0o600});return dir;
}
setup('test',data.test);
await Promise.all(folds.map(async (entries,i)=>{
  const dir=setup(`fold${i}`,entries),excluded=new Set(entries.map(e=>e.id));
  const home=join(dir,'home'),sessions=join(home,'.omp/agent/sessions/train');mkdirSync(sessions,{recursive:true});
  for(const e of data.train)if(!excluded.has(e.id))symlinkSync(e.path,join(sessions,e.id+'.jsonl'));
  const proc=Bun.spawn([process.execPath,join(dir,'eval/train.ts')],{cwd:dir,env:{...process.env,HOME:home},stdout:'ignore',stderr:'inherit'});
  if(await proc.exited)throw Error(`Fold ${i} refit failed`);
}));
console.log(`Frozen ${data.train.length} development and ${data.test.length} test sessions; disjoint development folds: ${folds.map(f=>f.length).join('/')}.`);
