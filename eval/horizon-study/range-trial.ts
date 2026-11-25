import {join} from 'node:path';import {writeFileSync,readFileSync} from 'node:fs';
import {privateDir,type Row} from './data';import {fit} from './ranges';import {calibratedRange as adjust} from '../../src/range-calibration';import {intervalPass,intervals} from './interval-score';import type {Observation} from './score';
const root=process.argv[2]!,devRoot=process.argv[3]??root;privateDir(root);privateDir(devRoot);
const read=(dir:string,n:string)=>JSON.parse(readFileSync(join(dir,n),'utf8')) as Row[];
const dev=[0,1,2].map(f=>read(devRoot,`dev${f}.json`));const base:Observation[]=[],cand:Observation[]=[];
for(const [f,test] of dev.entries()){
  const [j,k]=[0,1,2].filter(i=>i!==f),m=fit([...read(root,`inner${j}-${k}.json`),...read(root,`inner${k}-${j}.json`)]);
  for(const r of test){const o={sid:r.sid,pid:r.pid,elapsed:r.elapsed,y:r.y,slow:r.slow,v:r.b,lo:r.low,hi:r.upper};base.push(o);const [lo,hi]=adjust(m,r.e,r.b,r.low,r.upper);cand.push({...o,lo,hi});}
}
const baseline=intervals(base),candidate=intervals(cand);const selected=intervalPass(baseline,candidate);
writeFileSync(join(root,'range-selection.json'),JSON.stringify({baseline,candidate,selected},null,2),{mode:0o600,flag:'wx'});
writeFileSync(join(root,'range-model.json'),JSON.stringify(fit(dev.flat())),{mode:0o600,flag:'wx'});
console.log({selected});for(const k of Object.keys(baseline) as (keyof typeof baseline)[])console.log(k,baseline[k].scoreMin,candidate[k].scoreMin,baseline[k].coverage,candidate[k].coverage);
