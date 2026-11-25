import {readFileSync,writeFileSync} from 'node:fs';
import {join} from 'node:path';
import {privateDir,type Row} from './data';
import {eligible,priority,score,type Observation} from './score';
import {fit,predict} from './calibration';
const root=process.argv[2]!;privateDir(root);
const read=(file:string)=>JSON.parse(readFileSync(join(root,file),'utf8')) as Row[];
const dev=[0,1,2].map(i=>read(`dev${i}.json`));
const base:Observation[]=[],candidate:Observation[]=[];
const byFold=dev.map((test,i)=>{
  const [a,b]=[0,1,2].filter(k=>k!==i);
  const table=fit([...read(`inner${a}-${b}.json`),...read(`inner${b}-${a}.json`)]), at=base.length;
  for(const r of test){const p={sid:r.sid,pid:r.pid,y:r.y,elapsed:r.elapsed,slow:r.slow};base.push({...p,v:r.b});candidate.push({...p,v:predict(table,r.e,r.b)});}
  return [priority(score(base.slice(at))),priority(score(candidate.slice(at)))];
});
const b=score(base),c=score(candidate), selected=eligible(b,c)&&priority(b)-priority(c)>=0.25;
writeFileSync(join(root,'calibration-selection.json'),JSON.stringify({selected,byFold,baseline:b,calibrated:c},null,2),{flag:'wx',mode:0o600});
writeFileSync(join(root,'calibration-model.json'),JSON.stringify(fit(dev.flat())),{flag:'wx',mode:0o600});
console.log({selected,byFold});for(const [k,s] of Object.entries(c))console.log(k,b[k as keyof typeof b].maeMin,s.maeMin);
