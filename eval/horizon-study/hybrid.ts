// Dev follow-up: turn budgets worked better on compact workflows and calibrated
// residuals on longer ones. Route continuously on the model's predicted span,
// never on an offline duration band. Freeze this single blend before holdout.
import {join} from 'node:path';
import {readFileSync,writeFileSync} from 'node:fs';
import {privateDir,type Row,type Input} from './data';
import * as turns from './turns';
import * as calibration from './calibration';
import {eligible,input,priority,score,type Observation} from './score';
export type Model={turn:ReturnType<typeof turns.fit>;table:number[]};
export function predict(m:Model,r:Input):number{
  const w=Math.max(0,Math.min(1,(Math.log(Math.max(r.e+r.b,120))-Math.log(120))/Math.log(5)));
  if(w===0)return turns.predict(m.turn,r);
  const long=calibration.predict(m.table,r.e,r.b);
  if(w===1)return long;
  return Math.expm1((1-w)*Math.log1p(turns.predict(m.turn,r))+w*Math.log1p(long));
}
if(import.meta.main){
  const root=process.argv[2]!;privateDir(root);const read=(n:string)=>JSON.parse(readFileSync(join(root,n),'utf8')) as Row[];
  const dev=[0,1,2].map(k=>read(`dev${k}.json`)), base:Observation[]=[],hybrid:Observation[]=[];
  const byFold=dev.map((test,i)=>{
    const [j,k]=[0,1,2].filter(n=>n!==i),m={turn:turns.fit(dev.flatMap((g,f)=>f===i?[]:g)),table:calibration.fit([...read(`inner${j}-${k}.json`),...read(`inner${k}-${j}.json`)])},at=base.length;
    for(const r of test){const p={sid:r.sid,pid:r.pid,y:r.y,elapsed:r.elapsed,slow:r.slow};base.push({...p,v:r.b});hybrid.push({...p,v:predict(m,input(r))});}
    return [priority(score(base.slice(at))),priority(score(hybrid.slice(at)))];
  });
  const b=score(base),h=score(hybrid);const selected=eligible(b,h)&&priority(b)-priority(h)>=0.25;
  writeFileSync(join(root,'hybrid-selection.json'),JSON.stringify({selected,byFold,baseline:b,hybrid:h},null,2),{flag:'wx',mode:0o600});
  console.log({selected,byFold});for(const [k,s] of Object.entries(h))console.log(k,b[k as keyof typeof b].maeMin,s.maeMin);
}
