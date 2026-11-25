import { readFileSync,writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { privateDir,type Row } from './data';
import { eligible,input,priority,score,type Observation } from './score';
import * as deadlines from './deadlines';
import * as turns from './turns';
import * as uncertainty from './uncertainty';
import * as lifetime from './lifetime';

const root=process.argv[2]!;privateDir(root);
const folds=[0,1,2].map(f=>JSON.parse(readFileSync(join(root,`dev${f}.json`),'utf8')) as Row[]);
const arms={baseline:[] as Observation[],deadlines:[] as Observation[],turns:[] as Observation[],uncertainty:[] as Observation[],lifetime:[] as Observation[]};
const byFold:Record<string,number>[]=[];
for(const [i,test] of folds.entries()){
  const train=folds.flatMap((rows,f)=>f===i?[]:rows), dm=deadlines.fit(train), tm=turns.fit(train), lm=lifetime.fit(train);
  const offset=arms.baseline.length;
  for(const r of test){const x=input(r), preds={baseline:x.b,deadlines:deadlines.predict(dm,x),turns:turns.predict(tm,x),uncertainty:uncertainty.predict(x),lifetime:lifetime.predict(lm,x)};
    for(const k of Object.keys(arms) as (keyof typeof arms)[])arms[k].push({sid:r.sid,pid:r.pid,y:r.y,elapsed:r.elapsed,slow:r.slow,v:preds[k]});}
  byFold.push(Object.fromEntries(Object.entries(arms).map(([k,rows])=>[k,priority(score(rows.slice(offset)))])));
}
const summaries=Object.fromEntries(Object.entries(arms).map(([k,points])=>[k,score(points)])), base=summaries.baseline!;
const ranked=Object.entries(summaries).filter(([,s])=>eligible(base,s)).sort((a,b)=>priority(a[1])-priority(b[1]));
const [name,s]=ranked[0]!;
const selected=priority(base)-priority(s)>=0.25?name:'baseline';
writeFileSync(join(root,'comparison.json'),JSON.stringify({selected,byFold,summaries},null,2),{flag:'wx',mode:0o600});
console.log('Priority minutes, then medium / long / 30+ / short / all, and guardrail pass:');
for(const [k,s] of Object.entries(summaries))console.log(k,[priority(s),s.medium.maeMin,s.long.maeMin,s.halfHour.maeMin,s.short.maeMin,s.all.maeMin].map(n=>n.toFixed(3)).join(' '),eligible(base,s));
console.log('Selected:',selected,'; fold priority scores:',byFold);
