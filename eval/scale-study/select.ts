import {readFileSync,writeFileSync} from 'node:fs';
import {createInterface} from 'node:readline';import {createReadStream} from 'node:fs';import {join} from 'node:path';
import {forestSeconds,type ScaleForest} from './scale-forest';import {routedSeconds,type HorizonPolicy} from './scale-policy';
import {priority,score,eligible} from '../horizon-study/score';import {privateDir} from '../horizon-study/data';import type {ScaleRow} from './rows';
const root=process.argv[2]!;privateDir(root);
const rows:(Omit<ScaleRow,'x'>&{heads:number[]})[]=[];
for(let f=0;f<3;f++){
  const model:ScaleForest=JSON.parse(readFileSync(join(root,`forest${f}.json`),'utf8'));
  for await(const line of createInterface({input:createReadStream(join(root,`dev${f}.jsonl`))})){
    const {x,...rest}=JSON.parse(line) as ScaleRow;rows.push({...rest,heads:forestSeconds(model,x,rest.e)});
  }
}
const evaluate=(p:HorizonPolicy)=>score(rows.map(r=>({...r,v:routedSeconds(r.b,r.heads[p.head]!,r.e,p)})));
const ctrl=evaluate({head:0,weight:0,onset:0,useForecast:false});
const candidates:{p:HorizonPolicy;s:ReturnType<typeof score>;pass:boolean;delta:number}[]=[];
for(let head=0;head<rows[0]!.heads.length;head++)for(const weight of [.35,.65,1])for(const onset of [0,60,180,450])for(const useForecast of (onset?[false,true]:[false])){
  const p={head,weight,onset,useForecast},s=evaluate(p);candidates.push({p,s,pass:eligible(ctrl,s),delta:priority(s)-priority(ctrl)});
}
candidates.sort((a,b)=>a.delta-b.delta);const best=candidates.find(x=>x.pass);if(!best)throw Error('No eligible model');
writeFileSync(join(root,'selection.json'),JSON.stringify({ctrl,best,candidates}),{mode:0o600,flag:'wx'});console.log('development winner',best.p,'delta',best.delta);for(const k of ['short','medium','long','halfHour','all'] as const)console.log(k,ctrl[k].prompts,ctrl[k].maeMin,best.s[k].maeMin);
