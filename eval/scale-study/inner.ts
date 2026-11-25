// Fit the inner control on exactly one fold, never on the outer validation fold.
import {readFileSync,writeFileSync} from 'node:fs';import {join} from 'node:path';import {refit,privateDir} from '../horizon-study/data';import {rowsFor,type Split} from './rows';
const [root,k]=process.argv.slice(2);privateDir(root!);const i=Number(k),c:Split=JSON.parse(readFileSync(join(root!,'cohort.json'),'utf8'));
if(![0,1,2].includes(i))throw Error('Invalid inner fold');const m=await refit(root!,c.dev[i]!,[...c.dev.flatMap((d,f)=>f===i?[]:d),...c.official,...c.enriched]);
for(let f=0;f<3;f++)if(f!==i){const pred:number[]=[];for(const [s,entry] of c.dev[f]!.entries())for(const r of rowsFor(entry,s*3+f,m))pred.push(r.b);writeFileSync(join(root!,`inner${i}-${f}.json`),JSON.stringify(pred),{mode:0o600,flag:'wx'});}
