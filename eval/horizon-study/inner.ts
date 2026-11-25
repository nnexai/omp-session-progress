// Inner cross-fitting: no base model used to learn a correction has seen its
// outer validation fold. Each fit on ONE fold scores the other two separately.
import { join } from 'node:path';
import { writeFileSync } from 'node:fs';
import { entries,privateDir,refit,rowsFor } from './data';
const [corpus,root,arg]=process.argv.slice(2);privateDir(root!);
const k=Number(arg);if(![0,1,2].includes(k))throw Error('Expected fit fold 0..2');
const groups=[0,1,2].map(i=>entries(corpus!,'fold'+i));
const model=await refit(root!,groups[k]!,groups.flatMap((g,i)=>i===k?[]:g));
for(const [j,group] of groups.entries())if(j!==k){
  const rows=group.flatMap((ent,i)=>rowsFor(ent,i*3+j,j,model).map(({sid,pid,e,b,elapsed,y})=>({sid,pid,e,b,elapsed,y})));
  writeFileSync(join(root!,`inner${k}-${j}.json`),JSON.stringify(rows),{flag:'wx',mode:0o600});
}
console.log(`Nested predictions ready for training fold ${k}`);
