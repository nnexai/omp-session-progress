import {createHash} from 'node:crypto';import {readFileSync,writeFileSync} from 'node:fs';import {join} from 'node:path';import {privateDir} from './data';import {priority} from './score';
const [root,out]=process.argv.slice(2);if(!root||!out)throw Error('Expected PRIVATE_ROOT NEW_RESULT_JSON');privateDir(root);
const read=(f:string)=>JSON.parse(readFileSync(join(root,f),'utf8'));
const dev=read('comparison.json'),cal=read('calibration-selection.json'),hy=read('hybrid-selection.json'),range=read('range-selection.json');
const points={...dev.summaries,calibration:cal.calibrated,hybrid:hy.hybrid};
const summary={baselineCommit:'1d2301b',rangeArtifactSha256:createHash('sha256').update(readFileSync(join(root,'range-model.json'))).digest('hex'),
  selection:'Development: group all checkpoints by prompt, folds by whole session, exclude official and enriched sessions. Priority = 0.35*medium + 0.45*15+ + 0.20*30+ MAE. Required point gain >=0.25 min plus medium/all/short guardrails. Only calibrated intervals shipped.',
  development:{sessions:244,point:Object.fromEntries(Object.entries(points).map(([k,s]:[string,any])=>[k,{priority:priority(s),maeMin:Object.fromEntries(Object.entries(s).map(([name,g]:[string,any])=>[name,g.maeMin]))}])),range},
  official:read('official-result.json'),enriched:read('enriched-result.json')};
writeFileSync(out,JSON.stringify(summary,null,2)+'\n',{flag:'wx',mode:0o600});
