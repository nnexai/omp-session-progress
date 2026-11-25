import {createHash} from 'node:crypto';import {readFileSync,writeFileSync} from 'node:fs';import {join} from 'node:path';import {pseudonym} from '../extract';import {privateDir} from '../horizon-study/data';
const [source,extra]=process.argv.slice(2);privateDir(source!);privateDir(extra!);const n=JSON.parse(readFileSync(join(source!,'operational-stats.json'),'utf8')).sessions;
const entries=Array.from({length:n},(_,i)=>{
  const path=join(source!,'operational-sessions',i+'.jsonl'),bytes=readFileSync(path);
  let start=0;while(start<bytes.length){const nl=bytes.indexOf(10,start),end=nl<0?bytes.length:nl;
    const line=bytes.subarray(start,end).toString().trim();start=end+1;if(!line)continue;
    const head=JSON.parse(line);if(head.type==='session'&&typeof head.id==='string')return {path,id:pseudonym('s',head.id),digest:createHash('sha256').update(bytes).digest('hex'),group:'managed'};
  }throw Error('No managed session header');
});
const pids=new Set<number>();for(const line of readFileSync(join(source!,'operational.jsonl'),'utf8').trim().split('\n'))pids.add(JSON.parse(line).pid);for(const [name,v] of [['entries',entries],['pids',[...pids]]])writeFileSync(join(extra!,`managed-${name}.json`),JSON.stringify(v),{mode:0o600,flag:'wx'});
