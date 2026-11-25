import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import type { Entry } from './tool-timing/cohort';

export const sha=(input:string|Buffer)=>createHash('sha256').update(input).digest('hex');

/** Read exactly the selected bytes, even if the live source was later appended. */
export function frozenText(e:Entry):string {
  const bytes=readFileSync(e.path);
  if(sha(bytes)===e.digest)return bytes.toString('utf8');
  const hash=createHash('sha256');
  for(let start=0,end:number;(end=bytes.indexOf(10,start))!==-1;start=end+1){
    hash.update(bytes.subarray(start,end+1));
    if(hash.copy().digest('hex')===e.digest)return bytes.toString('utf8',0,end+1);
  }
  throw Error('No byte-identical frozen session prefix');
}
