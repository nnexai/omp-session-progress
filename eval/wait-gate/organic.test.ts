import { expect, test } from 'bun:test';
import { organicReplay, type Frame } from './organic';
import { forecast } from './forecast';
import { verificationTrace } from '../repeat-verification/identity';

type Item={t:number;message:Record<string,unknown>};
const user=(t:number):Item=>({t,message:{role:'user',content:'Synthetic request'}});
const start=(t:number,id:string,name:string,args:Record<string,unknown>={}):Item=>({t,message:{role:'assistant',stopReason:'toolUse',content:[{type:'toolCall',id,name,arguments:args}]}});
const end=(t:number,id:string,name:string,content=''):Item=>({t,message:{role:'toolResult',toolCallId:id,toolName:name,content}});
const done=(t:number):Item=>({t,message:{role:'assistant',stopReason:'stop',content:[]}});
const run=(t:number,id:string,command='bun test')=>start(t,id,'bash',{command});
const receipt=(job:string)=>`## Completed (1)\n\n### ${job} [bash] — completed`;
const replay=(items:Item[])=>organicReplay(verificationTrace(items.map((x,i)=>JSON.stringify({type:'message',id:`m${i}`,parentId:i?`m${i-1}`:null,timestamp:x.t,message:x.message})).join('\n'),'synthetic'))[0]!;
const initial=[user(0),run(1000,'a'),end(62000,'a','bash'),run(63000,'b'),end(64000,'b','bash','Backgrounded as job bg_one.')];
const frame=(rows:Frame[],second:number)=>{const result=rows.find(r=>r.elapsedS===second);if(!result)throw Error('Missing synthetic checkpoint');return result;};
test('untargeted wait is optimistic only and receipt cannot supply earlier strict proof',()=>{
  const without=replay([...initial,start(65000,'w','wait'),end(90000,'w','wait'),done(100000)]);
  const withReceipt=replay([...initial,start(65000,'w','wait'),end(90000,'w','wait',receipt('bg_one')),done(100000)]);
  const before=frame(without,75),after=frame(withReceipt,75);
  expect(after).toEqual(before);
  expect(after).toMatchObject({wait:true,identifiable:true,strict:false,prior:61,age:12});
  expect(forecast(10,after.prior,after.age,after.strict)).toMatchObject({ungated:49,gated:10});
  expect(forecast(10,after.prior,after.age,after.identifiable).gated).toBe(49);
  expect(frame(withReceipt,90)).toMatchObject({wait:false,identifiable:false,prior:null});
});

test('competing jobs, other tool activity, and task launches invalidate wait until a new wait',()=>{
  const other=replay([...initial,start(65000,'w','wait'),start(76000,'r','read'),end(80000,'r','read'),end(100000,'w','wait',receipt('bg_one')),done(110000)]);
  expect(frame(other,75).identifiable).toBe(true);
  expect(frame(other,90)).toMatchObject({wait:true,identifiable:false,strict:false});
  const competing=replay([...initial,run(65000,'c','cargo test'),end(66000,'c','bash','Backgrounded as job bg_two.'),start(67000,'w','wait'),end(100000,'w','wait',receipt('bg_one')),done(110000)]);
  expect(frame(competing,75)).toMatchObject({wait:true,identifiable:false,strict:false});
  const worker=replay([...initial,start(65000,'task','task'),end(66000,'task','task'),start(67000,'w','wait'),end(100000,'w','wait',receipt('bg_one')),done(110000)]);
  expect(frame(worker,75).identifiable).toBe(false);
});

test('foreground execution alone never enables the wait gate',()=>{
  const rows=replay([user(0),run(1000,'a'),end(62000,'a','bash'),run(63000,'b'),end(100000,'b','bash'),done(110000)]);
  const f=frame(rows,75);
  expect(f).toMatchObject({wait:false,identifiable:false,strict:false,prior:61,age:12});
  expect(forecast(10,f.prior,f.age,f.identifiable)).toMatchObject({ungated:49,gated:10});
});

test('asking and nested waits interrupt the optimistic attribution',()=>{
  const asking=replay([...initial,start(65000,'w','wait'),start(76000,'q','ask'),end(80000,'q','ask'),end(110000,'w','wait',receipt('bg_one')),done(120000)]);
  // The ask interval is excluded from the frozen active clock.
  expect(frame(asking,75).identifiable).toBe(true);
  expect(frame(asking,90).identifiable).toBe(false);
  const nested=replay([...initial,start(65000,'w','wait'),start(76000,'v','wait'),end(80000,'v','wait'),end(110000,'w','wait',receipt('bg_one')),done(120000)]);
  expect(frame(nested,90).identifiable).toBe(false);
});
