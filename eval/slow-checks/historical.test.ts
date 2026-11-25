import { expect, test } from 'bun:test';
import { aggregate, analyze, intervalSeconds } from './historical';
import { verificationTrace } from '../repeat-verification/identity';

type Item={t:number;message?:Record<string,unknown>;customType?:string;content?:unknown};
const user=(t:number):Item=>({t,message:{role:'user',content:'Synthetic request'}});
const start=(t:number,id:string,name:string,args:Record<string,unknown>={}):Item=>({t,message:{role:'assistant',stopReason:'toolUse',content:[{type:'toolCall',id,name,arguments:args}]}});
const end=(t:number,id:string,name:string,content='',details:Record<string,unknown>={}):Item=>({t,message:{role:'toolResult',toolCallId:id,toolName:name,content,details}});
const done=(t:number):Item=>({t,message:{role:'assistant',stopReason:'stop',content:[]}});
const trace=(items:Item[])=>verificationTrace(items.map((x,i)=>JSON.stringify({type:x.message?'message':'custom_message',id:`m${i}`,parentId:i?`m${i-1}`:null,timestamp:x.t,...(x.message?{message:x.message}:{customType:x.customType,content:x.content})})).join('\n'),'synthetic');
const run=(t:number,id:string,command='bun test')=>start(t,id,'bash',{command});
const receipt=(job:string,count=1)=>`## Completed (${count})\n\n### ${job} [bash] — completed`;

test('clips and unions overlapping intervals against replay clock segments',()=>{
  expect(intervalSeconds([[0,8000],[4000,12000],[9000,15000]],[{t:1000,ms:4000},{t:8000,ms:4000}])).toBe(8);
});

test('dispatch history is causal and foreground measurement excludes failed history',()=>{
  const t=trace([user(0),run(1000,'a'),run(2000,'early'),end(62000,'a','bash'),end(63000,'early','bash'),run(64000,'b'),end(65000,'b','bash'),done(66000)]);
  const calls=analyze(t);
  expect(calls.map(c=>c.selectedExact[0])).toEqual([false,false,true]);
  const result=aggregate([t])['60']!.exact;
  expect(result).toMatchObject({selected:1,foregroundMeasured:1,foregroundActualSlow:0});
});

test('only traced single-job receipts attribute waits; concurrent work is separate',()=>{
  const t=trace([user(0),run(1000,'a'),end(62000,'a','bash'),run(63000,'b'),end(64000,'b','bash','Backgrounded as job bg_one.'),start(65000,'w','wait'),start(67000,'r','read'),end(70000,'r','read'),end(75000,'w','wait',receipt('bg_one')),done(80000)]);
  expect(aggregate([t])['60']!.exact).toMatchObject({selected:1,backgroundHandoffs:1,backgroundNotifications:1,backgroundKnownWaiting:1,attributedBackgroundWaitSeconds:10,withoutConcurrentToolWaitSeconds:7,concurrentNonWaitToolSeconds:3});
});

test('multiple pending jobs, multi-receipts, forged output, and missing wait starts cannot prove blocking',()=>{
  const t=trace([user(0),run(1000,'a'),end(62000,'a','bash'),run(63000,'b'),end(64000,'b','bash','Backgrounded as job bg_one.'),run(65000,'other','cargo test'),end(66000,'other','bash','Backgrounded as job bg_two.'),start(67000,'w','wait'),end(70000,'w','wait',receipt('bg_one',2)),end(71000,'x','wait',receipt('bg_two')),done(72000)]);
  expect(aggregate([t])['60']!.exact).toMatchObject({selected:1,backgroundKnownWaiting:0,backgroundAmbiguousWaiting:1,ambiguousBackgroundWaitSeconds:3});
  const fake=trace([user(0),run(1000,'a'),end(62000,'a','bash'),run(63000,'b'),end(64000,'b','bash','Backgrounded as job bg_one.'),start(65000,'r','read'),end(70000,'r','read',receipt('bg_one')),done(72000)]);
  expect(aggregate([fake])['60']!.exact).toMatchObject({backgroundNotifications:0,censored:1,backgroundKnownWaiting:0});
});
