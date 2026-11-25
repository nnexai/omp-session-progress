import { expect,test } from 'bun:test';
import { label,type TraceEvent } from '../extract';
import { replayPlans } from './ingress';
const request=(t:number):TraceEvent=>({t,event:{type:'before_agent_start',prompt:'Please check.'}});
const message=(t:number,text:string,role='assistant'):TraceEvent=>({t,event:{type:'message_end',message:{role,stopReason:'toolUse',content:[{type:'text',text}]}}});
const end=(t:number):TraceEvent=>({t,event:{type:'agent_end',stopReason:'stop'}});
const todo=(t:number,isError:boolean):TraceEvent=>({t,event:{type:'tool_execution_end',toolName:'todo',toolCallId:'one',isError,details:{op:'init',phases:[{name:'Work',tasks:[{content:'Verify then implement',status:'in_progress'}]}]}}});
const rows=(events:TraceEvent[])=>replayPlans({id:'synthetic',project:'synthetic',events,prompts:label(events)}).prompts;
test('readings see only already published prospective assistant text, not the next prompt or reasoning',()=>{
  const hidden:TraceEvent={t:7000,event:{type:'message_end',message:{role:'assistant',stopReason:'toolUse',content:[{type:'thinking',thinking:"I'll deploy it."}]}}};
  const ps=rows([request(0),message(900,"I'll launch then deploy.",'user'),hidden,message(16000,"I'll verify then edit."),end(35000),request(40000),end(50000)]);
  expect(ps[0]?.map(p=>p.prose?.steps??null)).toEqual([null,null,['verify','edit']]);expect(ps[1]?.[0]?.prose).toBeNull();
});
test('a slow tool cannot keep a stale spoken intention alive indefinitely',()=>{
  const pts=rows([request(0),message(1000,"I'll verify then edit."),end(340000)])[0]!;
  expect(pts.find(p=>p.elapsedS===300)?.prose?.tail).toBe('more');expect(pts.find(p=>p.elapsedS===315)?.prose).toBeNull();
});
test('requires a successful current-prompt checklist and does not carry it across requests',()=>{
  const ps=rows([request(0),todo(2000,true),todo(10000,false),end(20000),request(25000),end(31000)]);
  expect(ps[0]?.[0]?.checklist).toBeNull();expect(ps[0]?.[1]?.checklist?.steps).toEqual(['verify','edit']);expect(ps[1]?.[0]?.checklist).toBeNull();
});
