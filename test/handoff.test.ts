import {expect,test} from 'bun:test';
import {applyEvent,emptyFacts} from '../src/facts';
import {handoffMinutes} from '../src/handoff';

function completedPlan(){
  const f=emptyFacts();applyEvent(f,{type:'before_agent_start',prompt:'Implement and verify the change'},0);
  const todo=(args:object,t:number)=>{applyEvent(f,{type:'tool_execution_start',toolName:'todo',toolCallId:'t',args},t);applyEvent(f,{type:'tool_execution_end',toolName:'todo',toolCallId:'t',isError:false},t);};
  todo({op:'init',items:['Implement','Verify']},1000);todo({op:'done',task:'Implement'},130000);todo({op:'done',task:'Verify'},240000);return f;
}
test('a confirmed handoff reduces an implausible ETA but expires if the answer takes longer',()=>{
  const f=completedPlan(),early=handoffMinutes(f,240,8,3,16)!,later=handoffMinutes(f,300,8,3,16)!;
  expect(early[0]).toBeLessThan(1);expect(early[0]).toBeGreaterThan(0);expect(early[1]).toBeLessThanOrEqual(early[0]);expect(early[2]).toBeGreaterThanOrEqual(early[0]);
  expect(later[0]).toBeGreaterThan(early[0]);expect(later[0]).toBeLessThan(8);expect(handoffMinutes(f,331,8,3,16)).toBeUndefined();
});
test('new work cancels handoff even when a check starts and finishes in the same clock tick',()=>{
  const f=completedPlan();applyEvent(f,{type:'tool_execution_start',toolName:'bash',toolCallId:'check'},240000);
  expect(handoffMinutes(f,240,8,3,16)).toBeUndefined();applyEvent(f,{type:'tool_execution_end',toolName:'bash',toolCallId:'check',isError:false},240000);
  expect(handoffMinutes(f,240,8,3,16)).toBeUndefined();
});
