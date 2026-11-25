import { expect,test } from 'bun:test';
import { summarize } from './metrics';
import type { ScoredPoint } from './types';
function reading(call:number,local:number,pooled:number,remaining:number):ScoredPoint {
  return {local,exact:pooled,pooled,remaining,slow:false,timing:{call,kind:'test',ageS:10,background:false,exact:[70],family:[70],priorPromptExact:0,priorPromptFamily:0},tool:{population:local,exact:pooled,pooled,gold:remaining,used:'exact'}};
}
test('overlapping long horizons keep one vote per prompt; tool errors count distinct session-scoped calls',()=>{
  const a=reading(1,3,1,1),b=reading(1,1,3,1),c=reading(2,1,5,1);
  const s=summarize([{session:'a',duration:1800,points:[a,a,a,c]}, {session:'b',duration:900,points:[b]}, {session:'b',duration:180,points:[]}]);
  expect(s.noReadings).toBe(1);expect(s.slices.long.mae.local).toBe(.75);expect(s.slices['30+'].prompts).toBe(1);
  expect(s.slices.long.mae.pooled).toBe(1.5);expect(s.tool.usedExact.calls).toBe(3);
  expect(s.tool.usedExact.callWeighted.mae.pooled).toBe(2);expect(s.tool.usedExact.callWeighted.naive.mae).toBe(0);
  expect(s.balancedMediumLong.mae.pooled).toBeNull();expect(s.slices['30+'].contrasts.pooledVsLocal.ci95).toBeNull();
});
