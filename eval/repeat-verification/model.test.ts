import { expect,test } from 'bun:test';
import { fit, forecast, predict } from './model';
import type { Span, TimingPrefix } from './types';
const span=(s:number,anchor:number|null,completed=true):Span=>({kind:'test',seconds:s,completed,background:false,noticeOnly:false,exactAnchorS:anchor,familyAnchorS:anchor});
const history=(ageS:number,exact:number[],family=exact):TimingPrefix=>({call:2,kind:'test',ageS,background:false,exact,family,priorPromptExact:0,priorPromptFamily:0});

test('compares the current age with *scaled* development reruns, even after exceeding a previous duration',()=>{
  const m=fit(Array.from({length:4},()=>[span(10,9),span(10,9),span(30,9),span(30,9)]));
  const t=forecast(m,'test',11,9,null)!;
  expect(t.population).toBeCloseTo(19/60); expect(t.exact).toBeCloseTo(19/60);
  const slower=forecast(m,'test',11,19,null)!;expect(slower.exact).toBeCloseTo(9/60);
  const eta=predict(m,5,history(11,[19]));expect(eta.exact).toBeCloseTo(5-10/60);
  expect(predict(m,5,{...history(11,[19]),background:true}).exact).toBe(5);
});
test('no history, unidentified tail or inadequate independent support never shortens the prompt',()=>{
  const runs=Array.from({length:4},()=>[span(15,null),span(20,null),span(60,40,false)]);
  const m=fit(runs);expect(predict(m,3,history(1,[])).pooled).toBe(3);
  expect(predict(m,3,history(25,[20])).pooled).toBe(3);
  expect(forecast(fit([runs.flat()]),'test',1,10,10)).toBeNull();
});
