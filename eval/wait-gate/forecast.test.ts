import { expect,test } from 'bun:test';
import { forecast } from './forecast';

test('an outstanding ten-minute check cannot extend ETA while the agent is doing something else',()=>{
  expect(forecast(60,600,120,false)).toEqual({local:60,ungated:480,gated:60,left:480});
  expect(forecast(60,600,120,true).gated).toBe(480);
  expect(forecast(60,600,140,false).gated).toBe(60);
});
test('a targeted wait never adds time twice or discards post-check work',()=>{
  expect(forecast(540,600,120,true).gated).toBe(540);
  expect(forecast(45,60,90,true).gated).toBe(45);
  expect(forecast(45,null,90,true)).toMatchObject({gated:45,left:null});
});
