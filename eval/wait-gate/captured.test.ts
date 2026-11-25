import { describe, expect, test } from 'bun:test';
import { aligned } from './captured';

const call={start:100,end:120,kind:'eval',text:'join',async:false};
const join=(c:typeof call)=>!c.async && c.text==='join';

describe('private span to native call alignment',()=>{
  test('accepts a uniquely enclosed span with bounded margins',()=>{
    expect(aligned([call],100.1,119.9,join)).toBe(true);
    expect(aligned([call],102,118,join)).toBe(true);
  });
  test('rejects gaps, mismatched clocks, async calls and ambiguity',()=>{
    expect(aligned([call],102.01,118,join)).toBe(false);
    expect(aligned([call],99.8,120,join)).toBe(false);
    expect(aligned([call],1100,1120,join)).toBe(false);
    expect(aligned([{...call,async:true}],100.1,119.9,join)).toBe(false);
    expect(aligned([call,{...call}],100.1,119.9,join)).toBe(false);
  });
});
