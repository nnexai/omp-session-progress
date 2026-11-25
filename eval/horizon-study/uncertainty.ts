import type { Input } from './data';

// Fixed before comparison: act only once a prompt has actually survived a minute.
// Move partway from the center towards its existing upper estimate, not past it.
export function predict(r:Input):number {
  const ramp=(v:number)=>Math.max(0,Math.min(1,v));
  const w=0.4*ramp((r.e-60)/240)*ramp((r.e+r.b-300)/600);
  return w===0?r.b:Math.expm1(Math.log1p(r.b)+w*(Math.log1p(r.upper)-Math.log1p(r.b)));
}
