// The router sees the observed age and the existing ETA, never the final duration.
export interface HorizonPolicy {head:number; weight:number; onset:number; useForecast:boolean}
export function routedSeconds(base:number, predicted:number, age:number, p:HorizonPolicy):number {
  const evidence=age+(p.useForecast?Math.min(base,600)*0.5:0);
  const weight=p.weight*(p.onset===0?1:Math.min(1,Math.max(0,(evidence-p.onset)/p.onset)));
  if(!weight)return base;
  // Limit a single model's correction on unfamiliar work. Blend on the same
  // log-seconds scale as display smoothing, including its 30 s stabilizer.
  const ratio=Math.max(0.20,Math.min(5,(predicted+30)/(base+30)));
  return Math.max(0,(base+30)*ratio**weight-30);
}
