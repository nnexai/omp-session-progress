// Cross-harness prior only: input is fixed-vocabulary scope and prompt length,
// not prompt text. Model values are locally trained numeric coefficients.
export interface TransferPriorModel { coefficients:number[]; residuals:number[] }

/** Conditional median remaining seconds, given survival to active elapsed time. */
export function transferPrior(model:TransferPriorModel, scope:number, logChars:number, elapsed:number):number {
  const c=model.coefficients;
  if(c.length!==26||!model.residuals.length)return 0;
  let z=c[0]!+c[1]!*Math.min(Math.log1p(100000),Math.max(0,logChars));
  for(let i=0;i<24;i++)z+=c[i+2]!*((scope>>>i)&1);
  const threshold=Math.log1p(Math.max(0,elapsed))-z;
  let lo=0,hi=model.residuals.length;
  while(lo<hi){const m=(lo+hi)>>>1;if(model.residuals[m]!<=threshold)lo=m+1;else hi=m;}
  // Above the last training residual, a zero estimate would claim completion.
  // Extrapolate a conservative residual tail instead.
  const residual=lo===model.residuals.length?threshold+0.5:model.residuals[lo+Math.floor((model.residuals.length-lo-1)/2)]!;
  return Math.max(0,Math.expm1(Math.min(16,z+residual))-elapsed);
}
