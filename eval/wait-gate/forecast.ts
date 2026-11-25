/** Seconds. The baseline already includes foreground work. A known blocking check
 * can impose a floor, never replace the rest of the prompt or be added twice. */
export function forecast(localS:number, priorS:number|null, ageS:number, waiting:boolean) {
  if (!Number.isFinite(localS) || localS<0 || !Number.isFinite(ageS) || ageS<0 ||
      (priorS!==null && (!Number.isFinite(priorS) || priorS<0))) throw Error('Invalid forecast input');
  const left=priorS===null?null:Math.max(0,priorS-ageS);
  const ungated=left===null?localS:Math.max(localS,left);
  return {local:localS,ungated,gated:waiting?ungated:localS,left};
}
