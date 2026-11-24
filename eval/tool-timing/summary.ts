import type { LunaAnswer } from './types';

export interface ScoredPoint {
  truth:number; local:number; timed:number; slow:boolean; eligible:boolean; treated:boolean;
  tool:number|null; post:number|null; goldTool:number|null; naive:number|null;
  ref:{index:number;delayMs:number}|null;
}
export interface ScoredPrompt { session:string; totalMin:number; missing:number; points:ScoredPoint[] }
type Arm='local'|'timed'|'hybrid';
const arms:Arm[]=['local','timed','hybrid'];
const sliceNames=['all','medium','long','30+','slow','short','lunaApplied'] as const;
type SliceName=typeof sliceNames[number];
type Errors=Record<Arm,number>;
const zeros=():Errors=>({local:0,timed:0,hybrid:0});
interface PromptScore {session:string; n:number; error:Errors; major:Errors; over15:Errors}
function average(xs:number[]) {return xs.length?xs.reduce((a,b)=>a+b,0)/xs.length:null;}
function means(rows:PromptScore[],key:'error'|'major'|'over15') {return Object.fromEntries(arms.map(a=>[a,average(rows.map(r=>r[key][a]/r.n))]));}
function clusters(rows:PromptScore[],a:Arm,b:Arm) {
  const result=new Map<string,[number,number]>();
  for(const r of rows){const pair=result.get(r.session)??[0,0];pair[0]+=(r.error[a]-r.error[b])/r.n;pair[1]++;result.set(r.session,pair);}
  return result;
}
// Average checkpoints within a prompt first. Resample entire sessions and
// preserve medium/long co-occurrence, not thousands of dependent clock ticks.
function paired(groups:PromptScore[][],sessions:string[],a:Arm,b:Arm) {
  if(groups.some(g=>!g.length))return null;
  const strata=groups.map(rs=>clusters(rs,a,b));
  const effect=average(groups.map(rs=>average(rs.map(r=>(r.error[a]-r.error[b])/r.n))!))!;
  const sims:number[]=[];let state=201193;
  for(let rep=0;rep<5000;rep++){
    const sums=strata.map(()=>[0,0]);
    for(let i=0;i<sessions.length;i++){
      state=(Math.imul(1664525,state)+1013904223)>>>0;
      const session=sessions[Math.floor(state/4294967296*sessions.length)]!;
      strata.forEach((s,k)=>{const pair=s.get(session);if(pair){sums[k]![0]!+=pair[0];sums[k]![1]!+=pair[1];}});
    }
    if(sums.every(x=>x[1]!>0))sims.push(average(sums.map(x=>x[0]!/x[1]!))!);
  }
  sims.sort((x,y)=>x-y);return {delta:effect,ci95:[sims[Math.floor(.025*sims.length)],sims[Math.floor(.975*sims.length)]]};
}
export function report(prompts:ScoredPrompt[],answers:LunaAnswer[]){
  const rows=Object.fromEntries(sliceNames.map(k=>[k,[]])) as unknown as Record<SliceName,PromptScore[]>;
  let missing=0,noCheckpoint=0,eligible=0,treated=0,applied=0,total=0;
  const componentMeans:{tool:number;naive:number;post:number}[]=[];
  for(const prompt of prompts){
    missing+=prompt.missing;if(!prompt.points.length){noCheckpoint++;continue;}
    const by=new Map<SliceName,PromptScore>();let n=0,te=0,ne=0,pe=0;
    for(const p of prompt.points){
      const ref=p.ref,answer=ref?answers[ref.index]:null;
      const live=!!(answer && answer.postMin!==null && ref && ref.delayMs>0 && ref.delayMs>=answer.latencyMs && p.tool!==null && p.post!==null);
      // Only the post-tool component is blended; never count the tool twice.
      const hybrid=live?p.tool!+.5*(p.post!+answer!.postMin!):p.timed;
      const pred={local:p.local,timed:p.timed,hybrid};
      total++;eligible+=Number(p.eligible);treated+=Number(p.treated);applied+=Number(live);
      if(p.treated&&p.naive!==null&&p.goldTool!==null&&p.post!==null&&p.tool!==null){
        n++;te+=Math.abs(p.tool-p.goldTool);ne+=Math.abs(p.naive-p.goldTool);pe+=Math.abs(p.post-(p.truth-p.goldTool));
      }
      const groups:SliceName[]=['all'];
      if(prompt.totalMin>=3&&prompt.totalMin<15)groups.push('medium');
      if(prompt.totalMin>=15)groups.push('long');
      if(prompt.totalMin>=30)groups.push('30+');
      if(prompt.totalMin<1)groups.push('short');
      if(p.slow)groups.push('slow');if(live)groups.push('lunaApplied');
      for(const g of groups){const r=by.get(g)??{session:prompt.session,n:0,error:zeros(),major:zeros(),over15:zeros()};r.n++;
        for(const a of arms){const err=Math.abs(pred[a]-p.truth);r.error[a]+=err;r.major[a]+=Number(err>=15);r.over15[a]+=Number(pred[a]>=15);}by.set(g,r);}
    }
    if(n)componentMeans.push({tool:te/n,naive:ne/n,post:pe/n});
    for(const [g,r]of by)rows[g].push(r);
  }
  const sessions=[...new Set(prompts.map(p=>p.session))];
  const slices=Object.fromEntries(sliceNames.map(k=>{const r=rows[k];return [k,{sessions:new Set(r.map(p=>p.session)).size,prompts:r.length,checkpoints:r.reduce((a,p)=>a+p.n,0),mae:means(r,'error'),major:means(r,'major'),over15:means(r,'over15'),timingVsLocal:paired([r],sessions,'timed','local'),lunaVsTiming:paired([r],sessions,'hybrid','timed')}];}));
  const balanced=Object.fromEntries(arms.map(a=>{
    const medium=slices.medium.mae[a],long=slices.long.mae[a];
    return [a,medium==null||long==null?null:(medium+long)/2];
  }));
  return {completedPrompts:prompts.length,noCheckpoint,missing,total,eligible,treated,applied,slices,balanced,
    timingVsLocal:paired([rows.medium,rows.long],sessions,'timed','local'),lunaVsTiming:paired([rows.medium,rows.long],sessions,'hybrid','timed'),hybridVsLocal:paired([rows.medium,rows.long],sessions,'hybrid','local'),
    component:{prompts:componentMeans.length,toolMae:average(componentMeans.map(v=>v.tool)),naiveToolMae:average(componentMeans.map(v=>v.naive)),postMae:average(componentMeans.map(v=>v.post))}};
}
