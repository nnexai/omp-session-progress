import {formatRemaining} from '../../src/widget';
import type {Observation} from './score';
const loss=(lo:number,hi:number,y:number)=>hi-lo+4*Math.max(0,lo-y,y-hi);
export function intervals(rows:Observation[]){
  const groups=new Map<string,Observation[]>();for(const r of rows){const k=`${r.sid}:${r.pid}`;let g=groups.get(k);if(!g){g=[];groups.set(k,g);}g.push(r);}
  const band=(accept:(d:number)=>boolean)=>{
    let prompts=0,n=0,sc=0,width=0,cov=0,visible=0,visiblePrompts=0,printedScore=0,printedHits=0;
    for(const g of groups.values()){
      if(!accept(g[0]!.elapsed+g[0]!.y))continue;
      let k=0,s=0,w=0,h=0,printed=0,ps=0,ph=0;
      for(const r of g){if(r.v===null||r.lo==null||r.hi==null)continue;
        if(r.lo>r.v+1e-8||r.hi+1e-8<r.v)throw Error('Unordered interval');
        k++;s+=loss(r.lo,r.hi,r.y);w+=r.hi-r.lo;h+=Number(r.lo<=r.y&&r.y<=r.hi);
        const text=formatRemaining(r.v/60,r.lo/60,r.hi/60,false),match=/^~(\d+)–(\d+)m/.exec(text);
        if(match){const l=Number(match[1])*60,u=Number(match[2])*60;printed++;ps+=loss(l,u,r.y);ph+=Number(l<=r.y&&r.y<=u);}
      }
      if(!k)continue;prompts++;n+=k;sc+=s/k;cov+=h/k;width+=w/k;visible+=printed/k;
      if(printed){visiblePrompts++;printedScore+=ps/printed;printedHits+=ph/printed;}
    }
    return {prompts,points:n,scoreMin:sc/prompts/60,widthMin:width/prompts/60,coverage:cov/prompts,printedRate:visible/prompts,printedPrompts:visiblePrompts,printedScoreMin:visiblePrompts?printedScore/visiblePrompts/60:null,printedCoverage:visiblePrompts?printedHits/visiblePrompts:null};
  };
  return {all:band(()=>true),short:band(t=>t<60),medium:band(t=>t>=180&&t<900),long:band(t=>t>=900),halfHour:band(t=>t>=1800)};
}
export function intervalPass(b:ReturnType<typeof intervals>,c:ReturnType<typeof intervals>){
  const objective=(s:typeof b)=>0.35*s.medium.scoreMin+0.45*s.long.scoreMin+0.20*s.halfHour.scoreMin;
  return objective(c)<objective(b)&&c.medium.scoreMin<=b.medium.scoreMin*1.05&&c.long.scoreMin<=b.long.scoreMin*1.05&&c.all.coverage>=Math.min(0.45,b.all.coverage)&&c.short.scoreMin===b.short.scoreMin;
}
