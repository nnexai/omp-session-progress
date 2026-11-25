import {applyEvent, emptyFacts, type Facts, type ObservedEvent} from '../../src/facts';
type WorkFacts = Facts & {requestScope:number; acceptedScope:number; previousAnswerScope:number; outputAtFirstTurn:number};
export const researchFacts = ():WorkFacts => Object.assign(emptyFacts(),{requestScope:0,acceptedScope:0,previousAnswerScope:0,outputAtFirstTurn:0});
export function researchEvent(f:WorkFacts,ev:ObservedEvent,ts:number):void {
  if (!applyEvent(f,ev,ts)) return;
  if(ev.type==='before_agent_start' && typeof ev.prompt==='string') {
    const text=ev.prompt.trim();f.requestScope=scopeBits(text.slice(0,2500));
    f.acceptedScope=text.length<180 && /\b(continue|proceed|go ahead|yes|do it|looks good|lgtm|approved|next)\b/i.test(text)?f.previousAnswerScope:0;
    f.previousAnswerScope=0;f.outputAtFirstTurn=0;
  } else if(ev.type==='message_end') {
    const m=ev.message as {role?:string;stopReason?:string;content?:{type?:string;text?:string}[];usage?:{output?:number}}|undefined;
    if(m?.role!=='assistant')return;
    if(f.assistantTurns===1 && typeof m.usage?.output==='number')f.outputAtFirstTurn=m.usage.output;
    if(m.stopReason==='stop')f.previousAnswerScope=scopeBits((Array.isArray(m.content)?m.content.filter(b=>b.type==='text'&&typeof b.text==='string').map(b=>b.text).join(' '):'').slice(-1500));
  }
}

// Fixed, public vocabulary; a request becomes category bits, never project names,
// token hashes, commands or an embedding that must be sent to another process.
const categories = [
  /\b(implement|create|add|build|support|introduce)\b/i,
  /\b(refactor|restructure|simplify|split|extract|remove|replace)\b/i,
  /\b(debug|fix|broken|fails?|wrong|hangs?|regression|crash)\b/i,
  /\b(trace|investigate|research|analy[sz]e|diagnose|figure out|look into)\b/i,
  /\b(review|audit|inspect|evaluate|assess|compare)\b/i,
  /\b(test|verify|validate|smoke|check)\b/i,
  /\b(benchmark|performance|latency|profil[ei]|experiment|accuracy|faster)\b/i,
  /\b(integrat\w*|migrat\w*|port|cutover|end.to.end|across|full suite)\b/i,
  /\b(install|deploy|release|rollout|restart|rebuild|package)\b/i,
  /\b(commit|push|merge|rebase|cherry.pick|pull request)\b/i,
  /\b(render|visual|design|layout|dialog|ui|ux|widget|screen)\b/i,
  /\b(browser|emulator|hardware|device|native|desktop|android|phone)\b/i,
  /\b(delegate|subagents?|workers?|supervis\w*|parallel|orchestrat\w*)\b/i,
  /\b(explain|summari[sz]e|describe|suggest|why|what|which)\b/i,
  /\b(plan|proposal|strategy|approach|architecture|ideas?)\b/i,
  /\b(document\w*|readme|changelog|guide|note|write.up)\b/i,
  /\b(quick|small|only|just|single|one.line|brief|simple)\b/i,
  /\b(all|each|every|across|whole|entire|multiple|comprehensive)\b/i,
  /\b(until|iterate|again|keep going|keep at|do not stop|repeat)\b/i,
  /\b(then|afterwards|after that|as well as|also|once)\b/i,
  /\b(read.only|no (?:code )?changes?|without (?:editing|changing)|do not (?:edit|change|implement))\b/i,
  /\b(go ahead|continue|proceed|yes|agreed|do it|approved)\b/i,
  /\b(evidence|acceptance|reproduc\w*|baseline|receipt|criteria)\b/i,
  /\b(blocked|waiting|failed|timeout|error|unexpected|mismatch)\b/i,
];
export function scopeBits(text:string):number {
  let bits=0; const bounded=text.slice(0,3000);
  for(let i=0;i<categories.length;i++)if(categories[i]!.test(bounded))bits|=1<<i;
  return bits;
}
function since(times:number[],n:number,elapsed:number){
  return times.length>=n?Math.log1p(Math.max(0,elapsed-times[times.length-n]!)):-1;
}
// Appends to a fresh feature vector only; the legacy 140 feature indices stay fixed.
export function addWorkShape(x:number[],f:WorkFacts,e:number):void {
  for(let i=0;i<categories.length;i++)x.push((f.requestScope>>>i)&1);
  // For brief approvals, the deliverable is often in the preceding answer, not “yes”.
  for(const i of [0,2,4,5,7,8,9,12,14,18])x.push((f.acceptedScope>>>i)&1);
  const t=f.turnTimesS,u=f.toolTimesS;
  for(const k of [1,3,8,20])x.push(since(t,k,e),since(u,k,e));
  x.push(Math.log1p(f.firstWriteAtS??e),Math.log1p(f.todoInitAtS??e),
    Math.log1p(Math.max(0,e-f.lastActivityEndAtS)),Math.log1p(t[0]??e),
    Math.log1p(u[0]??e),Math.log1p(f.priorPromptDurationsS.at(-2)??0),
    Math.log1p(f.outputAtFirstTurn),Math.log1p(!t.length?0:f.outputAtFirstTurn/Math.max(1,t[0]!)));
  // Current route size in the agent's own confirmed plan, at the instant it was posted.
  let checks=0,edits=0,deploys=0,investigations=0;
  if(f.todoUpdated)for(const item of f.todos){if(item.status==='done'||item.status==='dropped')continue;
    if(/test|verif|validat|smoke|check/i.test(item.text))checks++;
    if(/implement|build|updat|refactor|chang|add /i.test(item.text))edits++;
    if(/deploy|install|release|commit|push/i.test(item.text))deploys++;
    if(/investigat|research|inspect|analy|review/i.test(item.text))investigations++;
  }
  x.push(checks,edits,deploys,investigations);
}
