import type { Reading } from '../../estimator';
import { activeMs, DELEGATE_TOOLS, EXEC_TOOLS, EXPLORE_TOOLS, WRITE_TOOLS, type Facts } from '../../facts';

// The only text-derived signals allowed to cross the provider boundary are keys in
// this literal vocabulary. Never serialize a goal, todo, intent, tool name or path.
const signals = {
  investigation: /\b(investigat\w*|research|analy[sz]\w*|diagnos\w*|debug\w*)\b/i,
  implementation: /\b(implement\w*|refactor\w*|migrat\w*|redesign\w*|integrat\w*)\b/i,
  verification: /\b(verify|verif\w*|test\w*|check\w*|lint|smoke)\b/i,
  endToEnd: /\b(end.to.end|e2e|full suite|all tests|emulator|webdriver|playwright)\b/i,
  release: /\b(deploy\w*|install\w*|restart\w*|publish\w*|rollout|release)\b/i,
  review: /\b(review\w*|audit\w*|report\w*|compare|evaluat\w*)\b/i,
  deliver: /\b(commit\w*|push|summari[sz]\w*|document\w*)\b/i,
} as const;
const keys = Object.keys(signals) as Array<keyof typeof signals>;
function categories(texts: readonly string[]) { return keys.filter(k=>texts.some(t=>signals[k].test(t))); }
function family(name: string) {
  if (EXEC_TOOLS.has(name)) return 'execute';
  if (EXPLORE_TOOLS.has(name)) return 'inspect';
  if (WRITE_TOOLS.has(name)) return 'edit';
  if (DELEGATE_TOOLS.has(name)) return 'delegate';
  if (name==='todo') return 'plan';
  if (name==='ask') return 'ask';
  return 'other';
}
type Category = keyof typeof signals;
type Family = 'execute' | 'inspect' | 'edit' | 'delegate' | 'plan' | 'ask' | 'other';
export interface Evidence {
  elapsedS:number; localMin:number; phase:Reading['phase']; lowMin:number|null; highMin:number|null;
  request:Category[]; requestedItems:number; continuation:boolean; hasCurrentPlan:boolean;
  open:number; done:number; openKinds:Category[]; currentKinds:Category[];
  calls:number; errors:number; cleanExecAfterEdit:number; edits:number; delegates:number;
  recent:Family[]; running:{kind:Family;ageS:number;intentKinds:Category[]}[]; slowTool:boolean; wrapSignal:boolean;
}
export function evidence(f:Facts, r:Reading, now:number): Evidence {
  const localMin=r.estimate.remainingMinutes;
  if(localMin==null || !Number.isFinite(localMin) || localMin<0) throw Error('No displayed ETA');
  const elapsedS=activeMs(f,now)/1000;
  const underway=[...f.underway.values()];
  const open=f.todoUpdated?f.todos.filter(t=>t.status!=='done'&&t.status!=='dropped'):[];
  return {
    elapsedS,localMin,phase:r.phase,lowMin:r.estimate.lowMinutes,highMin:r.estimate.highMinutes,
    request:categories(f.goal?[f.goal]:[]),requestedItems:f.promptListItems,continuation:f.promptKwContinue,
    hasCurrentPlan:f.todoUpdated,open:open.length,done:f.todoDoneEventsS.length,
    openKinds:categories(open.map(t=>t.text)),currentKinds:categories(open.filter(t=>t.status==='in_progress').map(t=>t.text)),
    calls:f.toolCalls,errors:f.failedCalls,cleanExecAfterEdit:f.postWriteCleanExecCount,edits:f.catCounts.write,
    delegates:f.taskSubagentsLaunched,recent:f.toolHistory.slice(-4).map(t=>family(t.name)),
    running:underway.slice(0,4).map(t=>({kind:family(t.name),ageS:Math.max(0,Math.round(elapsedS-t.startedAtS)),intentKinds:categories([t.label])})),
    slowTool:underway.some(t=>t.name!=='ask'&&elapsedS-t.startedAtS>60),wrapSignal:f.agentWrapUp,
  };
}
