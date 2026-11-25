import type { Facts } from './facts';

// Rounded P25/P75/P95 remaining seconds at the first confirmed closeout in
// 162 development prompts. A listed background job is not assumed finished.
const CAPS = [{low:10, point:25, high:75}, {low:10, point:30, high:60}];

export function handoffStage(f:Facts, elapsed:number):number {
  if (!f.running || !f.todoUpdated || !f.todoDoneEventsS.length || elapsed<120 ||
      f.underway.size || f.asking.size || f.errorStreak || f.retry) return -1;
  const last=f.todoDoneEventsS[f.todoDoneEventsS.length-1]!;
  if (elapsed-last>90 || f.toolCalls!==f.todoDoneToolsAt[f.todoDoneToolsAt.length-1] || f.lastErrorAtS>last) return -1;
  let done=false;
  for (const item of f.todos) {
    if (item.status==='done') done=true;
    else if (item.status!=='dropped') return -1;
  }
  return done ? Number(f.bgJobs.size>0) : -1;
}

// This caps an implausibly large remaining time; it never reports completion.
// After 30 s with no further evidence, fade back to the unmodified estimator.
export function handoffMinutes(f:Facts,elapsed:number,point:number,low:number,high:number):[number,number,number] | undefined {
  const idx=handoffStage(f,elapsed);
  if(idx<0)return;
  const caps=CAPS[idx]!;
  const since=elapsed-f.todoDoneEventsS[f.todoDoneEventsS.length-1]!;
  const weight=Math.min(1,(90-since)/60);
  const cap=(minutes:number,seconds:number)=>minutes-weight*Math.max(0,minutes-seconds/60);
  return [cap(point,caps.point),cap(low,caps.low),cap(high,caps.high)];
}
