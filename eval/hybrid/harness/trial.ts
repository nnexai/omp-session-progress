import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { activeSegments, extractSession, GAP_CAP_MS, type Trace } from '../../extract';
import { Observer, type Clock } from '../../../src/observer';
import { evidence } from '../../../src/hybrid/runtime/evidence';
import { HybridForecast, type Ask } from '../../../src/hybrid/runtime/forecast';
import { mean, observe, score, slice, type Slice } from './grade';

interface Case { id:string; tag:string; short:number; commitment:string }
interface Mapping { path:string; sha:string; primary:number; short:number[] }
const hash = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');

// Matches the production-Observer horizons replay: after the display has
// settled in a stationary gap, skip only identical aligned timer ticks.
class ReplayClock implements Clock {
 time = 0;
 private timer: { at:number; fn:() => void } | null = null;
 now() { return this.time; }
 setTimeout(fn:() => void, ms:number) { const timer = { at:this.time + ms, fn }; this.timer = timer; return timer; }
 clearTimeout(handle:unknown) { if (this.timer === handle) this.timer = null; }
 advance(to:number, observer:Observer) {
  let stable = 0, previous:number | null | undefined;
  while (this.timer && this.timer.at <= to) {
   const timer = this.timer; this.timer = null; this.time = timer.at; timer.fn();
   const { facts, reading } = observer.report();
   const stationary = facts.asking.size > 0 || (facts.lastAt !== null && this.time >= facts.lastAt + GAP_CAP_MS);
   const remaining = reading?.estimate.remainingMinutes;
   stable = stationary && remaining === previous ? stable + 1 : 0; previous = remaining;
   const next = this.timer as { at:number; fn:() => void } | null;
   if (stable >= 100 && next) next.at += Math.max(0, Math.floor((to - next.at) / 1_000)) * 1_000;
  }
  this.time = to;
 }
}

export async function trial(c:Case, ask:Ask, mode:'policy'|'pilot', signal:AbortSignal,
 onTrainSample?:(packet:string, diagnosis:string)=>void):Promise<{grade:Record<string,number>,summary:string}> {
 // Resolve only a declared, committed case. Never report private paths or data.
 let row:Mapping, trace:Trace;
 try {
  const manifest:Case[] = JSON.parse(readFileSync(join(import.meta.dir, '..', 'manifest.json'), 'utf8'));
  const declared = manifest.find(x => x.id === c.id);
  if (!declared || declared.tag !== c.tag || declared.short !== c.short || declared.commitment !== c.commitment) throw Error();
  const map:Record<string,Mapping> = JSON.parse(readFileSync(join(homedir(), '.cache', 'omp-session-progress-eval', 'hybrid-v1', 'map.json'), 'utf8'));
  row = map[c.id]!;
  if (!row || !Array.isArray(row.short) || row.short.length !== c.short || new Set([row.primary, ...row.short]).size !== 1 + row.short.length ||
      hash(JSON.stringify([row.sha, row.primary, row.short])) !== c.commitment || hash(readFileSync(row.path)) !== row.sha) throw Error();
  trace = extractSession(row.path);
  if (hash(readFileSync(row.path)) !== row.sha) throw Error();
 } catch { throw new Error('Case provenance verification failed'); }
 const selected = [row.primary, ...row.short].sort((a,b) => a-b);
 if (selected.some(i => !Number.isInteger(i) || !trace.prompts[i] || trace.prompts[i]!.outcome !== 'completed')) throw new Error('Invalid selected prompt');
 const clock = new ReplayClock(), observer = new Observer(clock);
 observer.enable('replay', () => {});
 let cursor = 0;
 const advance = (at:number) => {
  while (cursor < trace.events.length && trace.events[cursor]!.t <= at) {
   const event = trace.events[cursor++]!;
   clock.advance(event.t, observer); observer.observe('replay', event.event);
  }
  clock.advance(at, observer);
 };
 const primary = slice(), slow = slice(), shorts:Slice[] = [];
 let long30 = false;
 let preview: { elapsed_s:number; expected_min:number; local_min:number; hybrid_min:number } | undefined;
 try {
  for (const index of selected) {
   const prompt = trace.prompts[index]!;
   const segments = activeSegments(trace.events, prompt);
   const totalMs = segments.reduce((n,s) => n+s.ms, 0);
   const isPrimary = index === row.primary;
   if (isPrimary && totalMs < 180_000 || !isPrimary && !(totalMs > 5_000 && totalMs < 60_000)) throw new Error('Invalid selected prompt duration');
   if (isPrimary) long30 = totalMs >= 1_800_000;
   const result = isPrimary ? primary : slice();
   if (!isPrimary) shorts.push(result);
   const forecast = new HybridForecast(mode);
   let segment = 0, elapsed = 0;
   for (let target = 5_000; target < totalMs; target = target === 5_000 ? 15_000 : target + 15_000) {
    if (signal.aborted) throw new Error('Trial aborted');
    while (segment < segments.length && elapsed + segments[segment]!.ms < target) elapsed += segments[segment++]!.ms;
    const current = segments[segment];
    if (!current) throw new Error('Invalid active segments');
    const at = current.t + target - elapsed;
    advance(at);
    const { facts, reading } = observer.report();
    if (!reading) throw new Error('Missing displayed ETA');
    const e = evidence(facts, reading, at);
    const prediction = await forecast.sample(e, at, ask, signal);
    const truth = (totalMs - target) / 60_000;
    observe(result, prediction, e.localMin, truth);
    if (isPrimary && e.slowTool) observe(slow, prediction, e.localMin, truth);
    if (isPrimary && mode === 'pilot' && !preview) preview = {elapsed_s: target/1000, expected_min: Math.round(truth), local_min: Math.round(e.localMin), hybrid_min: Math.round(prediction)};
    if (isPrimary && onTrainSample && [5_000,60_000,300_000,900_000].includes(target)) {
     onTrainSample(JSON.stringify(e), JSON.stringify({local_min:Math.round(e.localMin), hybrid_min:Math.round(prediction), true_min:Math.round(truth)}));
    }
   }
  }
 } finally { observer.disable(); }
 const grade = score(primary, shorts, slow, long30);
 // Zero short exposure is fixed at zero, with the actual denominator exposed.
 const summary = JSON.stringify({ primary: { checkpoints:primary.count, mae_min:mean(primary,'error'), baseline_mae_min:mean(primary,'baselineError'), major_miss:mean(primary,'major'), baseline_major_miss:mean(primary,'baselineMajor') }, short: { prompts:shorts.length, checkpoints:grade.short_checkpoints, catastrophe:grade.short_catastrophe, no_exposure_value:0 }, slow: { checkpoints:slow.count, mae_min:mean(slow,'error'), baseline_mae_min:mean(slow,'baselineError') }, long30: { present:long30, checkpoints:grade.long30_checkpoints, mae_min:grade.long30_mae_min }, ...(preview ? { preview } : {}) });
 return { grade, summary };
}
