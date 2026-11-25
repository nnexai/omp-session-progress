import { activeSegments, extractSession, GAP_CAP_MS, type PromptLabel, type Trace, type TraceEvent } from '../extract';
import { activeMs, DELEGATE_TOOLS, EXEC_TOOLS, EXPLORE_TOOLS, WRITE_TOOLS } from '../../src/facts';
import { evidence } from '../../src/hybrid/runtime/evidence';
import { Observer, type Clock } from '../../src/observer';
import type { Point, Prompt, Session, Span, Truth } from './types';

/** Only these coarse categories, never tool arguments or paths, leave the replay. */
export function toolFamily(name: string, intent: string): string {
  if (EXEC_TOOLS.has(name)) {
    if (/\b(end.to.end|e2e|full suite|all tests|emulator|webdriver|playwright)\b/i.test(intent)) return 'execute-e2e';
    if (/\b(verify|verif\w*|test\w*|check\w*|lint|smoke)\b/i.test(intent)) return 'execute-verification';
    return 'execute';
  }
  if (EXPLORE_TOOLS.has(name)) return 'inspect';
  if (WRITE_TOOLS.has(name)) return 'edit';
  if (DELEGATE_TOOLS.has(name)) return 'delegate';
  if (name === 'todo') return 'plan';
  if (name === 'ask') return 'ask';
  return 'other';
}

// Identical scheduling and saturation shortcut to eval/run.ts --horizons.
export class ReplayClock implements Clock {
  time = 0;
  private timer: { at: number; fn: () => void } | null = null;
  now() { return this.time; }
  setTimeout(fn: () => void, ms: number) {
    const timer = { at: this.time + ms, fn };
    this.timer = timer;
    return timer;
  }
  clearTimeout(handle: unknown) {
    if (this.timer === handle) this.timer = null;
  }
  advance(to: number, observer: Observer) {
    let stable = 0;
    let previous: number | null | undefined;
    while (this.timer && this.timer.at <= to) {
      const timer = this.timer;
      this.timer = null;
      this.time = timer.at;
      timer.fn();
      const { facts, reading } = observer.report();
      const stationary = facts.asking.size > 0 || (facts.lastAt !== null && this.time >= facts.lastAt + GAP_CAP_MS);
      const remaining = reading?.estimate.remainingMinutes;
      stable = stationary && remaining === previous ? stable + 1 : 0;
      previous = remaining;
      const next = this.timer as { at: number; fn: () => void } | null;
      if (stable >= 100 && next) {
        const ticks = Math.max(0, Math.floor((to - next.at) / 1_000));
        next.at += ticks * 1_000;
      }
    }
    this.time = to;
  }
}

type Call = { id: number; family: string; startS: number };
type Interval = Call & { endS: number; completed: boolean };

// This is deliberately separate from replay: these intervals are labels only.
// A duplicate start censors the overwritten call; an unmatched end is ignored.
function intervals(events: TraceEvent[], prompt: PromptLabel): { spans: Span[]; byId: Map<number, Interval> } {
  const segments = activeSegments(events, prompt);
  const offsets = new Map<number, number>();
  let elapsed = 0;
  for (const segment of segments) {
    offsets.set(segment.index, elapsed / 1000);
    elapsed += segment.ms;
  }
  offsets.set(prompt.endIndex, elapsed / 1000);
  const open = new Map<string, Call>();
  const byId = new Map<number, Interval>();
  const spans: Span[] = [];
  let nextId = 0;
  const close = (call: Call, endS: number, completed: boolean) => {
    const interval = { ...call, endS, completed };
    byId.set(call.id, interval);
    spans.push({ family: call.family, durationS: Math.max(0, endS - call.startS), completed });
  };
  for (let i = prompt.startIndex; i <= prompt.endIndex; i++) {
    const event = events[i]!.event;
    const at = offsets.get(i)!;
    if (event.type === 'tool_execution_start') {
      const key = String(event.toolCallId);
      const name = typeof event.toolName === 'string' ? event.toolName : 'tool';
      const old = open.get(key);
      if (old) { close(old, at, false); open.delete(key); }
      // Count all starts, including asks, for deterministic per-prompt IDs.
      const id = nextId++;
      if (name !== 'ask') open.set(key, { id, family: toolFamily(name, typeof event.intent === 'string' ? event.intent : ''), startS: at });
    } else if (event.type === 'tool_execution_end') {
      const key = String(event.toolCallId);
      const call = open.get(key);
      if (call) { close(call, at, true); open.delete(key); }
    }
  }
  for (const call of open.values()) close(call, elapsed / 1000, false);
  return { spans, byId };
}

/** Replay a session once, using only the prefix available at each checkpoint. */
export function collectSession(path: string): Session {
  return collectTrace(extractSession(path));
}

/** Replay an already captured in-memory snapshot without reopening the file. */
export function collectTrace(trace: Trace): Session {
  const clock = new ReplayClock();
  const observer = new Observer(clock);
  observer.enable('replay', () => {});
  let cursor = 0;
  let nextId = 0;
  const open = new Map<string, Call>();
  const advance = (at: number, endIndex: number) => {
    while (cursor <= endIndex && cursor < trace.events.length && trace.events[cursor]!.t <= at) {
      const { t, event } = trace.events[cursor++]!;
      clock.advance(t, observer);
      // Measure the tool start using exactly the production active-time clock.
      const startS = activeMs(observer.report().facts, t) / 1000;
      if (event.type === 'before_agent_start') { open.clear(); nextId = 0; }
      else if (event.type === 'tool_execution_start') {
        const key = String(event.toolCallId);
        const name = typeof event.toolName === 'string' ? event.toolName : 'tool';
        const id = nextId++;
        open.delete(key);
        if (name !== 'ask') open.set(key, { id, family: toolFamily(name, typeof event.intent === 'string' ? event.intent : ''), startS });
      } else if (event.type === 'tool_execution_end') open.delete(String(event.toolCallId));
      else if (event.type === 'agent_end') open.clear();
      observer.observe('replay', event);
    }
    clock.advance(at, observer);
  };
  const prompts: Prompt[] = [];
  const spans: Span[] = [];
  for (const prompt of trace.prompts) {
    const labels = intervals(trace.events, prompt);
    spans.push(...labels.spans);
    if (prompt.outcome !== 'completed') continue;
    const segments = activeSegments(trace.events, prompt);
    const totalMs = segments.reduce((sum, segment) => sum + segment.ms, 0);
    const points: Point[] = [];
    const truth: Truth[] = [];
    let elapsed = 0, segment = 0, missingPoints = 0;
    for (let target = 5_000; target < totalMs; target = target === 5_000 ? 15_000 : target + 15_000) {
      while (segment < segments.length && elapsed + segments[segment]!.ms < target) elapsed += segments[segment++]!.ms;
      const part = segments[segment]!;
      const at = part.t + target - elapsed;
      advance(at, prompt.endIndex);
      const { facts, reading } = observer.report();
      if (!facts.running || !reading || reading.estimate.remainingMinutes == null || !Number.isFinite(reading.estimate.remainingMinutes) || reading.estimate.remainingMinutes < 0) {
        missingPoints++;
        continue;
      }
      const nowS = activeMs(facts, at) / 1000;
      let selected: Call | null = null;
      for (const call of open.values()) {
        if (!selected || call.startS < selected.startS || (call.startS === selected.startS && call.id < selected.id)) selected = call;
      }
      const tool = selected ? { id: selected.id, family: selected.family, ageS: Math.max(0, nowS - selected.startS) } : null;
      points.push({ atMs: at, elapsedS: target / 1000, evidence: evidence(facts, reading, at), tool });
      const interval = selected ? labels.byId.get(selected.id) : undefined;
      truth.push({ remainingS: (totalMs - target) / 1000, toolLeftS: interval?.completed ? Math.max(0, interval.endS - target / 1000) : null });
    }
    prompts.push({ points, truth, missingPoints });
  }
  observer.disable();
  return { id: trace.id, spans, prompts };
}
