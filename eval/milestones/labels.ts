import { GAP_CAP_MS, type PromptLabel, type Trace, type TraceEvent } from '../extract';
import { applyEvent, emptyFacts, EXEC_TOOLS, WRITE_TOOLS } from '../../src/facts';
import type { Band, Stage } from './types';

type WorkStage = Exclude<Stage, 'none'>;
type PrefixState = { stage: Stage; episodesSeen: number; verificationsSeen: number; elapsedSinceStageS: number; verifying: boolean };
type Snapshot = {
  t: number; activeMs: number; asking: number; stage: Stage; stageAtMs: number;
  episodesSeen: number; verificationsSeen: number; verifying: boolean;
};

// Only explicit intent is examined for commands: paths and command bodies are
// deliberately unavailable. A task call is a dispatch only when it has tasks.
function classify(event: TraceEvent['event']): WorkStage | null {
  const name = typeof event.toolName === 'string' ? event.toolName : '';
  if (WRITE_TOOLS.has(name)) return 'edit';
  if (name === 'task') {
    const args = event.args;
    return args && typeof args === 'object' && Array.isArray((args as Record<string, unknown>).tasks)
      && ((args as Record<string, unknown>).tasks as unknown[]).length > 0 ? 'delegate' : null;
  }
  if (!EXEC_TOOLS.has(name)) return null;
  const intent = (typeof event.intent === 'string' ? event.intent : '')
    .replace(/\S*[/\\]\S*|\b[\w-]+\.[\w.-]+\b/g, ' ');
  if (/\b(?:deploy(?:ing|ed|ment)?|publish(?:ing|ed)?|releas(?:e|ing|ed)|push(?:ing|ed)?)\b/i.test(intent)) return 'release';
  if (/\b(?:end.to.end|e2e|full suite|all tests|emulator|webdriver|playwright|verify|verif\w*|test\w*|check\w*|lint|smoke)\b/i.test(intent)) return 'verify';
  return null;
}

const order: WorkStage[] = ['edit', 'verify', 'delegate', 'release'];
const bandOf = (count: number): Band => count === 0 ? 0 : count === 1 ? 1 : count < 4 ? 2 : 3;

/**
 * Offline proxy: distinct recognized stage starts form runs. Unrecognized work
 * does not interrupt a run. Simultaneous starts are a set, not an invented
 * sequence: a stage already in the preceding set is not a new run. For a
 * mixed set the displayed stage stays unchanged if possible, otherwise uses
 * the fixed order above. All events at a checkpoint are included.
 *
 * `prefix` excludes the offline band entirely. `at` adds the count of runs
 * beginning strictly later than the checkpoint. Elapsed time is active time:
 * open asks pause it and each event gap is capped. Optionally elapsedS is the
 * caller's prefix-only active clock, in the same units as Observer's clock.
 */
export function stageTimeline(events: TraceEvent[], prompt: PromptLabel): {
  prefix(atMs: number, elapsedS?: number): PrefixState;
  at(atMs: number, elapsedS?: number): PrefixState & { band: Band };
} {
  const snapshots: Snapshot[] = [];
  const runs: Array<{ t: number; count: number }> = [];
  const open = new Map<string, WorkStage | 'ask' | null>();
  let previous = new Set<WorkStage>();
  let stage: Stage = 'none', stageAtMs = 0, active = 0, episodes = 0, verifications = 0;
  let lastT = events[prompt.startIndex]?.t ?? 0;
  let asking = 0, verifying = 0;
  const stop = (id: string) => {
    const old = open.get(id);
    if (old === 'ask') asking--;
    if (old === 'verify') verifying--;
    open.delete(id);
  };
  for (let i = prompt.startIndex; i <= prompt.endIndex;) {
    const t = events[i]!.t;
    if (!asking) active += Math.min(Math.max(0, t - lastT), GAP_CAP_MS);
    lastT = t;
    const starts = new Set<WorkStage>();
    while (i <= prompt.endIndex && events[i]!.t === t) {
      const event = events[i++]!.event;
      if (event.type === 'tool_execution_start') {
        const id = String(event.toolCallId);
        const kind = event.toolName === 'ask' ? 'ask' : classify(event);
        // Retransmitted starts are not new calls or episodes.
        if (open.has(id) && open.get(id) === kind) continue;
        stop(id);
        open.set(id, kind);
        if (kind === 'ask') asking++;
        else if (kind) {
          if (kind === 'verify') verifying++;
          starts.add(kind);
        }
      } else if (event.type === 'tool_execution_end') stop(String(event.toolCallId));
      else if (event.type === 'agent_end') { open.clear(); asking = 0; verifying = 0; }
    }
    if (starts.size) {
      let count = 0;
      for (const kind of starts) if (!previous.has(kind)) {
        count++;
        if (kind === 'verify') verifications++;
      }
      episodes += count;
      if (count) runs.push({ t, count });
      const next = stage !== 'none' && starts.has(stage) ? stage : order.find(kind => starts.has(kind))!;
      if (next !== stage) { stage = next; stageAtMs = active; }
      previous = starts;
    }
    snapshots.push({ t, activeMs: active, asking, stage, stageAtMs, episodesSeen: episodes, verificationsSeen: verifications, verifying: verifying > 0 });
  }
  const prefix = (atMs: number, elapsedS?: number): PrefixState => {
    let lo = 0, hi = snapshots.length;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (snapshots[mid]!.t <= atMs) lo = mid + 1; else hi = mid; }
    const snap = snapshots[lo - 1];
    if (!snap) return { stage: 'none', episodesSeen: 0, verificationsSeen: 0, elapsedSinceStageS: 0, verifying: false };
    const nowActive = elapsedS === undefined
      ? snap.activeMs + (snap.asking ? 0 : Math.min(Math.max(0, atMs - snap.t), GAP_CAP_MS))
      : elapsedS * 1000;
    return { stage: snap.stage, episodesSeen: snap.episodesSeen, verificationsSeen: snap.verificationsSeen,
      elapsedSinceStageS: snap.stage === 'none' ? 0 : Math.max(0, nowActive - snap.stageAtMs) / 1000,
      verifying: snap.verifying };
  };
  return {
    prefix,

    at(atMs, elapsedS) {
      let lo = 0, hi = runs.length;
      while (lo < hi) { const mid = (lo + hi) >>> 1; if (runs[mid]!.t <= atMs) lo = mid + 1; else hi = mid; }
      let count = 0;
      for (let i = lo; i < runs.length; i++) count += runs[i]!.count;
      return { ...prefix(atMs, elapsedS), band: bandOf(count) };
    },
  };
}
/** Independent, offline cross-check: confirmed todo completions, not stages. */
export function todoTimeline(trace: Trace): Array<(atMs: number) => Band> {
  const facts = emptyFacts();
  const completions: number[][] = trace.prompts.map(() => []);
  let promptIndex = 0;
  for (let i = 0; i < trace.events.length; i++) {
    while (promptIndex < trace.prompts.length && i > trace.prompts[promptIndex]!.endIndex) promptIndex++;
    const { t, event } = trace.events[i]!;
    const inPrompt = promptIndex < trace.prompts.length && i >= trace.prompts[promptIndex]!.startIndex;
    const started = event.type === 'tool_execution_end' ? facts.underway.get(String(event.toolCallId)) : undefined;
    const isTodoEnd = event.type === 'tool_execution_end' && event.isError !== true
      && (started?.name ?? event.toolName) === 'todo';
    const before = facts.todoDoneEventsS.length;
    applyEvent(facts, event, t);
    if (inPrompt && isTodoEnd) {
      const added = Math.max(0, facts.todoDoneEventsS.length - before);
      for (let n = 0; n < added; n++) completions[promptIndex]!.push(t);
    }
  }
  return completions.map(times => atMs => {
    let lo = 0, hi = times.length;
    while (lo < hi) { const mid = (lo + hi) >>> 1; if (times[mid]! <= atMs) lo = mid + 1; else hi = mid; }
    return bandOf(times.length - lo);
  });
}
