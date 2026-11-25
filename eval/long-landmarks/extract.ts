// Usage: bun eval/long-landmarks/extract.ts PRIVATE_SCALE_ROOT NEW_PRIVATE_OUTPUT_DIR
// The existing scale-study manifest fences whole sessions and histories. No text or paths are exported.
import { closeSync, openSync, readFileSync, statSync, writeSync } from 'node:fs';
import { join } from 'node:path';
import { activeMs, applyEvent, emptyFacts, type Facts } from '../../src/facts';
import { activeSegments, extractSessionText, type TraceEvent } from '../extract';
import { frozenText } from '../frozen-session';
import type { Split } from '../scale-study/rows';
import { toolFamily } from '../tool-timing/replay';

const ages = [5, 60, 180, 300, 600, 900, 1200, 1500];
function snapshot(f: Facts, age: number) {
  const tools = [...f.underway.values()].filter(t => t.name !== 'ask');
  return {
    promptChars: f.promptChars, listItems: f.promptListItems, heavy: +f.promptKwHeavy, approval: +f.isApprovalPrompt,
    priorMax: f.priorMaxDurS, priorLast: f.priorPromptDurationsS[f.priorPromptDurationsS.length - 1] ?? 0,
    calls: f.toolCalls, turns: f.assistantTurns, writes: f.catCounts.write, tests: f.intentTestCount,
    todoOpen: f.todoUpdated ? f.todos.filter(t => t.status !== 'done' && t.status !== 'dropped').length : 0,
    todoDone: f.todoDoneEventsS.length, failures: f.failedCalls, delegates: f.taskDispatches,
    maxOpenAge: Math.max(0, ...tools.map(t => age - t.startedAtS)), maxDoneAge: f.maxToolDurS,
    files: f.filesEdited.size, reads: f.filesRead.size, scope: f.todoAppendCount,
    debug: f.debugLoops, cycles: f.editVerifyCycles, bg: f.bgJobs.size,
    open: tools.length, think: f.thinkTimeSumS, tokens: f.totalOutputTokens, wrap: +f.agentWrapUp,
  };
}

// These first onsets are retrospective *outcomes*. Never pass them or total duration into a classifier.
function onsetTracker() {
  const onsets = { execError: null, scope: null, blocked: null, slowTool: null, debugLoop: null, delegate: null } as Record<
    'execError' | 'scope' | 'blocked' | 'slowTool' | 'debugLoop' | 'delegate', number | null>;
  const verifies = new Set<string>();
  function observe(f: Facts, {event, t}: TraceEvent) {
    const s = activeMs(f, t) / 1000;
    for (const u of f.underway.values()) {
      if (u.name !== 'ask' && s - u.startedAtS >= 60) onsets.slowTool = Math.min(onsets.slowTool ?? Infinity, u.startedAtS + 60);
    }
    if (event.type === 'tool_execution_start') {
      const family = toolFamily(String(event.toolName), String(event.intent ?? ''));
      if (family === 'execute-verification' || family === 'execute-e2e') verifies.add(String(event.toolCallId));
    } else if (event.type === 'tool_execution_end' && verifies.delete(String(event.toolCallId)) && event.isError === true) {
      onsets.execError ??= s;
    }
    applyEvent(f, event, t);
    if (f.todoAppendCount > 0) onsets.scope ??= s;
    if (f.todoUpdated && f.todos.some(v => v.status === 'blocked')) onsets.blocked ??= s;
    if (f.debugLoops > 0) onsets.debugLoop ??= s;
    if (f.taskDispatches > 0) onsets.delegate ??= s;
  }
  return { onsets, observe };
}

const [root, dest] = process.argv.slice(2);
if (!root || !dest || (statSync(dest).mode & 0o077) !== 0) throw Error('Expected a private output directory');
const corpus: Split = JSON.parse(readFileSync(join(root, 'cohort.json'), 'utf8'));
const fd = openSync(join(dest, 'rows.jsonl'), 'wx', 0o600);
try {
  for (const [split, entries] of [['dev', corpus.dev.flat()], ['official', corpus.official], ['enriched', corpus.enriched]] as const) {
    let n = 0;
    for (const [sid, entry] of entries.entries()) {
      const trace = extractSessionText(frozenText(entry), entry.path);
      if (trace.id !== entry.id) throw Error('Frozen session identity mismatch');
      const f = emptyFacts(); let cursor = 0;
      for (const [pid, p] of trace.prompts.entries()) {
        if (p.outcome !== 'completed') continue;
        const seg = activeSegments(trace.events, p), total = seg.reduce((v, s) => v + s.ms, 0) / 1000;
        if (total <= ages[0]!) continue;
        // Fold earlier prompts (including interruptions), but never count their onsets as this prompt's.
        while (cursor < p.startIndex) { const e = trace.events[cursor++]!; applyEvent(f, e.event, e.t); }
        const first = trace.events[cursor++]!; applyEvent(f, first.event, first.t);
        const { onsets, observe } = onsetTracker();
        let j = 0, offsetMs = 0;
        const rows: { split: string; sid: number; pid: number; age: number; total: number; x: ReturnType<typeof snapshot> }[] = [];
        for (const age of ages) {
          if (age >= total) break;
          while (offsetMs + seg[j]!.ms < age * 1000) offsetMs += seg[j++]!.ms;
          const at = seg[j]!.t + age * 1000 - offsetMs;
          while (cursor <= p.endIndex && trace.events[cursor]!.t <= at) observe(f, trace.events[cursor++]!);
          if (!f.running || Math.abs(activeMs(f, at) - age * 1000) > 1) throw Error('Landmark diverged from production active clock');
          rows.push({ split, sid, pid, age, total, x: snapshot(f, age) });
        }
        while (cursor <= p.endIndex) observe(f, trace.events[cursor++]!);
        for (const row of rows) writeSync(fd, JSON.stringify({ ...row, onsets }) + '\n');
        n++;
      }
    }
    console.log(`${split}: ${entries.length} sessions, ${n} scored prompts`);
  }
} finally { closeSync(fd); }
