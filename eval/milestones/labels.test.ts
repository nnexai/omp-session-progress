import { expect, test } from 'bun:test';
import { label, type TraceEvent } from '../extract';
import { stageTimeline, todoTimeline } from './labels';

const event = (t: number, type: string, fields: Record<string, unknown> = {}): TraceEvent => ({ t, event: { type, ...fields } });
const start = (t: number, id: string, toolName: string, intent = '', args?: object) =>
  event(t, 'tool_execution_start', { toolCallId: id, toolName, intent, args });
const end = (t: number, id: string, fields: Record<string, unknown> = {}) =>
  event(t, 'tool_execution_end', { toolCallId: id, ...fields });
const prompt = (t = 0) => event(t, 'before_agent_start', { prompt: 'Work' });
const finish = (t: number) => event(t, 'agent_end', { stopReason: 'stop' });

test('runs ignore admin, collapse repeats, and count only future starts', () => {
  const events = [prompt(), start(10, 'a', 'edit'), end(11, 'a'), start(12, 'r', 'read'), end(13, 'r'),
    start(14, 'b', 'write'), end(15, 'b'), start(20, 'v', 'bash', 'Run tests'), end(22, 'v'),
    start(23, 'w', 'wait'), end(24, 'w'), start(30, 'd', 'task', '', { tasks: [{}] }), finish(40)];
  const timeline = stageTimeline(events, label(events)[0]!);
  expect(timeline.at(14)).toMatchObject({ stage: 'edit', episodesSeen: 1, verificationsSeen: 0, band: 2 });
  expect(timeline.at(20)).toMatchObject({ stage: 'verify', episodesSeen: 2, verificationsSeen: 1, verifying: true, band: 1 });
  expect(timeline.at(22).verifying).toBe(false);
  expect(timeline.at(30)).toMatchObject({ stage: 'delegate', episodesSeen: 3, band: 0 });
});

test('ties and duplicate starts do not manufacture transitions or stale open calls', () => {
  const events = [prompt(), start(10, 'a', 'edit'), start(10, 'b', 'write'),
    start(20, 'v', 'run', 'Verify changes'), start(20, 'c', 'edit'), start(20, 'v', 'run', 'Verify changes'),
    start(30, 'x', 'edit'), end(31, 'v'), start(40, 'v', 'run', 'opaque'), finish(50)];
  const timeline = stageTimeline(events, label(events)[0]!);
  expect(timeline.at(10)).toMatchObject({ stage: 'edit', episodesSeen: 1, band: 1 });
  expect(timeline.at(20)).toMatchObject({ stage: 'edit', episodesSeen: 2, verificationsSeen: 1, verifying: true, band: 0 });
  expect(timeline.at(30).episodesSeen).toBe(2);
  expect(timeline.at(31).verifying).toBe(false);
});

test('opaque commands and nondispatch are unclassified; release intent is explicit', () => {
  const events = [prompt(), start(1, 'a', 'bash', 'Run tests/deploy.sh', { path: 'tests/deploy.sh' }),
    start(2, 'b', 'task', '', { tasks: [] }), start(3, 'c', 'hub', 'Delegate'),
    start(4, 'd', 'bash', 'Publish package'), finish(5)];
  const timeline = stageTimeline(events, label(events)[0]!);
  expect(timeline.at(3)).toMatchObject({ stage: 'none', episodesSeen: 0, band: 1 });
  expect(timeline.at(4)).toMatchObject({ stage: 'release', episodesSeen: 1, band: 0 });
});

test('prefix is independent of future tail and uses capped active time across asks', () => {
  const initial = [prompt(), start(1_000, 'e', 'edit'), start(11_000, 'a', 'ask'), end(100_000, 'a')];
  const early = [...initial, finish(110_000)];
  const late = [...initial, start(900_000, 'v', 'bash', 'Verify'), finish(910_000)];
  const a = stageTimeline(early, label(early)[0]!);
  const b = stageTimeline(late, label(late)[0]!);
  expect(a.prefix(105_000)).toEqual(b.prefix(105_000));
  expect(b.prefix(105_000).elapsedSinceStageS).toBe(15);
  expect(b.prefix(800_000).elapsedSinceStageS).toBe(610);
  expect(b.prefix(105_000, 16).elapsedSinceStageS).toBe(15);
  expect(a.at(105_000).band).toBe(0);
  expect(b.at(105_000).band).toBe(1);
});

test('verification ends on replacement and cannot leak across prompts', () => {
  const events = [prompt(), start(1, 'v', 'bash', 'Verify'), start(2, 'v', 'read'), finish(3),
    prompt(4), start(5, 'x', 'edit'), finish(6)];
  const labels = label(events);
  expect(stageTimeline(events, labels[0]!).at(2).verifying).toBe(false);
  expect(stageTimeline(events, labels[1]!).at(5)).toMatchObject({ stage: 'edit', episodesSeen: 1, verificationsSeen: 0, verifying: false });
});

test('todo cross-check counts only confirmed future completions and preserves carried plans', () => {
  const events = [prompt(), start(1, 'init', 'todo', '', { op: 'init', items: ['A', 'B', 'C'] }), end(2, 'init'),
    start(3, 'fail', 'todo', '', { op: 'done', task: 'A' }), end(4, 'fail', { isError: true }),
    start(5, 'ok', 'todo', '', { op: 'done', task: 'A' }), end(6, 'ok'), finish(7),
    prompt(8), start(9, 'done', 'todo', '', { op: 'done', task: 'B' }), end(10, 'done'),
    start(11, 'init2', 'todo', '', { op: 'init', items: ['D'] }),
    end(12, 'init2', { details: { op: 'init', phases: [{ name: 'p', tasks: [{ content: 'D', status: 'completed' }] }] } }),
    finish(13)];
  const timeline = todoTimeline({ id: 'synthetic', project: 'synthetic', events, prompts: label(events) });
  expect(timeline[0]!(0)).toBe(1);
  expect(timeline[0]!(4)).toBe(1);
  expect(timeline[0]!(6)).toBe(0);
  expect(timeline[1]!(8)).toBe(1);
  expect(timeline[1]!(10)).toBe(0);
});
