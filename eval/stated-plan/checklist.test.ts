import { expect, test } from 'bun:test';
import type { TodoItem } from '../../src/facts';
import { checklistRoute } from './checklist';

const item = (text: string, status: TodoItem['status'] = 'pending'): TodoItem => ({ text, status });

test('abstains for an empty or entirely completed or dropped list', () => {
  expect(checklistRoute([])).toBeNull();
  expect(checklistRoute([item('Implement changes', 'done'), item('Test', 'dropped')])).toBeNull();
});

test('a bare test is not an exhaustive terminal promise', () => {
  expect(checklistRoute([item('Test', 'in_progress')])).toEqual({ steps: ['verify'], tail: 'unknown', conditional: false });
  expect(checklistRoute([item('Test', 'in_progress'), item('Handle the remaining matter')])?.tail).toBe('unknown');
});

test('distinguishes explicit final verification or handoff from substantive future work', () => {
  expect(checklistRoute([item('Final verification', 'in_progress')])?.tail).toBe('final');
  expect(checklistRoute([item('Verify then report', 'in_progress')])?.tail).toBe('final');
  expect(checklistRoute([item('Test', 'in_progress'), item('Report results')])?.tail).toBe('final');
  expect(checklistRoute([item('Verify then implement', 'in_progress')])?.tail).toBe('more');
  expect(checklistRoute([item('Test', 'in_progress'), item('Implement changes')])?.tail).toBe('more');
  expect(checklistRoute([item('Test', 'in_progress'), item('Delegate remaining work')])?.tail).toBe('more');
  expect(checklistRoute([item('Test', 'in_progress'), item('Release package')])?.tail).toBe('more');
});

test('does not treat later unknown, blocked, conditional, or repeated verification as final', () => {
  expect(checklistRoute([item('Final verification', 'in_progress'), item('Address remaining matter')])?.tail).toBe('unknown');
  expect(checklistRoute([item('Final verification', 'in_progress'), item('Report results', 'blocked')])).toMatchObject({ tail: 'unknown', conditional: true });
  expect(checklistRoute([item('Final verification, then fix if tests fail', 'in_progress')])).toMatchObject({ tail: 'unknown', conditional: true });
  expect(checklistRoute([item('Final verification', 'in_progress'), item('Test again')])?.tail).toBe('unknown');
});

test('unresolved earlier pending work is not assumed completed or ordered after the active item', () => {
  const route = checklistRoute([item('Implement changes'), item('Final verification', 'in_progress')]);
  expect(route).toEqual({ steps: ['edit', 'verify'], tail: 'unknown', conditional: true });
});

test('current statuses and declared order are authoritative after a replan', () => {
  const before = [item('Test', 'in_progress'), item('Implement changes')];
  const after = [item('Test', 'done'), item('Implement changes', 'dropped'), item('Final verification', 'in_progress')];
  expect(checklistRoute(before)?.tail).toBe('more');
  expect(checklistRoute(after)).toEqual({ steps: ['verify'], tail: 'final', conditional: false });
});

test('caps categorical output without using a truncated list as proof of finality', () => {
  const route = checklistRoute([item('Final verification', 'in_progress'), ...Array.from({ length: 12 }, () => item('Report results'))]);
  expect(route?.steps).toHaveLength(12);
  expect(route?.tail).toBe('unknown');
  expect(route?.steps.every(step => ['edit', 'verify', 'delegate', 'release', 'inspect', 'report', 'other'].includes(step))).toBe(true);
});
