import { mkdtempSync, readFileSync, rmSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';

const project = resolve(import.meta.dir, '../..');
const args = process.argv.slice(2);
if (args.length !== 1 || !args[0]) {
  console.error('Usage: bun eval/slow-checks/verify-frozen.ts FROZEN_MILESTONE_DIR');
  process.exit(2);
}

// Compare the entire aggregate, not just the baseline checks inside each study.
// Counts and other integers must match exactly; the tolerance is only for
// floating-point roundoff in the computed statistics.
function matches(actual: unknown, expected: unknown): boolean {
  if (typeof actual === 'number' && typeof expected === 'number') {
    if (!Number.isFinite(actual) || !Number.isFinite(expected)) return false;
    if (Number.isInteger(actual) && Number.isInteger(expected)) return actual === expected;
    return Math.abs(actual - expected) <= 1e-8 + 1e-10 * Math.max(Math.abs(actual), Math.abs(expected));
  }
  if (Array.isArray(actual) || Array.isArray(expected)) {
    return Array.isArray(actual) && Array.isArray(expected) && actual.length === expected.length &&
      actual.every((value, i) => matches(value, expected[i]));
  }
  if (actual !== null && expected !== null && typeof actual === 'object' && typeof expected === 'object') {
    const a = actual as Record<string, unknown>, e = expected as Record<string, unknown>;
    const keys = Object.keys(e);
    return Object.keys(a).length === keys.length && keys.every(key =>
      Object.hasOwn(a, key) && matches(a[key], e[key]));
  }
  return actual === expected;
}

const frozen = resolve(args[0]);
let workspace: string | undefined;
let current = 'workspace setup';
try {
  workspace = mkdtempSync(join(tmpdir(), 'omp-verify-frozen-'));
  // The studies only read these links; their aggregate files are written into
  // the fresh private workspace, never into the archive.
  for (const name of ['fold0', 'fold1', 'fold2', 'test']) {
    symlinkSync(join(frozen, name), join(workspace, name), 'dir');
  }
  const studies = [
    { name: 'milestone stage and todo', script: 'eval/milestones/score.ts', result: 'result.json', reference: 'eval/milestones/result.json', takesOutput: false },
    { name: 'stated plan', script: 'eval/stated-plan/study.ts', result: 'stated-plan.json', reference: 'eval/stated-plan/result.json', takesOutput: true },
    { name: 'repeat verification', script: 'eval/repeat-verification/study.ts', result: 'repeat-verification.json', reference: 'eval/repeat-verification/result.json', takesOutput: true },
  ];
  for (const study of studies) {
    current = study.name;
    const result = join(workspace, study.result);
    const command = [process.execPath, join(project, study.script), workspace];
    if (study.takesOutput) command.push(result);
    // Child errors can include private paths or records; never relay their output.
    const child = Bun.spawn(command, { cwd: project, stdout: 'ignore', stderr: 'ignore' });
    if (await child.exited !== 0) throw new Error('study failed');
    const actual: unknown = JSON.parse(readFileSync(result, 'utf8'));
    const expected: unknown = JSON.parse(readFileSync(join(project, study.reference), 'utf8'));
    if (!matches(actual, expected)) throw new Error('aggregate mismatch');
    console.log(`${study.name}: aggregate verified`);
  }
} catch {
  console.error(`Frozen verification failed: ${current}.`);
  process.exitCode = 1;
} finally {
  if (workspace) {
    try { rmSync(workspace, { recursive: true, force: true }); }
    catch { console.error('Frozen verification failed: temporary workspace cleanup.'); process.exitCode = 1; }
  }
}
