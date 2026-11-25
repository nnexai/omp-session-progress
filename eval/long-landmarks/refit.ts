// Usage: bun eval/long-landmarks/refit.ts COHORT_JSON PRIVATE_WORKROOT
// Freeze both training recipes. The enriched historical sessions are not a clean
// holdout for the shipped weights. Use only protected development histories here.
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { frozenText } from '../frozen-session';
import { privateDir } from '../horizon-study/data';
import type { Split } from '../scale-study/rows';
const [manifest, root] = process.argv.slice(2);
if (!manifest || !root) throw Error('Expected COHORT_JSON PRIVATE_WORKROOT');
privateDir(root);
const dest = mkdtempSync(join(root, 'refit-')), home = join(dest, 'home');
const sessions = join(home, '.omp/agent/sessions/frozen');
mkdirSync(sessions, {recursive: true, mode: 0o700});
const c: Split = JSON.parse(readFileSync(manifest, 'utf8'));
const protectedIds = [...c.official, ...c.enriched, ...c.fresh].map(e => e.id);
const seen = new Set(protectedIds);
for (const e of c.dev.flat()) {
  if (seen.has(e.id)) throw Error('Frozen cohort not disjoint');
  seen.add(e.id);
  writeFileSync(join(sessions, e.id + '.jsonl'), frozenText(e), {flag: 'wx', mode: 0o600});
}
await Promise.all(['41a40bf', '0bada07'].map(async rev => {
  const folder = join(dest, rev);
  mkdirSync(folder, {mode: 0o700});
  const archive = spawnSync('git', ['archive', rev, 'src', 'eval'], {maxBuffer: 32 * 1024 * 1024});
  if (archive.status !== 0 || spawnSync('tar', ['-x', '-C', folder], {input: archive.stdout}).status !== 0) throw Error('Cannot unpack frozen recipe');
  const state = join(folder, '.omp/ratchet/progress-time');
  mkdirSync(state, {recursive: true, mode: 0o700});
  writeFileSync(join(state, '_state.json'), JSON.stringify({test_ids: protectedIds}));
  const child = Bun.spawn([process.execPath, join(folder, 'eval/train.ts')], {env: {...process.env, HOME: home}, stdout: 'inherit', stderr: 'inherit'});
  if (await child.exited !== 0) throw Error(`Frozen recipe ${rev} failed`);
}));
rmSync(home, {recursive: true}); // Both trainers exited. Replay needs weights, not the temporary transcript copies.
writeFileSync(join(root, 'refit-root.txt'), dest, {flag: 'wx', mode: 0o600});
