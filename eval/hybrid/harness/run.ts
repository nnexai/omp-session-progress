#!/usr/bin/env bun
// Run from the project root; output contains aggregate grades, not session content.
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, relative, resolve, sep } from 'node:path';

const args = process.argv.slice(2);
const values = new Map<string, string>();
for (let i = 0; i < args.length; i += 2) {
 const name = args[i];
 if (!['--variant', '--out', '--reps', '--concurrency', '--timeout-s'].includes(name) || !args[i + 1] || args[i + 1].startsWith('--') || values.has(name)) throw Error('Invalid arguments');
 values.set(name, args[i + 1]);
}
const variant = values.get('--variant');
const outArg = values.get('--out');
if (!variant || !/^(baseline|v[1-9]\d*|pilot|error-pilot)$/.test(variant) || !outArg) throw Error('Required: --variant baseline|vN|pilot|error-pilot --out DIR');
function integer(name: string, fallback: number, max: number): number {
 const raw = values.get(name); const n = raw === undefined ? fallback : Number(raw);
 if (!Number.isInteger(n) || n < 1 || n > max) throw Error(`Invalid ${name}`);
 return n;
}
const reps = integer('--reps', 1, 100);
const concurrency = integer('--concurrency', 2, 8);
const timeoutS = integer('--timeout-s', 7200, 86400);
const project = resolve(import.meta.dir, '../../..');
if (resolve('.') !== project) throw Error('Run from the project root');
const manifest = JSON.parse(readFileSync(join(project, 'eval/hybrid/manifest.json'), 'utf8')) as Array<{id:string;tag:string;short:number;commitment:string}>;
const pilot = variant === 'pilot' || variant === 'error-pilot';
const cases = manifest.filter(c => (c.tag === 'pilot') === pilot);
if (pilot && cases.length !== 5) throw Error('Invalid pilot manifest');
let train: string[] = [];
if (!pilot) {
 const state = JSON.parse(readFileSync(join(project, '.omp/ratchet/progress-hybrid-horizon/_state.json'), 'utf8'));
 if (!Array.isArray(state.train_ids) || !Array.isArray(state.test_ids)) throw Error('Missing ratchet split');
 const ids = [...state.train_ids, ...state.test_ids];
 if (ids.length !== cases.length || new Set(ids).size !== cases.length || cases.some(c => !ids.includes(c.id))) throw Error('Ratchet split does not match manifest');
 train = state.train_ids;
}
const out = resolve(outArg);
if (pilot) {
 if (!existsSync(out)) throw Error('Pilot --out must be an existing private temporary directory');
 const realOut = realpathSync(out), tmp = realpathSync(tmpdir()), rel = relative(tmp, realOut), info = statSync(realOut);
 if (!rel || rel === '..' || rel.startsWith(`..${sep}`) || !info.isDirectory() || (info.mode & 0o077) !== 0 || info.uid !== process.getuid?.()) throw Error('Pilot --out must be a user-owned 0700 directory below the temporary directory');
} else mkdirSync(out, { recursive: true, mode: 0o700 });
const lock = join(out, '.run-lock');
try { mkdirSync(lock, { mode: 0o700 }); } catch { throw Error('Output directory is locked by another run'); }
let scratch = '';
try {
 // Bind a resumable directory to one experiment and its exact case set.
 const identity = JSON.stringify({ variant, cases, train });
 const identityPath = join(out, '.run-identity.json');
 if (existsSync(identityPath) && readFileSync(identityPath, 'utf8') !== identity) throw Error('Output directory belongs to another run');
 if (!existsSync(identityPath)) writeFileSync(identityPath, identity, { mode: 0o600 });
 scratch = mkdtempSync(join(tmpdir(), 'omp-hybrid-eval-'));
 const overlay = join(scratch, 'overlay.yml');
 writeFileSync(overlay, 'memory:\n  backend: off\nautolearn:\n  enabled: false\nadvisor:\n  enabled: false\n', { mode: 0o600 });
 mkdirSync(join(scratch, 'sessions'), { mode: 0o700 });
 mkdirSync(join(scratch, 'work'), { mode: 0o700 });
 const job = join(scratch, 'job.json');
 writeFileSync(job, JSON.stringify({ cases, train: pilot ? cases.map(c => c.id) : train, reps, concurrency, out, variant }), { mode: 0o600 });
 const proc = Bun.spawn(['omp', '--mode', 'rpc', '--no-extensions', '-e', join(project, 'eval/hybrid/harness/host.ts'), '--session-dir', join(scratch, 'sessions'), '--cwd', join(scratch, 'work'), '--config', overlay, '--no-skills', '--no-rules', '--no-context-files', '--no-title', '--no-lsp'], { cwd: join(scratch, 'work'), stdin: 'pipe', stdout: 'pipe', stderr: 'ignore' });
 const timer = setTimeout(() => proc.kill(), timeoutS * 1000);
 let done = false;
 try {
  proc.stdin.write(`${JSON.stringify({ type: 'prompt', id: 'eval', message: `/hybrid-eval ${job}` })}\n`); proc.stdin.flush();
  const decoder = new TextDecoder(); let buffer = '';
  for await (const chunk of proc.stdout) {
   buffer += decoder.decode(chunk, { stream: true });
   for (let n = buffer.indexOf('\n'); n >= 0; n = buffer.indexOf('\n')) {
    const line = buffer.slice(0, n); buffer = buffer.slice(n + 1);
    if (!line.includes('hybrid-eval:')) continue;
    try {
     const frame = JSON.parse(line);
     if (frame.type === 'extension_ui_request' && frame.method === 'notify' && typeof frame.message === 'string' && frame.message.startsWith('hybrid-eval:')) {
      done = frame.message === 'hybrid-eval: done'; proc.kill();
     }
    } catch { /* Ignore unrelated RPC output. */ }
   }
  }
  await proc.exited;
 } finally { clearTimeout(timer); proc.kill(); }
 if (!done) throw Error('Evaluation host failed or exceeded the whole-run timeout');
 console.log('Hybrid evaluation complete');
} finally {
 if (scratch) rmSync(scratch, { recursive: true, force: true });
 rmSync(lock, { recursive: true, force: true });
}
