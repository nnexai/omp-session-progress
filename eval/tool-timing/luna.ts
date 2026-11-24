#!/usr/bin/env bun
// The RPC host runs in a disposable OMP session. Only its validated packet reaches the model.
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, relative, resolve, sep } from 'node:path';

const [inputArg, outputArg, ...extra] = process.argv.slice(2);
if (!inputArg || !outputArg || extra.length) throw Error('Usage: bun eval/tool-timing/luna.ts <private-input-json> <private-output-json>');
function privateDirectory(path: string) {
 const dir = realpathSync(path), tmp = realpathSync(tmpdir()), rel = relative(tmp, dir), st = statSync(dir);
 if (!rel || rel === '..' || rel.startsWith(`..${sep}`) || !st.isDirectory() || (st.mode & 0o077) || st.uid !== process.getuid?.()) throw Error('Input and output must be in user-owned private directories below the temporary directory');
 return dir;
}
const input = realpathSync(inputArg), output = resolve(outputArg);
privateDirectory(dirname(input));
privateDirectory(dirname(output));
const inputStat = statSync(input);
if (!inputStat.isFile() || (inputStat.mode & 0o077) || inputStat.uid !== process.getuid?.()) throw Error('Input must be a user-owned private file');
if (input === output) throw Error('Input and output must differ');
const raw = readFileSync(input, 'utf8');
if (Buffer.byteLength(raw) > 2_000_000) throw Error('Input too large');
const items: unknown = JSON.parse(raw);
if (!Array.isArray(items) || items.length > 500) throw Error('Expected an array of at most 500 inputs');
const scratch = mkdtempSync(join(tmpdir(), 'omp-tool-timing-'));
try {
 chmodSync(scratch, 0o700);
 mkdirSync(join(scratch, 'sessions'), { mode: 0o700 });
 mkdirSync(join(scratch, 'work'), { mode: 0o700 });
 const overlay = join(scratch, 'overlay.yml'), job = join(scratch, 'job.json'), result = join(scratch, 'result.json');
 writeFileSync(overlay, 'memory:\n  backend: off\nautolearn:\n  enabled: false\nadvisor:\n  enabled: false\n', { mode: 0o600 });
 writeFileSync(job, JSON.stringify({ items, result }), { mode: 0o600 });
 const host = join(import.meta.dir, 'luna-host.ts');
 const proc = Bun.spawn(['omp', '--mode', 'rpc', '--no-extensions', '-e', host, '--session-dir', join(scratch, 'sessions'), '--cwd', join(scratch, 'work'), '--config', overlay, '--no-skills', '--no-rules', '--no-context-files', '--no-title', '--no-lsp'], { cwd: join(scratch, 'work'), stdin: 'pipe', stdout: 'pipe', stderr: 'ignore' });
 // The deadline also covers startup, credential resolution, and a stuck host.
 const timer = setTimeout(() => proc.kill(), Math.min(1_800_000, 120_000 + Math.ceil(items.length / 4) * 65_000));
 let done = false;
 try {
  proc.stdin.write(`${JSON.stringify({ type: 'prompt', id: 'tool-timing', message: `/tool-timing ${job}` })}\n`); proc.stdin.flush();
  const decoder = new TextDecoder(); let buffer = '';
  for await (const chunk of proc.stdout) {
   buffer += decoder.decode(chunk, { stream: true });
   if (buffer.length > 1_000_000) { proc.kill(); break; }
   for (let n = buffer.indexOf('\n'); n >= 0; n = buffer.indexOf('\n')) {
    const line = buffer.slice(0, n); buffer = buffer.slice(n + 1);
    if (!line.includes('tool-timing:')) continue;
    try {
     const frame = JSON.parse(line);
     if (frame.type === 'extension_ui_request' && frame.method === 'notify' && (frame.message === 'tool-timing: done' || frame.message === 'tool-timing: failed')) { done = frame.message === 'tool-timing: done'; proc.kill(); }
    } catch { /* Ignore unrelated RPC frames. */ }
   }
  }
  await proc.exited;
 } finally { clearTimeout(timer); proc.kill(); }
 let report: { answers?: unknown[]; errors?: Record<string, number> } | undefined;
 try { report = JSON.parse(readFileSync(result, 'utf8')); } catch { /* No complete report. */ }
 if (report?.errors && Object.keys(report.errors).length) console.error(`Tool timing errors: ${JSON.stringify(report.errors)}`);
 if (!done || !report || !Array.isArray(report.answers) || report.answers.length !== items.length || Object.keys(report.errors ?? {}).length) throw Error('Tool timing host failed or exceeded the wall deadline');
 // Refuse to overwrite existing files, including symlinks.
 writeFileSync(output, JSON.stringify(report.answers), { mode: 0o600, flag: 'wx' });
} finally { rmSync(scratch, { recursive: true, force: true }); }
