// Local-only corpus builder. No transcript content or identifying paths leave the private root.
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join, relative, resolve } from 'node:path';
import { homedir } from 'node:os';
import { activeSegments, extractSessionText, pseudonym } from '../extract';
import { frozenText } from '../frozen-session';
import type { Entry } from '../tool-timing/cohort';

const [rootArg, foldsArg, officialArg] = process.argv.slice(2);
if (!rootArg || !foldsArg || !officialArg) throw Error('Expected private study root, frozen folds directory, and official index directory');
const root = resolve(rootArg), folds = resolve(foldsArg), officialDir = resolve(officialArg);
if ((statSync(root).mode & 0o077) !== 0 || existsSync(join(root, 'cohort.json'))) throw Error('Expected private root without an existing cohort');
const hash = (s: string | Buffer) => createHash('sha256').update(s).digest('hex');
const readEntries = (name: string): Entry[] => JSON.parse(readFileSync(join(folds, name, 'entries.json'), 'utf8'));
const legacy = [0, 1, 2].map(i => readEntries(`fold${i}`));
const enriched = readEntries('test');
const index: Record<string, string> = JSON.parse(readFileSync(join(officialDir, 'index.json'), 'utf8'));
const official: Entry[] = Object.entries(index).map(([id, path]) => ({ id, path: resolve(officialDir, path), digest: hash(readFileSync(resolve(officialDir, path))), group: 'official' }));
const sources = [join(homedir(), '.omp/agent/sessions'), join(homedir(), '.local/state/cockpit/orchestration'), join(homedir(), '.local/share/herdr-kit/coord/sessions'), join(homedir(), '.local/state/cockpit/operations')];
const frozenDir = join(root, 'sessions'); mkdirSync(frozenDir, { mode: 0o700, recursive: true });
interface Row { type?: string; id?: string; version?: number; cwd?: string; title?: string; parentSession?: string; parentSessionId?: string; parentId?: string | null; message?: { role?: string; content?: string | { type?: string; text?: string }[] } }
type Meta = { ids: Set<string>; prompts: Set<string>; family: string; headerId: string };
function metadata(text: string): Meta | null {
  const rows: Row[] = [];
  for (const line of text.split('\n')) { if (!line.trim()) continue; try { rows.push(JSON.parse(line)); } catch { return null; } }
  const h = rows.find(x => x.type === 'session');
  if (!h || h.version !== 3 || typeof h.id !== 'string' || typeof h.cwd !== 'string') return null;
  const cwd = h.cwd.toLowerCase();
  // Normalize worktrees to their underlying project, without persisting that name.
  const project = cwd.match(/\/(?:dev\/prj|worktrees|projects)\/([^/]+)/)?.[1] ?? cwd;
  const family = hash(project);
  const ids = new Set<string>(), prompts = new Set<string>();
  for (const r of rows) {
    if (r.type !== 'session' && typeof r.id === 'string') ids.add(r.id);
    if (r.type !== 'message' || r.message?.role !== 'user') continue;
    const content = r.message.content;
    const t = (typeof content === 'string' ? content : Array.isArray(content) ? content.filter(b => b.type === 'text').map(b => b.text ?? '').join('\n') : '').replace(/\s+/g, ' ').trim().toLowerCase();
    if (t.length >= 40) prompts.add(hash(t));
  }
  return { ids, prompts, family, headerId: h.id };
}
const occupiedIds = new Set<string>(), occupiedPrompts = new Set<string>(), occupiedHeaders = new Set<string>();
const familySets = new Map<string, Set<string>>();
function register(m: Meta, set: string) {
  for (const id of m.ids) occupiedIds.add(id);
  for (const p of m.prompts) occupiedPrompts.add(p);
  occupiedHeaders.add(m.headerId);
  const sets = familySets.get(m.family) ?? new Set<string>(); sets.add(set); familySets.set(m.family, sets);
}
// Preserve original cohort identities and frozen bytes. Existing overlaps are reported,
// never silently repaired; additions may not duplicate any existing history or prompt.
let legacyOverlap = 0;
for (const [set, entries] of [...legacy.map((es, i) => [`dev${i}`, es] as const), ['enriched', enriched] as const, ['official', official] as const]) {
  for (const e of entries) {
    const text = frozenText(e), m = metadata(text);
    if (!m) throw Error('Invalid frozen session');
    if (occupiedHeaders.has(m.headerId) || [...m.ids].some(id => occupiedIds.has(id)) || [...m.prompts].some(p => occupiedPrompts.has(p))) legacyOverlap++;
    register(m, set);
    const path = join(frozenDir, `${e.id}-${e.digest.slice(0, 12)}.jsonl`);
    if (!existsSync(path)) writeFileSync(path, text, { mode: 0o600, flag: 'wx' });
    e.path = path;
  }
}
// The historical refit split was not deduplicated by history. Preserve it in
// legacy, but keep overlapping heldout requests and cross-fold histories out
// of the new training folds. Heldout cohorts remain unchanged for continuity.
const protectedIds = new Set<string>(), protectedPrompts = new Set<string>(), protectedHeaders = new Set<string>();
function protect(m: Meta) {
  protectedHeaders.add(m.headerId);
  for (const id of m.ids) protectedIds.add(id);
  for (const p of m.prompts) protectedPrompts.add(p);
}
for (const e of [...official, ...enriched]) protect(metadata(frozenText(e))!);
let legacyRemoved = 0;
const safeLegacy = legacy.map(entries => entries.filter(e => {
  const m = metadata(frozenText(e))!;
  if (protectedHeaders.has(m.headerId) || [...m.ids].some(id => protectedIds.has(id)) || [...m.prompts].some(p => protectedPrompts.has(p))) { legacyRemoved++; return false; }
  protect(m);
  return true;
}));
const candidates: string[] = [];
function walk(dir: string) {
  if (!existsSync(dir)) return;
  for (const ent of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, ent.name);
    if (ent.isDirectory()) { if (!/^(blobs|node_modules|target|fixtures?|experiments?|smoke|test-results)$/i.test(ent.name)) walk(p); }
    else if (ent.isFile() && ent.name.endsWith('.jsonl')) candidates.push(p);
  }
}
for (const source of sources) walk(source);
// Custom-session-files contain pointers, not transcripts. Follow only real,
// approved local stores; never traverse arbitrary paths from pointer contents.
for (const pointerDir of [join(homedir(), '.omp/custom-session-files'), join(homedir(), '.omp/agent/custom-session-files')]) {
  if (!existsSync(pointerDir)) continue;
  for (const name of readdirSync(pointerDir)) {
    const p = join(pointerDir, name);
    if (!statSync(p).isFile()) continue;
    const target = readFileSync(p, 'utf8').trim();
    if (target.endsWith('.jsonl') && [join(homedir(), '.local/state/cockpit/'), join(homedir(), '.local/share/herdr-kit/coord/sessions/'), join(homedir(), '.omp/agent/sessions/')].some(prefix => target.startsWith(prefix)) && existsSync(target) && statSync(target).isFile()) candidates.push(target);
  }
}
const uniqueCandidates = [...new Set(candidates)].sort();
// The shipped trainer enumerated exactly one directory below SESSIONS_DIR.
// Any matching history there could already be in its weights, even if it was
// absent from the later development manifest. Conservatively fence copies too.
const trainedIds = new Set<string>(), trainedPrompts = new Set<string>(), trainedHeaders = new Set<string>();
for (const path of uniqueCandidates) {
  const rel = relative(sources[0]!, path);
  if (rel.startsWith('..') || rel.split('/').length !== 2) continue;
  const st = statSync(path);
  if (st.size < 200 || st.size > 64 * 1024 * 1024) continue;
  const m = metadata(readFileSync(path, 'utf8'));
  if (!m) continue;
  trainedHeaders.add(m.headerId);
  for (const id of m.ids) trainedIds.add(id);
  for (const p of m.prompts) trainedPrompts.add(p);
}
const reject: Record<string, number> = {};
const rejected = (reason: string) => { reject[reason] = (reject[reason] ?? 0) + 1; };
type Candidate = { entry: Entry; meta: Meta; band: string; possiblePretrained: boolean };
const pool: Candidate[] = [];
for (const path of uniqueCandidates) {
  const st = statSync(path);
  if (st.size < 200 || st.size > 64 * 1024 * 1024 || st.mtimeMs > Date.now() - 24 * 3600_000) { rejected('size-or-recent'); continue; }
  const rel = relative(sources[0]!, path);
  if (!rel.startsWith('..') && rel.split('/').length !== 2) { rejected('native-subagent'); continue; }
  // Cockpit operation stores and Herdr coordinator stores are managed actor
  // transcripts, not proven independent human mains. Never score them as such.
  if (path.startsWith(join(homedir(), '.local/state/cockpit/'))) { rejected('cockpit-operational'); continue; }
  // Coordinator-managed transcripts are not independent user mains.
  if (path.startsWith(sources[2]!)) { rejected('managed'); continue; }
  const lower = path.toLowerCase();
  if (/omp-session-progress|omp-progress-|smoke|fixture|experiment|ratchet|benchmark/.test(lower)) { rejected('scope'); continue; }
  const bytes = readFileSync(path), text = bytes.toString('utf8'), m = metadata(text);
  if (!m) { rejected('format'); continue; }
  const rows: Row[] = text.split('\n').filter(Boolean).map(line => JSON.parse(line));
  const h = rows.find(r => r.type === 'session')!;
  const title = rows.find(r => r.type === 'title')?.title ?? h.title ?? '';
  if (h.parentSession || h.parentSessionId || h.parentId || /omp-session-progress|omp-progress-|smoke|fixture|experiment|ratchet|benchmark/i.test(String(h.cwd) + ' ' + title)) { rejected('scope-or-fork'); continue; }
  // Reject explicit delegation/worker bootstrap and harness-authored prompts.
  const firstUser = rows.find(r => r.type === 'message' && r.message?.role === 'user');
  const firstText = JSON.stringify(firstUser?.message?.content ?? '');
  if (/worker agent|delegated task|you are a (?:subagent|worker)|⁂ role|cockpit_inbox|coord delegate/i.test(firstText) || /\b(?:worker|subagent)\b/i.test(title)) { rejected('worker-or-automated'); continue; }
  if (occupiedHeaders.has(m.headerId) || [...m.ids].some(id => occupiedIds.has(id)) || [...m.prompts].some(p => occupiedPrompts.has(p))) { rejected('history-overlap'); continue; }
  const trace = extractSessionText(text, path);
  let max = 0;
  for (const p of trace.prompts) if (p.outcome === 'completed') max = Math.max(max, activeSegments(trace.events, p).reduce((n, s) => n + s.ms, 0));
  if (max <= 5000) { rejected('no-scored-prompt'); continue; }
  const band = max >= 1800_000 ? '30+' : max >= 900_000 ? '15-30' : max >= 180_000 ? '3-15' : 'short';
  const id = pseudonym('s', m.headerId), digest = hash(bytes);
  const possiblePretrained = trainedHeaders.has(m.headerId) || [...m.ids].some(id => trainedIds.has(id)) || [...m.prompts].some(p => trainedPrompts.has(p));
  pool.push({ entry: { id, path, digest, group: `${band}/other` }, meta: m, band, possiblePretrained });
  // Duplicates among new candidates are excluded conservatively before splitting.
  register(m, 'candidate');
}
// Hold out a fixed hash-ranked quarter of independent new sessions per duration.
// Where a project is already represented, this is a project-overlapping holdout,
// not a new-project generalization claim. History and request dedup remain strict.
const freshIds = new Set<string>();
for (const band of ['short', '3-15', '15-30', '30+']) {
  const eligible = pool.filter(c => c.band === band && !c.possiblePretrained).sort((a, b) => hash('fresh-v2:' + a.entry.id).localeCompare(hash('fresh-v2:' + b.entry.id)));
  for (const c of eligible.slice(0, Math.floor(eligible.length / 4))) freshIds.add(c.entry.id);
}
const fresh: Entry[] = [], dev = safeLegacy.map(es => [...es]), aux: Entry[] = [];
for (const c of pool) {
  const existing = [...(familySets.get(c.meta.family) ?? [])].filter(x => x !== 'candidate');
  const bytes = readFileSync(c.entry.path);
  if (hash(bytes) !== c.entry.digest) { rejected('changed'); continue; }
  const path = join(frozenDir, `${c.entry.id}-${c.entry.digest.slice(0, 12)}.jsonl`);
  if (!existsSync(path)) writeFileSync(path, bytes, { mode: 0o600, flag: 'wx' }); c.entry.path = path;
  if (freshIds.has(c.entry.id)) fresh.push(c.entry);
  else {
    const fold = existing.length === 1 && existing[0]!.startsWith('dev') ? Number(existing[0]!.slice(3)) : parseInt(hash('fold-v1:' + c.meta.family).slice(0, 8), 16) % 3;
    dev[fold]!.push(c.entry);
  }
}
const stats = { discovered: uniqueCandidates.length, legacy: legacy.map(x => x.length), legacyRemoved, dev: dev.map(x => x.length), newMainDev: dev.reduce((n, x, i) => n + x.length - safeLegacy[i]!.length, 0), fresh: fresh.length, excludedFreshPossiblePretrained: pool.filter(c => c.possiblePretrained).length, official: official.length, enriched: enriched.length, aux: aux.length, legacyOverlap, rejected: reject, freshBands: Object.fromEntries(['short', '3-15', '15-30', '30+'].map(b => [b, fresh.filter(e => e.group.startsWith(b + '/')).length])) };
writeFileSync(join(root, 'cohort.json'), JSON.stringify({ dev, legacy, fresh, official, enriched, aux, stats }), { mode: 0o600, flag: 'wx' });
console.log(JSON.stringify(stats));
