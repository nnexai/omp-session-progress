// Usage: bun eval/long-landmarks/display.ts COHORT_JSON PRIVATE_WORKROOT OUTPUT_JSON [official|enriched]
// Enriched uses ONLY isolated refits from refit.ts. Individual observations stay in memory.
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { activeSegments, extractSessionText } from '../extract';
import { frozenText } from '../frozen-session';
import { ReplayClock } from '../tool-timing/replay';
import { privateDir } from '../horizon-study/data';
import type { Split } from '../scale-study/rows';
import type { Observer } from '../../src/observer';

const [cohortPath, rootArg, output, split = 'official'] = process.argv.slice(2);
if (!cohortPath || !rootArg || !output || (split !== 'official' && split !== 'enriched')) throw Error('Expected COHORT_JSON PRIVATE_WORKROOT OUTPUT_JSON [official|enriched]');
const root = resolve(rootArg);
privateDir(root);
const cohort: Split = JSON.parse(readFileSync(cohortPath, 'utf8'));
if (cohort.official.length !== 40 || cohort.enriched.length !== 42) throw Error('Expected the frozen historical split');
const ages = [5, 60, 180, 300, 600, 900, 1200, 1500];
const arms = { v3: '41a40bf', v4: '0bada07' } as const;
type Arm = keyof typeof arms;
type Row = { sid: number; totalMinutes: number; long: boolean; scores: Record<Arm, Map<number, number | null>> };
const constructors = {} as Record<Arm, typeof Observer>;
const pointFormatters = {} as Record<Arm, (median: number, low: null, high: null, narrow: boolean) => string>;
for (const [arm, revision] of Object.entries(arms) as [Arm, string][]) {
  let dest: string;
  if (split === 'enriched') {
    dest = join(readFileSync(join(root, 'refit-root.txt'), 'utf8').trim(), revision);
    const excluded: string[] = JSON.parse(readFileSync(join(dest, '.omp/ratchet/progress-time/_state.json'), 'utf8')).test_ids;
    if (!cohort.enriched.every(e => excluded.includes(e.id))) throw Error('Refit did not protect the enriched holdout');
  } else {
    dest = mkdtempSync(join(root, `display-${arm}-${revision}-`));
    const archive = spawnSync('git', ['archive', revision, 'src'], { maxBuffer: 100 * 1024 * 1024 });
    if (archive.status !== 0) throw Error(`Unable to archive ${revision}`);
    const unpack = spawnSync('tar', ['-x', '-C', dest], { input: archive.stdout });
    if (unpack.status !== 0) throw Error(`Unable to extract ${revision}`);
  }
  constructors[arm] = (await import(join(dest, 'src/observer.ts'))).Observer;
  pointFormatters[arm] = (await import(join(dest, 'src/widget.ts'))).formatRemaining;
}
function displayedPoint(value: number, arm: Arm): number {
  const text = pointFormatters[arm](value, null, null, true);
  if (text === '<30s') return 0.25;
  const parts = /^~(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+)s)?$/.exec(text);
  if (!parts) throw Error('Unexpected point format');
  return Number(parts[1] ?? 0) * 60 + Number(parts[2] ?? 0) + Number(parts[3] ?? 0) / 60;
}
const rows = new Map<string, Row>();
for (const [sid, entry] of cohort[split].entries()) {
  const trace = extractSessionText(frozenText(entry), entry.path);
  if (trace.id !== entry.id) throw Error('Frozen session mismatch');
  for (const arm of Object.keys(arms) as Arm[]) {
    const clock = new ReplayClock();
    const observer = new constructors[arm](clock);
    observer.enable('eval', () => {});
    let cursor = 0;
    for (const [pid, prompt] of trace.prompts.entries()) {
      if (prompt.outcome !== 'completed') continue;
      const segments = activeSegments(trace.events, prompt);
      const total = segments.reduce((sum, segment) => sum + segment.ms, 0);
      const key = `${sid}:${pid}`;
      let row = rows.get(key);
      if (!row) { row = { sid, totalMinutes: total / 60_000, long: total >= 1_800_000, scores: { v3: new Map(), v4: new Map() } }; rows.set(key, row); }
      let segment = 0, sum = 0;
      const consume = () => {
        const { t, event } = trace.events[cursor++]!;
        clock.advance(t, observer);
        observer.observe('eval', event);
      };
      // Intermediate observations match the established production replay cadence.
      for (let t = 5000; t < total && t <= 1_500_000; t = t === 5000 ? 15000 : t + 15000) {
        while (sum + segments[segment]!.ms < t) sum += segments[segment++]!.ms;
        const at = segments[segment]!.t + t - sum;
        while (cursor <= prompt.endIndex && trace.events[cursor]!.t <= at) consume();
        clock.advance(at, observer);
        if (ages.includes(t / 1000)) {
          const { facts, reading } = observer.report(), estimate = reading?.estimate;
          const value = !facts.running || facts.asking.size || estimate?.progress == null ? null : estimate.remainingMinutes;
          row.scores[arm].set(t / 1000, value != null && Number.isFinite(value) ? displayedPoint(value, arm) : null);
        }
      }
      while (cursor <= prompt.endIndex) consume();
    }
    observer.disable();
  }
}
const all = [...rows.values()];
// Cluster draws resample entire contributing sessions. For a paired change, compute wins
// for the same long/shorter pair at both ages before reweighting it. Midrank all ties.
function rankInterval(rows: Row[], arm: Arm, age: number, earlier?: number) {
  const valid = rows.filter(r => r.scores[arm].get(age) != null && (earlier == null || r.scores[arm].get(earlier) != null));
  const ids = [...new Set(valid.map(r => r.sid))], n = ids.length;
  const bySid = new Map(ids.map((sid, i) => [sid, i]));
  const pos = valid.filter(r => r.long), neg = valid.filter(r => !r.long);
  if (!pos.length || !neg.length) return null;
  const wins = new Float64Array(n*n), den = new Float64Array(n*n);
  const win = (p: Row, q: Row, a: number) => {
    const d = p.scores[arm].get(a)! - q.scores[arm].get(a)!;
    return d > 0 ? 1 : d === 0 ? .5 : 0;
  };
  for (const p of pos) for (const q of neg) {
    const k = bySid.get(p.sid)! * n + bySid.get(q.sid)!;
    den[k]++; wins[k] += win(p, q, age) - (earlier == null ? 0 : win(p, q, earlier));
  }
  const draws: number[] = [], counts = new Uint32Array(n);
  let seed = 92741;
  for (let b = 0; b < 2000; b++) {
    counts.fill(0);
    for (let i = 0; i < n; i++) {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      counts[Math.floor(seed / 4294967296 * n)]++;
    }
    let w = 0, d = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const wt = counts[i]! * counts[j]!, k = i * n + j;
      w += wt * wins[k]!; d += wt * den[k]!;
    }
    if (d) draws.push(w/d);
  }
  draws.sort((a,b) => a-b);
  return { value: wins.reduce((s,v) => s+v, 0) / (pos.length*neg.length),
    ci95: [.025,.975].map(q => draws[Math.floor(q*draws.length)]), validDraws: draws.length };
}
function evaluate(subset: Row[], arm: Arm, age: number) {
  const positives = subset.filter(row => row.long).length;
  const negative = subset.length - positives;
  const forecast = subset.flatMap(row => {
    const score = row.scores[arm].get(age);
    return score == null ? [] : [{ score, long: row.long, actualRemaining: row.totalMinutes - age / 60 }];
  });
  const yes = forecast.filter(row => row.long), no = forecast.filter(row => !row.long);
  // Fixed top ceil(n/4) slots; split the cutoff tie uniformly, rather than favoring input order.
  const slots = Math.ceil(forecast.length / 4);
  forecast.sort((a, b) => b.score - a.score);
  const cutoff = slots ? forecast[slots - 1]!.score : null;
  const above = cutoff == null ? [] : forecast.filter(row => row.score > cutoff);
  const tied = cutoff == null ? [] : forecast.filter(row => row.score === cutoff);
  const captured = above.filter(row => row.long).length + (tied.length ? (slots - above.length) * tied.filter(row => row.long).length / tied.length : 0);
  const median = (v: number[]) => { v.sort((a,b) => a-b); return v.length ? (v[(v.length-1)>>1]! + v[v.length>>1]!) / 2 : null; };
  return {
    riskSet: subset.length, long: positives, shorter: negative,
    forecast: forecast.length, forecastLong: yes.length, forecastShorter: no.length,
    noForecastLong: positives - yes.length, noForecastShorter: negative - no.length,
    auc: rankInterval(subset, arm, age),
    medianLongMinutesLeft: { predicted: median(yes.map(r => r.score)), actual: median(yes.map(r => r.actualRemaining)) },
    pointImpliesHalfHour: { long: yes.filter(r => r.score + age/60 >= 30).length, shorter: no.filter(r => r.score + age/60 >= 30).length },
    topQuartile: { slots, cutoffMinutes: cutoff, expectedLongCaptured: captured,
      captureAmongForecastLong: yes.length ? captured / yes.length : null,
      captureAmongAllLong: positives ? captured / positives : null,
      precision: slots ? captured / slots : null },
  };
}
const byAge = Object.fromEntries(ages.map(age => {
  // Presence at the landmark establishes survival; null is a genuine no-forecast.
  const subset = all.filter(row => row.scores.v3.has(age));
  return [age, Object.fromEntries((Object.keys(arms) as Arm[]).map(arm => [arm, evaluate(subset, arm, age)]))];
}));
const matched = [
  { survivorAge: 600, scoreAges: [300, 600] },
  { survivorAge: 900, scoreAges: [300, 600, 900] },
].map(({ survivorAge, scoreAges }) => ({
  survivorAge,
  arms: Object.fromEntries((Object.keys(arms) as Arm[]).map(arm => {
    const survivors = all.filter(row => row.scores[arm].has(survivorAge));
    // The same complete-case prompts are compared at every score age.
    const complete = survivors.filter(row => scoreAges.every(age => row.scores[arm].get(age) != null));
    return [arm, {
      survivors: survivors.length,
      survivorLong: survivors.filter(row => row.long).length,
      survivorShorter: survivors.filter(row => !row.long).length,
      excludedForMissingAny: survivors.length - complete.length,
      excludedLong: survivors.filter(row => row.long).length - complete.filter(row => row.long).length,
      scores: Object.fromEntries(scoreAges.map(age => [age, evaluate(complete, arm, age)])),
      changesFromFirstScore: Object.fromEntries(scoreAges.slice(1).map(a => [a, rankInterval(complete, arm, a, scoreAges[0])])),
    }];
  })),
}));
const report = {
  population: split === 'official' ? 'Frozen official holdout, 40 sessions, shipped weights.' : 'Frozen duration-enriched holdout, 42 sessions. Both recipes refitted on protected development histories; these are not the shipped weights.',
  endpoint: 'Total completed active time >= 1800 seconds. Risk sets include prompts still running strictly past the landmark; ask waits excluded and silent gaps capped at 600 seconds.',
  scores: 'Production Observer smoothed remaining estimate rendered with the frozen widget narrow point formatter, converted back to minutes for ranking; <30s is one tied bucket. Wide layouts may instead show a range and are not ranked here. Independent revision clocks use established timer and checkpoint cadence; prefix evidence only. Higher ranks as more likely long.',
  method: 'Prompt-weighted AUC with half credit for ties; top ceil(forecast count/4) with uniformly shared cutoff ties. Matched comparisons use the same complete-case prompts within each arm.',
  caveat: 'Observational rankings, not calibrated probabilities or causal effects. Intervals resample contributing sessions (2000 draws), not individual checkpoints. Correlated prompts and tiny late risk sets make estimates uncertain. No held-out tuning.',
  revisions: arms, agesSeconds: ages, byAge, matched,
};
writeFileSync(output, JSON.stringify(report, null, 2) + '\n', { mode: 0o600 });
console.log(JSON.stringify(report));
