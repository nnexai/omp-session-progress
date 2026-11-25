import type { ScoredPoint, ScoredPrompt } from './types';

const arms = ['local', 'exact', 'pooled'] as const;
type Arm = typeof arms[number];
type Values = Record<Arm, number>;
type Filter = (prompt: ScoredPrompt, point: ScoredPoint) => boolean;
type Observation = { session: string; prompt: number; call: number | null; values: Values };
type Units = { session: string; values: Values[] }[];
type ArmMeans = Record<Arm, number | null>;
interface Contrast { delta: number | null; ci95: number[] | null }
interface Contrasts { exactVsLocal: Contrast; pooledVsLocal: Contrast; pooledVsExact: Contrast }
interface Slice {
  sessions: number; prompts: number; calls: number; readings: number;
  mae: ArmMeans; miss15: ArmMeans; forecast15: ArmMeans; forecast30: ArmMeans; contrasts: Contrasts;
}
const mean = (xs: number[]) => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
const exact = (p: ScoredPoint) => !!p.timing?.exact.length;
const family = (p: ScoredPoint) => !!p.timing?.family.length;
const filters = {
  all: () => true,
  medium: (p: ScoredPrompt) => p.duration >= 180 && p.duration < 900,
  long: (p: ScoredPrompt) => p.duration >= 900,
  '30+': (p: ScoredPrompt) => p.duration >= 1800,
  short: (p: ScoredPrompt) => p.duration < 60,
  slow: (_: ScoredPrompt, p: ScoredPoint) => p.slow,
  exactMatch: (_: ScoredPrompt, p: ScoredPoint) => exact(p),
  familyOnly: (_: ScoredPrompt, p: ScoredPoint) => !exact(p) && family(p),
  anyVerification: (_: ScoredPrompt, p: ScoredPoint) => p.timing !== null,
  foregroundVerification: (_: ScoredPrompt,p:ScoredPoint) => p.timing?.background===false,
  backgroundVerification: (_: ScoredPrompt,p:ScoredPoint) => p.timing?.background===true,
  longMatched: (t:ScoredPrompt,p:ScoredPoint) => t.duration>=900 && (exact(p)||family(p)),
  halfHourMatched: (t:ScoredPrompt,p:ScoredPoint) => t.duration>=1800 && (exact(p)||family(p)),
  historyUsed: (_:ScoredPrompt,p:ScoredPoint) => !!p.tool?.used,
  verificationOver60: (_:ScoredPrompt,p:ScoredPoint) => (p.timing?.ageS??0)>60,
  crossPromptExact: (_: ScoredPrompt, p: ScoredPoint) => (p.timing?.priorPromptExact ?? 0) > 0,
  crossPromptFamily: (_: ScoredPrompt, p: ScoredPoint) => (p.timing?.priorPromptFamily ?? 0) > 0,
  crossPromptFamilyOnly: (_: ScoredPrompt, p: ScoredPoint) => (p.timing?.priorPromptExact ?? 0) === 0 && (p.timing?.priorPromptFamily ?? 0) > 0,
} satisfies Record<string, Filter>;

// Calls are unique only within a session. Avoid constructing or returning identifiers.
function counts(rows: { session: string; prompt: number; point: ScoredPoint }[]) {
  const sessions = new Set<string>(), prompts = new Set<number>();
  const calls = new Map<string, Set<number>>();
  for (const r of rows) {
    sessions.add(r.session); prompts.add(r.prompt);
    if (r.point.timing) {
      const ids = calls.get(r.session) ?? new Set<number>();
      ids.add(r.point.timing.call); calls.set(r.session, ids);
    }
  }
  return { sessions: sessions.size, prompts: prompts.size,
    calls: [...calls.values()].reduce((n, ids) => n + ids.size, 0), readings: rows.length };
}

// Each unit is averaged before aggregation: prompts for ETA, calls or prompts for tool error.
function units(rows: Observation[], weighting: 'prompt' | 'call') {
  const bySession = new Map<string, Map<number, { n: number; sums: Values }>>();
  for (const r of rows) {
    const key = weighting === 'call' ? r.call : r.prompt;
    if (key === null) continue;
    const group = bySession.get(r.session) ?? new Map();
    const unit = group.get(key) ?? { n: 0, sums: { local: 0, exact: 0, pooled: 0 } };
    unit.n++;
    for (const a of arms) unit.sums[a] += r.values[a];
    group.set(key, unit); bySession.set(r.session, group);
  }
  return [...bySession].map(([session, group]) => ({ session, values: [...group.values()].map(u =>
    Object.fromEntries(arms.map(a => [a, u.sums[a] / u.n])) as Values) }));
}
function averages(group: Units): ArmMeans {
  return Object.fromEntries(arms.map(a => [a, mean(group.flatMap(s => s.values.map(v => v[a])))])) as ArmMeans;
}

// A paired session-cluster bootstrap; each group is a stratum, equally weighted.
// Deltas are treatment minus baseline, so negative MAE differences are improvements.
function paired(groups: Units[], arm: Arm, baseline: Arm) {
  const sessionSet = new Set(groups.flatMap(g => g.map(s => s.session)));
  const sessions = [...sessionSet];
  const maps = groups.map(g => new Map(g.map(s => [s.session, s.values])));
  const effects = groups.map(g => mean(g.flatMap(s => s.values.map(v => v[arm] - v[baseline]))));
  if (effects.some(v => v === null)) return { delta: null, ci95: null };
  const delta = mean(effects as number[]);
  // A single contributing session in any stratum cannot support a cluster interval.
  if (sessions.length < 2 || groups.some(g => g.length < 2)) return { delta, ci95: null };
  const samples: number[] = [];
  let state = 201193;
  for (let rep = 0; rep < 3000; rep++) {
    const sums = groups.map(() => 0), ns = groups.map(() => 0);
    for (let j = 0; j < sessions.length; j++) {
      state = (Math.imul(1664525, state) + 1013904223) >>> 0;
      const id = sessions[Math.floor(state / 4294967296 * sessions.length)]!;
      maps.forEach((map, i) => {
        for (const v of map.get(id) ?? []) { sums[i]! += v[arm] - v[baseline]; ns[i]!++; }
      });
    }
    if (ns.every(n => n > 0)) samples.push(mean(sums.map((s, i) => s / ns[i]!))!);
  }
  samples.sort((a, b) => a - b);
  return { delta, ci95: samples.length >= 1000 ? [samples[Math.floor(.025 * samples.length)]!, samples[Math.floor(.975 * samples.length)]!] : null };
}
function contrasts(groups: Units[]) {
  return { exactVsLocal: paired(groups, 'exact', 'local'), pooledVsLocal: paired(groups, 'pooled', 'local'), pooledVsExact: paired(groups, 'pooled', 'exact') };
}

export function summarize(prompts: ScoredPrompt[]) {
  const all = prompts.flatMap((p, prompt) => p.points.map(point => ({ session: p.session, prompt, point })));
  const subset = (filter: Filter) => all.filter(r => filter(prompts[r.prompt]!, r.point));
  const eta = (rows: typeof all, metric: 'mae' | 'miss15' | 'forecast15' | 'forecast30') => units(rows.map(r => {
    const values = Object.fromEntries(arms.map(a => {
      const prediction = r.point[a];
      return [a, metric === 'mae' ? Math.abs(prediction - r.point.remaining) : metric === 'miss15' ? Number(Math.abs(prediction - r.point.remaining) >= 15) : Number(prediction >= (metric === 'forecast15' ? 15 : 30))];
    })) as Values;
    return { session: r.session, prompt: r.prompt, call: r.point.timing?.call ?? null, values };
  }), 'prompt');
  const slice = (rows: typeof all) => {
    const mae = eta(rows, 'mae');
    return { ...counts(rows), mae: averages(mae), miss15: averages(eta(rows, 'miss15')),
      forecast15: averages(eta(rows, 'forecast15')), forecast30: averages(eta(rows, 'forecast30')),
      contrasts: contrasts([mae]) };
  };
  const slices = Object.fromEntries(Object.entries(filters).map(([name, filter]) => [name, slice(subset(filter))])) as Record<keyof typeof filters, Slice>;
  const medium = eta(subset(filters.medium), 'mae'), long = eta(subset(filters.long), 'mae');
  const mediumMeans = averages(medium), longMeans = averages(long);
  const balanced = Object.fromEntries(arms.map(a => [a, mediumMeans[a] === null || longMeans[a] === null ? null : (mediumMeans[a]! + longMeans[a]!) / 2]));
  const tool = (filter: Filter, historyArm: 'exact' | 'pooled', familyHistory = false) => {
    const rows = subset((p, point) => filter(p, point) && point.timing !== null && point.tool !== null && point.tool.gold !== null);
    const observations = rows.map(r => {
      const t = r.point.tool!;
      return { session: r.session, prompt: r.prompt, call: r.point.timing!.call,
        values: { local: Math.abs(t.population - t.gold!), exact: Math.abs(t.exact - t.gold!), pooled: Math.abs(t.pooled - t.gold!) } };
    });
    const naiveObservations = rows.map(r => {
      const timing = r.point.timing!, t = r.point.tool!;
      const history = [...(familyHistory || !timing.exact.length ? timing.family : timing.exact)].sort((a, b) => a - b);
      const middle = Math.floor(history.length / 2);
      const median = history.length % 2 ? history[middle]! : (history[middle - 1]! + history[middle]!) / 2;
      const naive = Math.max(0, median - timing.ageS) / 60;
      return { session: r.session, prompt: r.prompt, call: timing.call,
        values: { local: Math.abs(t.population - t.gold!), exact: Math.abs(naive - t.gold!), pooled: Math.abs(t[historyArm] - t.gold!) } };
    });
    const weighted = (weight: 'prompt' | 'call') => {
      const group = units(observations, weight), naive = units(naiveObservations, weight);
      return { mae: averages(group), contrasts: contrasts([group]), naive: {
        mae: averages(naive).exact, naiveVsPopulation: paired([naive], 'exact', 'local'),
        historyVsNaive: paired([naive], 'pooled', 'exact'),
      } };
    };
    return { ...counts(rows), promptWeighted: weighted('prompt'), callWeighted: weighted('call') };
  };
  const coverage = Object.fromEntries(Object.entries({
    verification: (_: ScoredPrompt, p: ScoredPoint) => p.timing !== null,
    populationTimerSupported: (_: ScoredPrompt, p: ScoredPoint) => p.tool !== null,
    exactAvailable: (_: ScoredPrompt, p: ScoredPoint) => exact(p),
    familyOnlyAvailable: (_: ScoredPrompt, p: ScoredPoint) => !exact(p) && family(p),
    anyAvailable: (_: ScoredPrompt, p: ScoredPoint) => exact(p) || family(p),
    usedExact: (_: ScoredPrompt, p: ScoredPoint) => p.tool?.used === 'exact',
    usedFamily: (_: ScoredPrompt, p: ScoredPoint) => p.tool?.used === 'family',
    availableButFallback: (_: ScoredPrompt, p: ScoredPoint) => (exact(p) || family(p)) && !p.tool?.used,
    exactAvailableButNotUsed: (_: ScoredPrompt, p: ScoredPoint) => exact(p) && p.tool?.used !== 'exact',
  } satisfies Record<string, Filter>).map(([name, filter]) => [name, counts(subset(filter))]));
  return {
    method: { durationUnit: 'seconds', predictionUnit: 'minutes', weighting: 'mean readings within prompt, then mean prompts',
      bootstrap: 'paired whole-session percentile bootstrap, 3000 replicates, seed 201193; treatment minus baseline',
      toolBaseline: 'population; tool errors use matched readings with observed tool remainder',
      naive: 'median prior exact durations, otherwise family durations, minus call age, floored at zero; used-family subgroup uses family history' },
    completedPrompts: prompts.length, noReadings: prompts.filter(p => !p.points.length).length,
    slices,
    balancedMediumLong: { mae: balanced, contrasts: contrasts([medium, long]) },
    coverage,
    tool: { exactSupport: tool((_, p) => exact(p), 'exact'), anySupport: tool((_, p) => exact(p) || family(p), 'pooled'),
      exactSupportSlow: tool((_, p) => exact(p) && (p.timing?.ageS??0)>60, 'exact'),
      anySupportSlow: tool((_, p) => (exact(p) || family(p)) && (p.timing?.ageS??0)>60, 'pooled'),
      usedExact: tool((_, p) => p.tool?.used === 'exact' && exact(p), 'pooled'),
      usedFamily: tool((_, p) => p.tool?.used === 'family' && family(p), 'pooled', true) },
  };
}
