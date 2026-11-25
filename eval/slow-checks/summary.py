"""Aggregate a single opted-in capture. Durations here are wall time, not ETA active time."""
import json

NS = 1_000_000_000
CATEGORIES = ('joinOnly', 'otherWorkOnly', 'joinedElsewhere', 'conflicting', 'unobserved')


def partition(start, end, number, spans):
    """Union overlapping tool intervals; never infer agent activity from a gap."""
    edges = [(start, -1, 0), (end, -1, 0)]
    for s in spans:
        lo, hi = max(start, s['start']), min(end, s['end'])
        if hi <= lo:
            continue
        kind = 1 if s['kind'] == 'other' else 0 if s['check'] == number else 2
        edges.extend(((lo, kind, 1), (hi, kind, -1)))
    active = [0, 0, 0]
    totals = dict.fromkeys(CATEGORIES, 0)
    t = start
    for at, kind, change in sorted(edges):
        flags = [bool(v) for v in active]
        category = 'unobserved' if not any(flags) else 'conflicting' if sum(flags)>1 else CATEGORIES[flags.index(True)]
        totals[category] += at-t
        if kind >= 0:
            active[kind] += change
        t = at
    return totals


def summarize(rows):
    def seconds(ns): return round(ns/NS, 4)
    wall = [r['runtime'] for r in rows]
    predicted = [r for r in rows if r['prior'] is not None]
    gates = [r['gateError'] for r in predicted if r['gateError'] is not None]
    total = sum(wall)
    return {'calls': len(rows), 'runtimeSeconds': {'sum': seconds(total), 'min': seconds(min(wall)) if wall else None,
            'max': seconds(max(wall)) if wall else None},
            'withPrior': len(predicted), 'previousRuntimeMaeSeconds': seconds(sum(abs(r['runtime']-r['prior']) for r in predicted)/len(predicted)) if predicted else None,
            'serialRuntimeVsRecordedJoinMaeSeconds': seconds(sum(abs(r['joinOnly']-r['prior']) for r in predicted)/len(predicted)) if predicted else None,
            'joinsScoredAtEntry': len(gates), 'remainingJoinMaeSeconds': seconds(sum(gates)/len(gates)) if gates else None,
            'activitySeconds': {k: seconds(sum(r[k] for r in rows)) for k in CATEGORIES},
            'explicitJoinShare': sum(r['joinOnly'] for r in rows)/total if total else None,
            'perCallJoinSecondsRange': [seconds(min(r['joinOnly'] for r in rows)), seconds(max(r['joinOnly'] for r in rows))] if rows else None}


def report(root):
    read = lambda p: json.loads(p.read_bytes())
    spans = [read(p) for p in root.glob('span-*')]
    rows, censored, failed = [], 0, 0
    for d in root.glob('check-*'):
        if not (d/'start').exists() or not (d/'end').exists():
            censored += 1
            continue
        s, e = read(d/'start'), read(d/'end')
        failed += int(e['code'] != 0)
        # Only successful checks enter the accuracy comparison, but disclose exclusions.
        if e['code'] != 0:
            continue
        number = int(d.name.split('-')[1])
        r = partition(s['ns'], e['ns'], number, spans)
        prior = s.get('prior_ns')
        # Score remaining wait at entry to an uninterrupted, targeted join. Do
        # not score joins made after exit or assume interrupted joins ran to exit.
        arrivals = [a['start'] for a in spans if a['kind']=='join' and a['check']==number
                    and s['ns'] <= a['start'] < e['ns'] <= a['end']]
        gate_error = None
        if prior is not None and arrivals and not r['conflicting']:
            at = min(arrivals)
            gate_error = abs(max(0, prior-(at-s['ns'])) - (e['ns']-at))
        r.update(runtime=e['ns']-s['ns'], prior=prior, gateError=gate_error)
        rows.append(r)
    return {'clock': 'monotonic wall time, per call; no inference from inactivity',
            'censoredOrNotLaunched': censored, 'failed': failed, 'allSuccessful': summarize(rows),
            'cold': summarize([r for r in rows if r['prior'] is None]),
            'priorAtLeast60s': summarize([r for r in rows if (r['prior'] or 0)>=60*NS]),
            'priorAtLeast600s': summarize([r for r in rows if (r['prior'] or 0)>=600*NS])}
