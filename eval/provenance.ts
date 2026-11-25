import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { EVAL_DIR } from './paths';

/** A private, independently recorded reference; never publish cohort digests. */
export function privateProvenance() {
  const path = process.env.OMP_PROGRESS_PROVENANCE_FILE ?? join(EVAL_DIR, 'tool-timing-provenance.json');
  const value = JSON.parse(readFileSync(path, 'utf8'));
  const provenance = value.provenance ?? value;
  for (const key of ['developmentContentSetSha256', 'holdoutContentSetSha256', 'refitWeightsSha256']) {
    if (typeof provenance[key] !== 'string' || !/^[a-f0-9]{64}$/.test(provenance[key])) throw Error('Invalid private provenance reference');
  }
  return provenance as { developmentContentSetSha256: string; holdoutContentSetSha256: string; refitWeightsSha256: string };
}
