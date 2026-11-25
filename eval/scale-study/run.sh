#!/usr/bin/env bash
# Usage: bash eval/scale-study/run.sh NEW_PRIVATE_ROOT FROZEN_FOLDS OFFICIAL_INDEX_DIR [PYTHON]
# Optional PYTHON must already provide numpy, scipy and catboost (tested: 1.2.10).
# The extra argument enables the rejected challengers. No dependency downloads.
set -euo pipefail
umask 077
[[ $# == 3 || $# == 4 ]] || { echo 'Expected new private root, frozen folds, official index directory, optional Python' >&2; exit 2; }
mkdir -m 700 -- "$1"; root=$(cd "$1" && pwd); folds=$(cd "$2" && pwd); official=$(cd "$3" && pwd)
cd "$(dirname "$0")/../.."
s=eval/scale-study; mkdir "$root/base"; git archive 0053259a7d | tar -x -C "$root/base"
bun "$s/corpus.ts" "$root" "$folds" "$official"
bun "$s/fresh-extract.ts" "$root"
bun "$s/handoff-fit.ts" "$root"
bun "$s/replay.ts" "$root" control
for split in official enriched; do bun "$s/replay.ts" "$root" "$split" "$root/base" . handoff; done
for split in pi omp; do
  if bun -e 'process.exit(JSON.parse(await Bun.file(process.argv[1]).text()).length?0:1)' "$root/extra/$split-entries.json"; then bun "$s/replay.ts" "$root" "$split" "$root/base" . handoff; fi
done
for f in 0 1 2; do bun "$s/rows.ts" "$root" "$f"; done
bun "$s/operational-extract.ts" "$root"
bun "$s/managed-index.ts" "$root" "$root/extra"
bun "$s/replay.ts" "$root" managed "$root/base" . handoff
[[ $# == 4 ]] || exit 0
py=$4; export OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
for f in 0 1 2; do bun "$s/inner.ts" "$root" "$f"; bun "$s/intent-rows.ts" "$root" "$f"; done
for f in 0 1 2 final; do "$py" "$s/boost.py" "$root" fit "$f"; done
bun "$s/select.ts" "$root"
for method in utility minutes survival intent; do
  for f in 0 1 2 3; do "$py" "$s/$method.py" "$root" "$f"; done
  "$py" "$s/policy_score.py" "$root" "$method"
done
for f in 0 1 2; do "$py" "$s/posterior.py" "$root" "$f"; done
"$py" "$s/policy_score.py" "$root" posterior
# Restore the exact rejected Observer only in private scratch. Export refuses
# to relabel another head as the locked candidate if a changed corpus selects it.
cp -R "$root/base" "$root/challenger"; patch -d "$root/challenger" -p1 < "$s/rejected/integration.patch"
"$py" "$s/export.py" "$root" > "$root/challenger/src/horizon-data.json"
for split in official enriched; do bun "$s/replay.ts" "$root" "$split" "$root/base" "$root/challenger" scale; done
bun "$s/workers.ts" "$root"; "$py" "$s/workers-fit.py" "$root"
bun "$s/transfer-extract.ts" "$root"; "$py" "$s/transfer-fit.py" "$root"
"$py" "$s/operational-fit.py" "$root"
