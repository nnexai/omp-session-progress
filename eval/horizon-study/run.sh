#!/usr/bin/env bash
# Uses private folds from eval/milestones/run.sh. Local fits only; no provider calls or configuration changes.
set -euo pipefail
[[ $# == 2 ]] || { echo 'Usage: bash eval/horizon-study/run.sh FROZEN_MILESTONE_DIRECTORY OFFICIAL_INDEX_DIRECTORY' >&2; exit 2; }
folds=$(realpath "$1"); official=$(realpath "$2"); cd "$(dirname "$0")/../.."
umask 077
root=$(mktemp -d "${TMPDIR:-/tmp}/omp-horizon-study-XXXXXXXX")
mkdir "$root/baseline"
git archive refs/tags/replay/1d2301b | tar -x -C "$root/baseline"
for step in data inner; do printf '%s\n' 0 1 2 | xargs -P3 -n1 bun "eval/horizon-study/$step.ts" "$folds" "$root"; done
for step in compare calibrate hybrid range-trial; do bun "eval/horizon-study/$step.ts" "$root"; done
printf '%s\n' official enriched | xargs -P2 -n1 bun eval/horizon-study/holdout.ts "$root" "$folds" "$root/baseline" "$official"
bun eval/horizon-study/summary.ts "$root" "$root/result.json"
printf 'Private scratch and numeric calibration artifact: %s\n' "$root"
