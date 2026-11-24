#!/usr/bin/env bash
# Requires the private cohort + frozen control from eval/tool-timing/run.sh.
# This experiment is local-only. It does not run the old study or call Luna.
set -euo pipefail
if [[ $# != 1 ]]; then echo 'Usage: bash eval/milestones/run.sh <frozen-tool-timing-directory>' >&2; exit 2; fi
prior=$(realpath "$1")
cd "$(dirname "$0")/../.."
umask 077
root=$(mktemp -d "${TMPDIR:-/tmp}/omp-milestone-study-XXXXXXXX")
printf 'Private milestone workspace: %s\n' "$root"
bun eval/milestones/refit.ts "$prior" "$root"
printf '%s\n' fold0 fold1 fold2 test | xargs -P 4 -I '{}' bun "$root/{}/eval/milestones/extract.ts"
bun eval/milestones/score.ts "$root"
