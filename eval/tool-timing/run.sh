#!/usr/bin/env bash
# Explicit opt-in: the final step calls the configured pinned research model @widget role
# using allowlisted workload categories and counts. No raw historical text is sent.
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${1:-}" != --with-luna || $# != 1 ]]; then
  echo 'Usage: bash eval/tool-timing/run.sh --with-luna' >&2
  exit 2
fi
umask 077
root=$(mktemp -d "${TMPDIR:-/tmp}/omp-tool-timing-XXXXXXXX")
printf 'Private study directory: %s\n' "$root"
bun eval/tool-timing/cohort.ts "$root"
mkdir -p "$root/source/.omp/ratchet/progress-time"
cp -R src eval "$root/source/"
cp .omp/ratchet/progress-time/_state.json "$root/source/.omp/ratchet/progress-time/"
# Refit the existing recipe against only the manifest's development symlinks.
# Do not change the role model's HOME, or the shipped model in this checkout.
HOME="$root/fit-home" bun "$root/source/eval/train.ts"
bun "$root/source/eval/tool-timing/study.ts" "$root"
