#!/usr/bin/env bash
# Interactive-TUI check on the installed omp, in a disposable session on a
# private tmux server (-L omp-progress-check; your tmux is not touched).
# Captures the screen as text at 100 and 54 columns while:
#   1. /progress on (instant local opt-in, no dialog) and a constrained five-phase prompt run,
#   2. /new removes the widget; /progress on again and an underspecified prompt runs at 54 columns,
#   3. /progress off removes it.
# Usage: scripts/tui-check.sh   Evidence: evidence/tui-<time>/
set -u
P=$(cd "$(dirname "$0")/.." && pwd)
OUT="$P/evidence/tui-$(date -u +%Y-%m-%dT%H-%M-%SZ)"
S=$(mktemp -d /tmp/omp-progress-tui-XXXX)
mkdir -p "$OUT" "$S/work" "$S/sessions"
printf 'memory:\n  backend: off\nautolearn:\n  enabled: false\nadvisor:\n  enabled: false\n' > "$S/overlay.yml"
T="tmux -L omp-progress-check"
n=0
cap() { n=$((n + 1)); $T capture-pane -p -t check > "$OUT/$(printf %02d $n)-$1.txt"; }
keys() { $T send-keys -t check -l "$1"; sleep 1; $T send-keys -t check Enter; }

$T kill-server 2>/dev/null
env -u HC_TERMINAL_ID -u HERDR_PANE_ID -u HERDR_WORKSPACE_ID -u TMUX \
	$T -f /dev/null new-session -d -s check -x 100 -y 42 -c "$S/work" \
	"omp --no-extensions -e $P/src/index.ts --session-dir $S/sessions --cwd $S/work --model @tiny --config $S/overlay.yml --no-skills --no-rules --no-context-files --no-title --no-lsp --auto-approve --max-cost 1"
sleep 8
keys "/progress on"; sleep 2; cap opted-in-100

keys "Synthetic test task in this scratch dir. Track phases with the todo tool (init first, mark each done): 1) create data.txt with numbers 1-5, one per line; 2) run: sleep 12; wc -l data.txt; 3) create sum.sh printing the sum of data.txt, then run: sleep 12; sh sum.sh; 4) run: sleep 12; cat missing-input.txt (expected to fail, continue); 5) reply with one sentence stating the sum. Do nothing else."
for i in $(seq 1 14); do sleep 5; cap "fixture-100-$((i * 5))s"; done
$T resize-window -t check -x 54; sleep 2; cap fixture-final-54
$T resize-window -t check -x 100; sleep 2

keys "/new"; sleep 3; cap after-new
keys "/progress on"; sleep 2
$T resize-window -t check -x 54; sleep 1
keys "Have a look around this folder and tidy things up a bit if it seems worth it."
for i in $(seq 1 5); do sleep 5; cap "ambiguous-54-$((i * 5))s"; done
$T resize-window -t check -x 100; sleep 2; cap ambiguous-final-100

keys "/progress off"; sleep 2; cap after-off
cp "$S"/sessions/*.jsonl "$OUT/" 2>/dev/null
keys "/exit"; sleep 3
$T kill-server 2>/dev/null
rm -rf "$S"
echo "$OUT"
