# omp-session-progress

An opt-in, read-only [OMP](https://github.com/can1357/oh-my-pi) extension that adds a native progress
widget above the editor. A compact local hybrid model (survival-regime GBDT + Truncated Log-Normal
Survival ResMLP + causal online calibration) estimates how far into the **current prompt** the agent
is and how much active working time remains—in real time (< 0.25 ms), with **zero LLM calls**,
**$0 cost**, and **nothing leaving your machine**.

```
 PROGRESS  ━━━━━━━━━━────────────  ~42%  ~2–4m left · 1m 24s               editing
```

When an active blocker occurs (provider retry, model error, blocked todo, or repeated tool errors), a
second line appears underneath:

```
 PROGRESS  ━━━━━━━━━━────────────  ~42%  ~2–4m left · 1m 24s             ⚠ blocked
  └─ ⚠ provider retry 1/3: rate limited
```

It is strictly read-only and local:

- Zero external API calls, zero model configuration needed, zero cost.
- Nothing is sent to any provider or written into the observed session.
- It never changes the session's plan, tools, transcript, or model.
- It never takes keyboard focus and writes nothing to the chat transcript.

It is off by default, and an opt-in lasts for the current session only.

## Install and use

Requires OMP 19.1.x (tested with `omp/19.1.1`). Clone the repo and symlink the entry point into OMP's
extension directory so every new `omp` session loads it:

```sh
git clone https://github.com/nnexai/omp-session-progress.git
mkdir -p ~/.omp/agent/extensions
ln -s "$PWD/omp-session-progress/src/index.ts" ~/.omp/agent/extensions/omp-session-progress.ts
```

There is nothing to build or install: OMP loads the TypeScript directly. To try it for one run
without installing, use `omp -e path/to/omp-session-progress/src/index.ts`. Installed or not, it
stays off until you run `/progress on` in a session.

| Command | Effect |
|---|---|
| `/progress on` | Turns the widget on immediately (replays the current session branch if opted in mid-prompt). |
| `/progress off` | Removes the widget and stops the live timer. |
| `/progress` | Toggles on or off. |

The widget is removed on session switch (`/new`, `/resume`, `/fork`) and on exit. `/clear`, `/branch`,
and tree navigation reset observed state.

## Reading the widget

The widget renders 1 borderless row normally (or 2 when a real observed blocker is active), styled
with OMP's active theme (`theme.progress`, `theme.tree`, `theme.status`, `theme.icon`, `theme.fg`,
`theme.bold`). It intentionally omits running/last tool labels and next-todo lines so it does not
repeat OMP's native TODO tree and tool spinner. Below 60 columns it compacts labels and drops the
right-aligned phase badge; every row is bounded to one line at any width.

| Element | Meaning |
|---|---|
| Progress bar + `~N%` | Elapsed share of active time, derived from the displayed point forecast (`elapsed / (elapsed + remaining)`). Only the actual end of the turn turns it green (`✓ 100% · done in …`). |
| `~2–4m left` / `~2m left` | Predicted active working time remaining. Shows the model's estimated middle range when uncertainty is wide, or a single estimate when tight. It is not a deadline. |
| Elapsed (`1m 24s`) | Active working time on the current prompt, ticking every second while the agent works (pauses during open `ask` waits; caps silent gaps at 10 minutes). |
| Phase badge (right) | Workflow phase inferred directly from observed events: `planning`, `exploring`, `editing`, `verifying`, `delegating (N)`, `waiting`, `retrying`, `wrapping up`, `⚠ blocked`, `done`, `idle`. |
| Blocker row (`└─ ⚠ …`) | Shown only when there is a real observed blocker (provider retry, model error, blocked todo, or consecutive tool errors). |

During the first ~3 seconds of a new prompt before any tool call or todo plan arrives, the widget
cleanly abstains (`starting…`) rather than showing a noisy guess.

## How estimation works

A compact local v4 hybrid ensemble (**6-head survival-regime GBDT + 2-seed Truncated Log-Normal
Survival ResMLP + 4-channel public-concept Ridge text priors + causal online self-calibration**,
150 general causal features, < 0.25 ms per estimate) predicts log remaining active time
`z = ln(remS + 30)` and derives progress consistently as `elapsed / (elapsed + remaining)`:

- **Cross-family model blend**:
  - **4 core GBDT heads** (`sparse`/`zero`, `dense`/`elapsed`, `allDense`/`zero` prompt-weighted across all prompts `>5s`, `clock`/`elapsed`) plus **2 long-survival GBDT specialists** (`elapsed >= 5 min`), with smooth causal regime transitions at `<25 s` and `~12 min`.
  - **2 Truncated Log-Normal Survival ResMLP heads** (`src/survival-net.ts`): pure-TypeScript residual MLPs trained with exact conditional survival negative log-likelihood `ln p(T | T > e, x)` over total duration `ln(T + 30) ~ N(μ(x), σ(x)²)`, evaluated via closed-form conditional remaining median `T₅₀(e)`. Smoothly ramped from `0%` on very short prompts to `15%` past ~60 s for genuine smooth-neural vs axis-aligned-tree diversity.
- **Strictly causal online self-calibration (`src/observer.ts`)**:
  - **Empirical-Bayes session bias**: as prompts finish within a session, tracks a recency-weighted (`0.8^k`, shrunk by `1 / (wt + 2.5)`) mean log residual of the model's own past forecasts in that session (`+0.6 × sessBias`), adapting automatically to unusually fast or slow repositories/workflows.
  - **Centered within-prompt forecast slippage**: after 60 s active elapsed, compares the current implied total duration against the model's own forecast from `<= 55%` of elapsed time, subtracts the out-of-fold octave-bin mean drift (`slipCenter`), and applies `+0.25 × centeredSlip`.
- **Calibrated uncertainty range**: two quantile heads (`qLow` P25 on `allDense`, `qHigh` P75 on `clock`)
  estimate the initial range. After one active minute, their bounds are blended with a 6×6 out-of-fold
  residual quantile table (`src/range-data.json`) indexed by elapsed age and predicted time left.
- **Confirmed plan closeout (`src/handoff.ts`)**: after the last confirmed todo on prompts at least two
  active minutes old, caps implausibly large point/range ETAs using observed plan-closure-to-end
  durations. Pending, blocked, or failed work does not qualify; any new foreground tool cancels it,
  and without new evidence it fades after 30 s and expires at 90 s.

- **150 general causal features (`src/facts.ts`, `src/features.ts`, `src/text-concepts.ts`):**
  - **4-channel 40-concept Ridge text priors**: maps prompt text (inheriting the preceding assistant's
    plan text on brief approvals like "go ahead"), remaining todo items, recent tool intents, and
    recent assistant prose onto 40 fixed public software-engineering concepts via L2 Ridge regression
    (out-of-fold cross-fitted during training; zero private vocabulary or project tokens stored).
  - **Markov tool-transition & debug-loop counters**: tracks `write → exec_err`, `exec_err → write`,
    `exec_ok → todo_done`, `exec_ok → git`, and full `write → error → write` debug loops.
  - **Async background jobs & `wait` snapshots**: tracks async `task` and `bash` launches from
    `result.details` and reconciles running/completed job counts, durations, and ages on `wait`
    snapshots, plus underway tool family & age.
  - **Tool intent semantics & edit→verify stage progression**: classifies observed tool intents
    (`test`/`verify`, `e2e`/`benchmark`/`browser`, `build`, `git`/`commit`, `release`), counts
    edit→verify cycles, stage transitions, verifications seen, tools since last write, and recency
    of errors/tests/git actions.
  - **Working-set file dynamics**: unique files read and edited (including `edit` `details.path` /
    `perFileResults`), re-edit ratio, recent read novelty, and test/doc file edit flags.
  - **Confirmed todo milestones & unshrunk velocity**: authoritative `result.details.phases` (only
    on `!isError`), per-prompt completions, phases, shrunk EWMA + raw/median unshrunk pace,
    current-todo excess duration ratio, scope growth, and remaining verify/wrap-up todo share.
  - **Context growth, thinking latency & prompt structure**: prompt-delta context tokens, compaction
    count, LLM thinking latency, tool call fanout, recent output tokens, assistant prose signals,
    `t=0` prompt structure (question, tiny command, code/log block, file paths), and causal prior
    prompt durations.
  - On mid-session opt-in (`/progress on`), `seedFromBranch` replays `ctx.sessionManager.getBranch()`
    so elapsed time, todos, tools, and prior prompt history are immediately accurate.

## Evaluation on historic sessions

The original 100-prompt benchmark uses the same four-probe grader as the earlier `@widget` runs.
Model selection used session-grouped development folds; the same 40 historical test sessions as
Neon's baseline were excluded from fitting. They are a comparison set, not a new prospective trial.

The shipped widget and unit tests need no session data. Historical replay and refitting require your **own private fixtures**: the session index in `$OMP_PROGRESS_EVAL_DIR`, the generated `eval/cases.jsonl`, and the local `.omp/ratchet` split. Per-prompt truth, hybrid manifests, provider logs and terminal captures are deliberately not distributed. A fresh clone can run `bun test` without them; the published aggregate reports are not substitutes for the missing inputs.

```sh
bun test
bun eval/run.ts              # Original benchmark plus replay of the displayed, smoothed estimate
bun eval/run.ts --horizons   # Also score every completed prompt in the 40 held-out sessions
bun eval/train.ts            # Regenerate weights locally; no API calls
```

### Does more data improve the point ETA on long tasks?

The reproducible improvement was **at handoff, not throughout the task**. We fit the correction
on 162 eligible development prompts and replayed the live, smoothed observer. At the first
confirmed closeout on tasks lasting 30+ minutes, mean absolute error fell from **1m 35s to 10s**
on the original comparison set (6 eligible prompts) and **2m 52s to 14s** on the duration-enriched
set (13 eligible prompts). These are separate comparisons; the sets share one historical request.
Over the *entire* 30+ minute trajectory the average gain was under a second in both. Sub-minute
predictions were unchanged. The correction does not make the early route predictable.

We also tried learning from 221 Codex root sessions, 61 parent-matched OMP workers and 300 managed
OMP sessions. These populations were kept separate from personal OMP sessions. Neither transfer
nor a larger boosted model justified replacing the existing predictor. The best development
survival model saved 1m 41s on 30+ minute tasks, but failed to reproduce that gain on both
historical comparisons. [Aggregate results](eval/scale-study/result.json)
and [local replay runner](eval/scale-study/run.sh).

### When can a half-hour prompt be distinguished from a shorter one?

Compare prompts **still running at the same active age**; a five-minute forecast gets no credit for separating a half-hour task from one that already ended. We replayed the rounded point shown by the narrow widget and also fitted a separate risk classifier using prompt and session context, tools, todos and workflow counts. Of 237 protected development sessions, 187 contributed 779 completed prompts over five seconds, including 39 half-hour prompts. Each surviving prompt gets one vote per landmark.

| Active age | Official: reached 30 min / still running | Duration-enriched: reached 30 min / still running | Widget ranking AUC: shipped v4 / clean v4 refit |
|---|---:|---:|---:|
| 1 min | 10 / 153 | 17 / 127 | 0.78 / 0.64 |
| 5 min | 10 / 70 | 17 / 81 | 0.72 / 0.59 |
| 10 min | 10 / 52 | 17 / 59 | 0.80 / 0.64 |
| 15 min | 10 / 31 | 17 / 37 | 0.77 / 0.71 |
| 20 min | 10 / 21 | 17 / 26 | 0.73 / 0.67 |

AUC is the chance that the widget ranks a half-hour prompt above a shorter survivor (random: 0.50). Among the **same ten-minute survivors**, v4's five-to-ten-minute update raised AUC from 0.66 to 0.80 on official and from 0.52 to 0.64 on enriched. Session-bootstrap 95% intervals for those paired gains were +0.001 to +0.318 and +0.051 to +0.234, respectively. This improvement is not just shorter runs dropping out. The small classifier did not replicate it (changes −0.10 and +0.06). At ten minutes the median displayed time left among the eventual long runs was **9 min official / 7 min enriched**, versus **40 min actually left** in both. Conversely, only 42% / 37% of the highest-ranked quarter actually ran half an hour: better ordering is not a reason to declare all high scorers long.

Official uses the shipped weights; enriched uses both frozen recipes retrained on the same protected development histories, not weights fitted on enriched sessions. These historical cohorts share one request, are not pooled, and contain only 9 / 12 sessions with half-hour prompts. The enriched prevalence is not a deployment base rate. Time is active time on completed prompts, not wall time or an evaluation of interrupted work.

At five minutes, 10/10 official and 16/17 enriched long prompts could be paired with a shorter survivor from a different session with similar observable work. Failed verification was more common on the long side in official, but the imbalance did not reproduce consistently. Neither slow tools nor newly blocked work supplied a consistent turning point. The post-landmark comparison counted only pairs both observable for the **same** further five minutes; small counts and missing/unrecorded signals do not prove the later route was unforeseeable.

[Shipped ETA rankings](eval/long-landmarks/display.json) · [Duration-enriched replication](eval/long-landmarks/display-enriched.json) · [Risk classifier and paired intervals](eval/long-landmarks/result.json) · [Matched evidence before and after each landmark](eval/long-landmarks/matched.json). To reproduce from the private [scale-study cohort](eval/scale-study/run.sh):

```sh
umask 077; work=$(mktemp -d)
bun eval/long-landmarks/extract.ts "$scale_root" "$work"
python eval/long-landmarks/separation.py "$work/rows.jsonl" > "$work/result.json"  # NumPy + SciPy
python eval/long-landmarks/matched.py --rows "$work/rows.jsonl" --output "$work/matched.json"
bun eval/long-landmarks/display.ts "$scale_root/cohort.json" "$work" "$work/display.json" official
bun eval/long-landmarks/refit.ts "$scale_root/cohort.json" "$work"
bun eval/long-landmarks/display.ts "$scale_root/cohort.json" "$work" "$work/display-enriched.json" enriched
```

**Final range trial: no change shipped.** At ten active minutes we tested raising only the upper bound for jobs whose displayed ETA crossed a cutoff learned on development sessions. The proper interval score (lower is better) improved from **30.32 to 27.68** on the 52 official survivors but worsened from **43.84 to 48.27** on the 59 duration-enriched survivors. The latter ranges widened by 4.64 minutes on average for a 1.7 percentage-point coverage gain. A useful ranking did not yield a reliable improvement in the displayed range. [Trial and acceptance rule](eval/long-landmarks/upper-trial.ts) · [Aggregate results](eval/long-landmarks/upper-result.json).

### Held-out results (40 sessions excluded from training)

| Estimator | Held-out Score | Progress Acc | Remaining Acc | Within 2× | Cost / Latency |
|---|---:|---:|---:|---:|---|
| Halfway baseline (`remaining = elapsed`) | 0.648 | 0.750 | 0.546 | 50.0% | $0 / 0 ms |
| Raw `@widget` LLM (recorded role) | 0.705 | 0.831 | 0.579 | 41.2% | $0.00024 / ~800 ms |
| Shipped `@widget` + geomean (`progress-cost/v6`) | 0.730 | 0.841 | 0.619 | 53.1% | $0.00011 / ~800 ms |
| Neon's local 3-anchor GBDT | 0.781 | 0.851 | 0.710 | 67.5% | $0 / 0.03 ms |
| Local mixed-landmark GBDT | 0.781 | 0.852 | 0.709 | 65.6% | $0 / 0.04 ms |
| Local survival-regime GBDT (140 features) | 0.781 | 0.851 | 0.711 | 67.5% | $0 / 0.09 ms |
| **Local v4 hybrid GBDT + SurvivalNet (150 features)** | **0.781** | **0.851** | **0.711** | **65.6%** | **$0 / 0.25 ms** |

The important comparison is **minutes missed while the widget is on screen**, not the four-probe
score alone. All local versions were replayed through the same production Observer, including its
one-second timer, causal online calibration (v4), and smoothing. The expanded audit scored 6,382
checkpoints across 257 completed prompts; four more ended before the first checkpoint at 5 s. No
minimum tool count was required. Each prompt gets equal weight within a row; duration bands overlap.

| Actual active duration or state | Prompts scored | Neon's 3-anchor MAE | Survival-regime 140f MAE | **v4 Hybrid + Online MAE** |
|---|---:|---:|---:|---:|
| All completed prompts | 257 | 2.32 min | 2.25 min | **2.20 min** |
| Under 1 minute | 104 | 0.49 min | 0.40 min | **0.33 min** |
| 3–15 minutes | 62 | 2.80 min | 2.68 min | **2.62 min** |
| 15+ minutes | 31 | 10.56 min | **10.19 min** | 10.30 min |
| 30+ minutes | 10 | 18.18 min | **16.69 min** | 17.45 min |
| While a non-ask tool has been running over 60 s | 50 | 5.66 min | 5.52 min | **5.47 min** |

On the 1,983-checkpoint chronological live Observer replay, v4 reached a new high of **0.7829**
(up from 0.7798; within 2× **64.9%** vs 64.7%), with **0.0%** premature `>=85%` readings and
**0/203** false `>=15 min` alarms on sub-minute prompts. Range coverage on those 1,983 readings is
**52.5%** (target 50%), and refitted out-of-fold interval scores improved across all three bands
(3–15 min: **9.2 → 8.75**, 15+ min: **35.5 → 35.19**, 30+ min: **59.2 → 59.18**). In all versions,
the target is *active* time: waiting for an `ask` answer is excluded, and a silent gap contributes
at most ten minutes.

### Can the displayed range improve without changing the point ETA?

**Yes.** On long prompts the old range was often too wide to guide a decision. The widget
now looks up the middle half of past forecast errors for similar **elapsed time and current
ETA**. A 6×6 table fitted to out-of-fold predictions adjusts the old bounds gradually, starting
after the first active minute. No transcript, extra model call, or extra widget row is needed.

| Held-out prompts (official / separate slow-task cohort) | Mean interval width, minutes (old → new) | Actual time inside range (old → new) | Interval score (old → new; lower is better) |
|---|---:|---:|---:|
| Official: 3–15 min (62) | 12.0 → 7.0 | 68% → 65% | 14.1 → 9.2 |
| Official: 15+ min (31) | 28.0 → 16.9 | 55% → 48% | 42.5 → 35.5 |
| Official: 30+ min (10) | 43.8 → 25.8 | 50% → 47% | 71.2 → 59.2 |
| Separate: 3–15 min (61) | 12.0 → 7.4 | 64% → 63% | 14.4 → 9.8 |
| Separate: 15+ min (37) | 33.0 → 20.5 | 59% → 53% | 50.9 → 44.4 |
| Separate: 30+ min (17) | 46.4 → 28.0 | 53% → 46% | 77.4 → 70.3 |

These are **central 50% ranges**, not promises. The interval score charges one minute per
minute of width and four per minute the truth falls outside. Width alone is not a win.
The table scores unrounded bounds at all eligible readings, one vote per prompt. The
[aggregate](eval/horizon-study/result.json) also reports how often the wide widget printed a
rounded range, its coverage when printed, and paired whole-session bootstrap intervals.
For 30+ minute prompts the interval-score improvement was 12.0 (95% interval 7.3–17.1)
on the official set and 7.1 (3.0–11.8) on the separate set. Long-run samples are small;
these are previously used historical sets, not a prospective trial.

**Point estimates:** We tested completion-by-deadline probabilities, remaining agent turns ×
seconds per turn, tilting toward the upper estimate, a total-duration countdown, a correction
that can move the point in either direction, and a hybrid routing short-looking work to the
turn estimator. The turn model helped shorter tasks but increased 30+ minute error. The
bidirectional correction was the strongest candidate within the guardrails: it saved 0.57
minutes on 30+ minute development prompts but cost 0.16 minutes on medium ones. Its
priority-weighted gain was below the preselected 0.25-minute threshold. We kept Neon's
point ETA, progress bar, and rendering. All estimates for sub-minute prompts are unchanged.

**Reproduce locally:** `bash eval/horizon-study/run.sh FROZEN_MILESTONE_DIRECTORY OFFICIAL_INDEX_DIRECTORY`.
The first argument is the private corpus from the milestone study; the second contains the
local `index.json` for the original benchmark. The study uses 244 development sessions in
three whole-session folds, with nested refits for calibration. It tests the original shipped
weights on 40 excluded sessions and refits the control without the 42 slow-task cohort sessions
before scoring those. Forecasts use only the current prefix; the production Observer supplies
the final smoothed readings. The private output includes `range-model.json` (the contents of
`src/range-data.json`) and an aggregate `result.json`. No provider is contacted.

### Does predicting the running tool separately help?

It helped predict **the tool**, not **the prompt**. In a separate comparison we held out 42
historical sessions outside the previous benchmark index, enriched for slow tools and 30+ minute
tasks. We refit the same local GBDT recipe without those sessions; its scores here are not scores
of the shipped weights. All three arms saw the same chronological Observer readings. Of 177
completed prompts, 176 lasted long enough to score (8,027 readings).

| Mean absolute error, active minutes | Refit local control | Tool + post-tool estimate | Same timing + Luna for post-tool work |
|---|---:|---:|---:|
| Medium (61 prompts) | 3.12 | 3.05 | 3.05 |
| Long (37 prompts) | 12.36 | 12.53 | 12.52 |
| 30+ minutes (17 prompts, 12 sessions) | 19.59 | 19.90 | 19.88 |
| Tool running over 60 seconds (60 prompts) | 7.93 | 8.12 | 8.10 |
| Medium and long weighted equally | 7.74 | 7.79 | 7.79 |

After a tool had run for 30 seconds, the experiment estimated its remaining time from comparable
calls that had survived that age. It estimated work *after that call* using similar development
prompts. For concurrent calls it selected the oldest; it did not sum overlapping runtimes. On
matched checkpoints, tool-duration error fell from **1.42 to 1.18 minutes** versus subtracting age
from the usual completed duration. Post-tool work still had **7.25 minutes** of error.

The third arm kept the same tool timing and blended the local post-tool forecast 50:50 with the
model's estimate. All 86 responses matched the pinned model; only 608 readings could use a reply while
its tool was still active. The paired change versus timing alone was −0.001 minutes (95%
session-bootstrap interval −0.009 to +0.008). None of the three arms forecast 15+ minutes for any
of the 49 sub-minute prompts. The widget therefore keeps its local predictor and existing UI.

[Aggregate scores, denominators and uncertainty](eval/tool-timing/result.json).
Unlike the widget, **this opt-in research command calls a provider**. It sends allowlisted workload categories,
counts and a local post-tool estimate, not requests, commands, paths or tool output. To run it, set
`OMP_TOOL_TIMING_EXPECTED_MODEL` to the exact model ID approved for your `@widget` role, then run
`bash eval/tool-timing/run.sh --with-luna`. Both the configured and served model must match.
The refit never replaces the shipped weights. It writes the model and content-set digests to a **private**
`provenance.json` separate from publishable aggregates. For dependent studies, set `OMP_PROGRESS_PROVENANCE_FILE`
to that file or a privately retained original tool-timing report for the same frozen cohort.

### Is knowing which milestones remain useful?

Yes in hindsight, much less so from the evidence available live. This local-only experiment used
the same frozen control, 42 held-out sessions and 8,027 readings as the tool-timing study, not a
second independent test population. It counted new runs of observable editing, verification,
delegation or release work. Repeated calls of the same kind count once; reads and opaque commands
do not count. **No new stage** does not mean **finished**: the current stage can run for minutes.
These are trace-observable stages, not a human-annotated count of deliverables.

| Mean absolute ETA error, active minutes | Local control | Predicted remaining stages | Actual stage band (hindsight only) |
|---|---:|---:|---:|
| 3–15 minutes (61 prompts) | 3.12 | 3.19 | 2.87 |
| 15+ minutes (37 prompts) | 12.36 | 12.15 | 10.83 |
| 30+ minutes (17 prompts, 12 sessions) | 19.59 | 19.26 | 17.34 |
| Medium and long weighted equally | 7.74 | 7.67 | 6.85 |

With the same estimator and inputs, except for the hindsight label, error fell from 7.82 to 7.51
minutes when told **whether any new stage remained**, and to 6.85 when told **0, 1, 2–3 or 4+**.
The four-band gain was 0.97 minutes (paired session-bootstrap 95% interval: 0.63–1.34).
In contrast, a local classifier predicting those bands saved only 0.07 minutes versus the
existing control (interval: −0.11 to +0.23 minutes saved). On explicitly identified in-flight
verifications, its “no new stage” predictions had 47% precision and 36% recall (65 readings in
32 prompts, prompt-weighted). No arm forecast 15+ minutes on any of the 49 sub-minute prompts.

As a separate check on readings with a current plan, knowing the exact **band of future confirmed
todo completions** gave only a 0.06-minute gain versus the matched estimator, with an interval
crossing zero. Counting checked boxes is not equivalent to knowing which kinds of work are ahead.
The experiment did not change the widget.

`bash eval/milestones/run.sh <frozen-tool-timing-directory>` runs without a provider. It verifies
the cohort and control digests and regenerates out-of-fold development forecasts: no session's
local ETA input comes from a model that trained on that session. The classifier sees only the
observed prefix; hindsight bands are restricted to training labels and separately marked diagnostic
forecasts. [Aggregate results and paired comparisons](eval/milestones/result.json).

### Does the agent's stated plan improve the ETA?

Not on this cohort. This replay compared the previous stage predictor **with and without**
the order of actions in the agent's public plan or confirmed checklist. The clock, prompts,
checkpoints, and remaining-stage labels stayed fixed. Error is in active minutes; lower is better.

| Task duration | No added plan signal | Ordered prose + checklist |
|---|---:|---:|
| 3–15 minutes | 3.19 | 3.25 |
| 15+ minutes | 12.15 | 12.15 |
| 30+ minutes | 19.26 | 19.24 |
| Medium and long weighted equally | 7.67 | 7.70 |

The paired 95% session-bootstrap interval for the balanced change was **−0.013 to +0.062
minutes** (negative favors the plan). Prose alone scored 7.68, the checklist alone 7.70, and
both sources with action counts but **without order** scored 7.69. None forecast 15+ minutes
on a sub-minute task. These are variants of the experimental stage predictor, not new widget
models. The widget's local control scored 7.74; the interval for ordered plans versus that
control also included no change.

At the 65 explicitly identified in-flight verification readings (32 prompts), the extractor
found prospective prose at **1** and a current checklist at **54**. An unconditional declaration
of substantive work after a check was right at **33 of 35** readings; for context, **57 of all 65**
readings did have another stage. Both explicit final-check declarations were followed by more
work. These are correlated readings, not 65 independent trials. Adding plan order left the
predictor's prompt-weighted final-stage precision and recall unchanged at 47% and 36%.
This is evidence against changing the ETA on this basis, not evidence that understanding
agent prose is impossible: the conservative extractor seldom saw usable prose at verification.

`bun eval/stated-plan/study.ts <frozen-milestone-directory> <new-aggregate.json>` reruns the
comparison locally. The extractor uses only public assistant text already recorded at the
checkpoint and the current prompt's successfully updated checklist. New public statements
supersede old ones, prose expires after five active minutes, and terminal replies, private
reasoning and quoted code are not plan sources. The four probability adjustments train on
out-of-fold development predictions; all are fixed before scoring the previously used 42-session
holdout. Only aggregate counts, metrics and cohort hashes leave replay.
[Aggregate results, coverage and intervals](eval/stated-plan/result.json).

### Can an earlier verification in the same session predict a rerun?

**Yes for the short calls we could measure, but not enough to change the displayed prompt ETA.**
On the same frozen 42-session holdout, 124 recognized verification calls started and 92 had
an observed end. At launch, 58 of those 92 had a comparable earlier run (37 exact command
and working-directory matches, 21 runner/action/scope matches, seven sessions total).
On those 58 calls, using earlier durations cut call-weighted mean absolute timing error
from **79 seconds to 1.9 seconds** versus a timer trained on development sessions without
session history. **All 58 finished in under a minute.** Only two recognized calls with an
observed end took a minute or longer, neither had a match, and none took ten active minutes.
This does not validate the timer for slow end-to-end tests.

| Mean absolute prompt ETA error, active minutes | Refit local control | With same-session timing |
|---|---:|---:|
| Medium (61 prompts) | 3.12 | 3.12 |
| Long (37 prompts) | 12.36 | 12.36 |
| 30+ minutes (17 prompts, 12 sessions) | 19.59 | 19.59 |
| Non-ask tool running over 60 seconds | 7.93 | 7.93 |

Of 8,027 display readings, **115** had comparable history, across only five calls.
At just **one** reading did both the population and history curves have enough calls
still running at that age to estimate the remaining time. The balanced medium/long error
changed by less than 0.001 minute. No 15+ minute miss rate changed; none of the 49 sub-minute
prompts was forecast at 15+ minutes. A separate, post-hoc check removed the support threshold:
using the last comparable duration minus age as a floor never raised the displayed local ETA.

The experiment freezes up to 16 past durations at launch and compares the actual call age
to development runs that survived the same age relative to their histories. A foreground call
adjusts only the estimated *difference* in time left; a background call may overlap other
work and can only extend the ETA. A background-job handoff is not a completed test:
we pair it with a native completion notice or wait receipt **within the prompt**. That measures
time to notification, not necessarily process runtime or a passing result. Of 41 background
starts on the holdout, nine had such a paired end. Opaque scripts, unrecognized wrappers,
unpaired jobs and completions after the prompt limit what these data can establish.

`bun eval/repeat-verification/study.ts <frozen-milestone-directory> <new-aggregate.json>`
replays locally without a provider or changing the widget. Command identities never leave
their session. [Aggregate results, coverage and paired intervals](eval/repeat-verification/result.json).

### How much of a slow check did the agent actually wait for?

We measured three real invocations of the same end-to-end check in one opted-in session.
Each invocation rebuilt and compared the full milestone, stated-plan and repeat-verification
aggregates against their references, without delays or provider calls. All three passed.
The check ran in the background; to measure blocking, the agent explicitly joined that
specific process using a synchronous tool call. The join on the first run arrived after exit.

The three checks had similar runtimes but very different effects on the agent: an immediate
join blocked for most of the check, a later join for only part of it, and a post-exit join
did not block at all. The previous duration helped predict process exit, not total agent
wait. Unobserved time is neither assumed work nor assumed waiting. These deliberately different
schedules describe one workload, not independent sessions or a whole-prompt ETA evaluation.

We separately checked the original 244 development and 42 held-out sessions. No exact
rerun followed a measured 60+ active-second foreground verification. One development call
matched the same runner/action/scope, but no such match occurred on holdout. Some waits on
held-out background checks could not be uniquely attributed. A receipt is not a process-exit
timestamp; ambiguous waits are not evidence that the agent was free.

**Collect another session locally (Linux, opt-in).** `python3 -B eval/slow-checks/capture.py init`
prints a new private directory. Use it only for that session. Then:

```sh
python3 -B eval/slow-checks/capture.py run \"$DIR\" 1 -- bun test
python3 -B eval/slow-checks/capture.py join \"$DIR\" 1
python3 -B eval/slow-checks/capture.py report \"$DIR\"
```

Dispatch `run` as a background tool call if the agent will continue; wait until it has
started before `join`. Run `join` **synchronously** when actually blocked (e.g. awaited
`sh()` in OMP `eval`, which doesn't auto-background). Optionally wrap an independent
synchronous command with `work \"$DIR\" -- …` to measure overlap. An asynchronous `join` or
`work` cannot establish what the agent was doing. Use a new number for each launch. The
collector records monotonic times, exit codes and a directory-salted command/working-directory
fingerprint in the private directory; no command text, output or transcript is saved.
It does not monitor other commands, install anything, or keep running after the calls finish.
[Historical denominators and attribution](eval/slow-checks/historical-result.json).

### Does a check’s remaining time improve the prompt ETA while the agent is waiting?

**Not in the measured cases.** We tried treating the last successful duration of the exact
same check as a floor: `max(local prompt ETA, prior check duration − check age)`, but only
while the agent was synchronously waiting for that check. At every other point the forecast
was the shipped local ETA. We also scored an ungated floor and, as a diagnostic, replacing
the prompt ETA with the check time left during an identified wait.

| Evidence | What happened |
|---|---|
| Completed controlled prompt | The gated floor did not change the prompt ETA. Substituting just the remaining check time underestimated the work after the final check. |
| Separate live synchronous join | Lock contention identified a real wait before exit. The gate left the larger prompt ETA unchanged. This validates the signal, not whole-prompt accuracy. |
| 42 held-out sessions, 176 completed prompts | 1,701 of 8,027 readings were inside generic waits. None could be tied prospectively to a uniquely identified check with a measured slow repeat. No forecast changed, including on the 17 prompts lasting 30+ minutes. This is zero coverage, not evidence the floor works on long prompts. |

We kept the estimator and widget unchanged. A check can block the agent without being
the last thing it needs to do. The missing test case is a natural long prompt where a
known blocking check has **more time left than the local prompt ETA**.

To reproduce offline, run `bun eval/wait-gate/organic.ts FROZEN_CORPUS NEW_RESULT_JSON`
against the original split. For a same-boot capture taken as above, set `PI_SESSION_FILE`
to the opted-in session and run `bun eval/wait-gate/captured.ts "$DIR" NEW_RESULT_JSON 3`
after its prompt has finished. This validates numbered dispatches and synchronous joins
against native call boundaries; late receipts are not used to attribute a forecast.
For the separate live signal check, run `bun eval/wait-gate/live-join.ts "$DIR" N NEW_RESULT_JSON`
as the agent’s **foreground, awaited** operation while check `N` is running, with the same
session variable. Only aggregate readings are written; keep captures and transcripts local.
[Completed-prompt replay code](eval/wait-gate/captured.ts) · [Live-join probe](eval/wait-gate/live-join.ts) ·
[Historical coverage and duration buckets](eval/wait-gate/organic-result.json).

## Live checks

```sh
bun scripts/live-smoke.ts   # installed omp over RPC, disposable session
scripts/tui-check.sh        # installed omp in interactive TUI (tmux), 100 and 54 columns
```

## Layout

- `src/facts.ts`: OMP events & branch replay → observed facts, confirmed todo milestones, Markov transitions, phase/blocker/next inference, O(n) sanitization
- `src/text-concepts.ts`: fixed 40-concept public software-engineering dictionary and L2 Ridge scorer (4 text channels, zero private vocabulary)
- `src/features.ts`: deterministic 150-feature extraction shared identically by live inference and training
- `src/gbdt.ts`: compact zero-dependency histogram-binned GBDT trainer and inference engine
- `src/survival-net.ts`: pure-TypeScript Exact Truncated Log-Normal Survival ResMLP trainer and closed-form conditional median inference
- `src/model-data.ts`: pre-trained v4 hybrid GBDT + SurvivalNet ensemble, quantile heads, text priors, and slippage centers (~588 KiB)
- `src/range-calibration.ts` & `src/range-data.json`: out-of-fold 6×6 residual quantile table for calibrated ranges
- `src/handoff.ts`: confirmed plan-closeout ETA cap
- `src/estimator.ts`: consistent progress, remaining-time range, confidence, and early abstention
- `src/observer.ts`: local reactive state machine, causal online calibration (session bias + centered slippage), prompt-scoped smoothing, 1 s live tick (no OMP imports)
- `src/widget.ts`: theme-aware native progress bar widget renderer (no OMP imports)
- `src/index.ts`: OMP extension wiring: events, `/progress`, mid-session branch seeding, TUI widget
- `eval/`: `extract.ts`, `grade.ts`, `build-cases.ts`, `train.ts`, `run.ts`
- `test/observer.test.ts`: deterministic unit tests (`bun test`)
