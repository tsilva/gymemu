# Breakout history-length investigation

Checked 2026-09-13 against the dataset's actual native provider, version **0.5.12**,
source commit `f06b9c81f4d717fd32782c5d877bb79a9042d266`, and dataset commit
`b8091d248295eb5135011dd9b943c75f4a7d50be`.

## What the simulator requires

Two distinct position observations are the basic lower bound for estimating velocity
during unobstructed motion. They do not guarantee an exact next rendered frame across
collisions, subpixel quantization, resets, or hidden controller state.

The native implementation is more involved than constant-speed left/right movement:

- [`update_paddle`](https://github.com/tsilva/env-BreakoutAtari2600-turbo-native/blob/f06b9c81f4d717fd32782c5d877bb79a9042d266/src/lib.rs#L596)
  first smooths the previous controller measurement with the current paddle position.
  It then updates an internal charge and measurement for the following native frame.
- A cold directional hold uses repeat values `0, 1, 2, 3, 4, 5, 60`: the seventh
  consecutive native action reaches the capped controller-charge increment of 60.
  **60 is controller-charge units, not pixels per frame.** Rendered speed also depends
  on the threshold table, smoothing, and screen limits.
- The repeat counter is not reset when a direction is released. Once `paddle_held`
  is false it can remain hidden and unchanged during neutral actions. A later direction
  can therefore respond differently after apparently identical recent images.
- [`step_native`](https://github.com/tsilva/env-BreakoutAtari2600-turbo-native/blob/f06b9c81f4d717fd32782c5d877bb79a9042d266/src/lib.rs#L436)
  uses `(tick + 2) & 3` for serving. Ball position/velocity use 16.16 fixed point;
  rendered pixels lose fractional state. Collision latches and game phase also persist.

The collected frames are two native updates apart. An eight-frame stack spans fourteen
native updates, comfortably longer than the cold acceleration ramp, but does not reveal
all the hidden variables above. There is no justified claim that a fixed short pixel stack
is a complete simulator state.

As a concrete counterexample, two valid constructed native snapshots differed only in
the hidden repeat counter (0 versus 60). After **128 identical HUD-masked captures**
under native NOOP, their paddles were both at x=80. The same native RIGHT action, repeated
for two native updates, produced x=80 versus x=83 and different RGB images. Continued
NOOP cannot reveal the frozen repeat state in these pre-serve snapshots. This demonstrates
hidden-state ambiguity in the native simulator; it is not a claim that those particular
snapshots were sampled from the dataset. Native NOOP is also distinct from the dataset's
button/serve action.

## What the recorded data shows

I checked **251,300 training transitions** from steps 0–499 of all 514 training episodes.
For each history length, the input key was the ordered sequence of exact frame IDs plus
the current executed action. Missing initial history had a distinct padding marker;
episode IDs and seeds were excluded because they are not model inputs. A group is
ambiguous when its identical input key has more than one successor frame ID.

| History frames | Ambiguous input groups | Examples in those groups | Fraction of examples |
| --- | ---: | ---: | ---: |
| 1 | 8,511 | 47,467 | 18.89% |
| 2 | 2,231 | 13,264 | 5.28% |
| 4 | 1,086 | 6,767 | 2.69% |
| 8 | 475 | 3,272 | 1.30% |
| 16 | 247 | 1,661 | 0.66% |
| 32 | 79 | 634 | 0.25% |
| 64 | 58 | 576 | 0.23% |

These are empirical ambiguity counts, not predicted model error rates, and not evidence
that unobserved inputs are unambiguous. Increasing history also decreases the number of
repeated keys available for this test. One must not read the decreasing counts as an
independent causal measurement of model quality.

**Decision:** start with eight frames and make the length configurable. It reduces observed
opening ambiguity compared with four while retaining a modest CNN input. No tested length
through 64 is an exact sufficient minimum. The requested simple frame-stack-plus-action
baseline remains an approximation; no simulator state, past-action channels, episode seed,
or clock is added to its inputs.

## Empty history

There are 30 distinct initial frame IDs among the 514 training episodes. Empty history
plus a no-action START category therefore has multiple targets too. The plain deterministic
MSE predictor can converge toward their conditional mean. Training includes every initial
frame once, with ordinary sample weighting; player startup uses this same empty-history
contract and never loads a real starting screenshot behind the scenes.

## Reproducing the audit

Read each training episode in step order. Keep a deque of the last H source frame IDs,
left-pad it with a sentinel for missing history, and append the integer decoded from
`native_action_json`. Group those keys and count groups with different
`successor_frame_id` values. Include only steps below 500 for the table above. Deduplication
was validated by RGB hashes, so equal frame IDs mean identical masked RGB images.

The native counterexample used `BreakoutVecEnv.get_state()`/`set_state()` with the
version-pinned `BTO11` layout from
[`serialize_lane`](https://github.com/tsilva/env-BreakoutAtari2600-turbo-native/blob/f06b9c81f4d717fd32782c5d877bb79a9042d266/src/lib.rs#L1169).
Both lanes began with identical valid state and an equilibrium paddle; only
`paddle_repeat` differed. This research dependency is intentionally absent from the
trainer and player: neither script executes a real simulator.

## Scheduled sampling

[Bengio et al., 2015](https://arxiv.org/abs/1506.03099) introduce a curriculum that
gradually substitutes model outputs for recorded inputs during sequence training to
address the mismatch with autoregressive inference. Gymemu's `breakout_scheduled`
recipe adapts this idea to whole RGB frames, keeping executed actions and supervised
targets recorded. It uses independent frame selection within short sequential
prefixes and detaches feedback from gradients.

This is a proposed experiment for recovering from generated-frame errors, not evidence
that the paper's results transfer to Breakout or that missing state becomes observable.
The 80% ceiling, two-epoch warmup, and six-epoch ramp are tunable starting choices.
Evaluate recorded-context prediction and generated rollouts separately; lower held-out
pixel MSE alone cannot establish improved ball survival or collision behavior.

## Gradient clipping at the horizon-8 transition, 2026-09-16

A bounded experiment started from the completed horizon-4 checkpoint of
`gymemu-ar-fast-20260916-125818`. The checkpoint contains weights but no optimizer state,
so each arm used fresh Adam at learning rate 0.001. This was a matched weight restart,
not an exact continuation of the original optimizer trajectory.

The six arms compared no clipping with global L2 caps of 1.0 and 0.1 on two matched
training-batch sequences, seeds 47 and 48. Each ran 1,000 horizon-8 updates with batch
size 32 and CUDA bf16. All arms used eager execution and the same frozen source as
the original run. Evaluation used float32 on 512 fixed held-out one-step targets and
31 fixed held-out starts with up to 32 autoregressive predictions each. This diagnostic
subset does not replace full held-out evaluation. The repeats share one trained
checkpoint; they are not independent model-training seeds.

Final rollout RGB MSE changes relative to each matched no-clipping control:

| Global norm cap | Batch seed 47 | Batch seed 48 |
| --- | ---: | ---: |
| 1.0 | +7.1% | +879.9% |
| 0.1 | +558.2% | -85.5% |

Positive changes mean worse predictions. All six final models had worse rollout MSE
than the starting checkpoint. Cap 1.0 clipped 1.5% and 28.2% of updates; cap 0.1 clipped
14.1% and 6.1%. Raw gradients were measured before clipping, and every clipped update
passed its post-clipping norm bound. Initial losses, raw gradients, and Adam update
norms matched exactly across arms sharing a batch sequence.

Clipping alone did not reliably stabilize this short weight-restart experiment.
The result does not rule out another threshold or clipping combined with a lower
learning rate, and does not establish why a clipped trajectory deteriorated.
Fresh optimizer state and the short horizon-8 training budget limit transfer to the
original run. A lower-learning-rate comparison from the same checkpoint is a next
experiment, not a validated remedy.

The [W&B report](https://wandb.ai/tsilva/gymemu-Breakout-Atari2600-v0/reports/Horizon-changes-and-gradient-instability---AR-fast--VmlldzoxNzk0NjQ4Nw==)
contains all six runs and their evaluation curves. Local artifacts are under
`runs/clip-ablation-20260916/`, including the harness, source archive, fixed sample
indices, metrics, optimizer states, and checkpoints. The source checkpoint SHA-256 is
`9a563ae1eccc435c91caf314843730fcdb7d8312f00bfc0537140a7bcbac69e9`.
The original run was stopped with explicit user approval after preserving its latest
checkpoint separately; neither original checkpoint was modified by the experiment.

## Detached feedback at the horizon-8 transition, 2026-09-16

A subsequent matched experiment changed only whether gradients crossed generated
feedback. Both arms used fully generated histories, the same per-step RGB and
ball-region losses averaged over valid steps, and horizon 8. The detached arm detached
the entire history before every forward pass; each current prediction retained its own
loss graph. No clipping was applied. This differs from the scheduled-sampling recipe,
which mixes recorded and generated history and supervises only the final prediction.

The same completed horizon-4 checkpoint, fresh Adam at 0.001, batch size 32, CUDA bf16,
eager execution, paired batch sequences 47 and 48, and 1,000-update budget were used.
Evaluation reused the clipping study's fixed held-out targets and float32 metrics.
These are two batch-sequence repeats from one checkpoint, not independent training seeds.

Before any updates, four training batches were tested at horizons 1, 2, 4 and 8 in bf16
and float32. Predicted frames and losses were exactly identical between full and detached
feedback within each precision. Horizon-1 gradients matched exactly. The full-loss
harness also matched native loss and gradients on the first batch at every horizon.
At horizon 8, full/detached gradient-norm ratios ranged from 2.64–22.60 in bf16 and
3.36–35.11 in float32. This isolates gradient amplification through feedback on those
batches; it does not show that every instability comes from that mechanism.

| Batch seed | Feedback gradients | Final rollout RGB MSE | Final one-step MSE | Peak gradient norm |
| --- | --- | ---: | ---: | ---: |
| 47 | Full | 0.00334863 | 0.00074026 | 10.4588 |
| 47 | Detached | 0.00198216 | 0.00014035 | 0.01335 |
| 48 | Full | 0.01726577 | 0.00895711 | 356.3761 |
| 48 | Detached | 0.00198282 | 0.00013461 | 0.01391 |

Detached feedback reduced final rollout MSE by 40.8% and 88.5% versus its matched
controls. Both detached models improved about 9.5% over the starting rollout MSE
of 0.00219142. Ball-region rollout MSE also improved, to about 0.00821 in both repeats.
The result supports detached feedback for a longer follow-up experiment; it does not
establish long-run stability, full-held-out superiority, or playable ball dynamics.
Cutting feedback gradients also removes credit assignment from later losses to earlier
predictions. Fresh optimizer state remains a limitation of this weight-restart study.

The existing W&B report includes the four runs. Artifacts are preserved under
`runs/detach-ablation-20260916/`: the harness, frozen-gradient comparisons, sample indices,
runtime provenance, metrics, checkpoints and optimizer states. The production training
implementation and direct baseline were not changed.

## Detached execution recipe qualification, 2026-09-16

The detached objective is now available as `recipe=breakout_detached`, with full
generated history and an explicit saved `detach_feedback` setting. Its full-run
curriculum is 1, 2, 4, then 8 steps over ten epochs. Existing autoregressive recipes
retain full feedback gradients, and the direct reference remains unchanged.

`recipe=breakout_detached_fast` selects batch 64, automatic compiler layout optimization,
and factored action inputs. A Beast-3 RTX 4090 benchmark measured 1,795 windows/s at
horizon 8, versus 1,251 for the compiled batch-32 reference. All seven long benchmark
variants used the same sample order and retained health/probe instrumentation. See
[performance details](performance.md#detached-rollout-training) for the full matrix,
numerical differences, memory use, and timing exclusions.

A matched continuation check processed 32,000 training windows per arm from the same
horizon-4 checkpoint with fresh Adam. Across two batch sequences, fast rollout MSE was
1.5% lower and 2.4% higher than eager detached training; peak gradient norms stayed below
0.005 in all four arms. These are short stability checks, not evidence of long-run
quality equivalence. The compiled CUDA train/save/reload/playback smoke passed all four
curriculum horizons. The full ten-epoch run was prepared but not started during this
qualification. Artifacts and the staged-source launch manifest are under
`runs/detached-throughput-20260916/`.
