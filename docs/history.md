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

## Paddle uncertainty in the H4/R4 checkpoint, 2026-09-17

The frozen local `gymemu-h4r4-20260917-072430/latest.pt` has SHA-256
`117aa9cc42d952aca3543af13cc7bfc2a0ae86e4ea5f9ab0d8835e074ea2b279`
and epoch-9 metadata. This investigation used that exact file, not the changing
remote `latest.pt`. Dataset revision was
`676ff6388f4218d3c3a3ce9f2f33e075fa7314a3`.

**Finding:** conflicting next-paddle targets exist for identical complete model
inputs, including all four executed actions. Hidden controller state provides a
source-level mechanism for this ambiguity. The checkpoint produces softened edges
close to an empirical target average on a verified conflicting-input group. Model
approximation adds residual error; this does not attribute every faint pixel to
missing information or estimate its fraction of full-run error.

### Exact screenshot reproduction

Episode 5, target position 140, reproduces the screenshot's full-frame MSE:
`4.288699e-6` on MPS and `4.288702e-6` on CPU. The four actions are all LEFT.
The recorded paddle centers are 137.5, 134.5, 125.5, 115.5, then target 105.5.
The predicted center is 105.2905, about 0.21 pixels too far left. The faint red
outside the target covers about 0.297 equivalent fully bright columns. These
centers and intensities use the paddle-height red-channel profile, excluding walls.

The displayed mode is teacher forcing: prediction uses recorded input history.
Changing only the newest action moves the predicted center to 107.4954 for button,
110.4994 for RIGHT, and 105.2905 for LEFT. Thus this checkpoint uses the action;
absence of an immediate direction reversal is not proof of failed conditioning.
The native controller itself has lag. CPU reproduces MPS's faint edge, so it is not
an MPS-specific playback artifact.

In the bounded training scan below, this screenshot input had only one matching
training example, with the same target. No conflicting target was established for
this exact input. Its residual therefore cannot be uniquely assigned to ambiguity.

### Conflicting inputs in the actual training data

Scan target positions 4 through 500 inclusive, or the episode end, across training
episodes: **916,367 transitions**, **581,784 distinct H4+A4 inputs**. Keys contain
four ordered frame IDs and four executed actions. There were **3,820** keys with
different successor RGB images, of which **3,797** had different paddle bounds.
Paddle bounds use the longest red component of at least four pixels on row 190,
inside columns 12–149, excluding side walls. Counts describe this opening-prefix
sample, not all training data or the percentage of model errors caused by ambiguity.

For a concrete example, episodes 13 and 19 at target position 87 have exactly the
same four input frame IDs `[1065,1077,1089,1101]` and actions `[0,0,0,2]`, meaning
button, button, button, LEFT. The decoded input tensors and action tensors were
explicitly checked equal. Across 100 matching training examples in the scan:

| Target | Occurrences | Paddle center |
| --- | ---: | ---: |
| Frame 1115 | 44 | 62.5 |
| Frame 47613 | 56 | 63.5 |

No deterministic function of those inputs can match both targets. Their empirical
mean has edge intensities 0.44 and 0.56 relative to full paddle brightness. The
checkpoint gives about 0.59 and 0.45, retaining a bright common interior and faint
edges. Its paddle-strip MSE to that empirical mean is `8.59e-5`; variance among the
actual target strips is `9.22e-4`. It approximates, but does not exactly equal, the
mean. These 100 examples need not be independent draws, and their frequencies are
not calibrated future probabilities. A second tensor-verified group has 21 examples
with paddle centers 92.5 and 94.5.

Adding the preceding four frames and four actions to those same examples resolves
**2,680 of the 3,797** conflicting H4 groups into H8 groups with unique observed
paddle bounds. **1,117** H4 groups retain at least one conflicting H8 subgroup.
This supports extra history as a useful input ablation, not proof of full
observability or a causal measurement of the older checkpoint's advantage.

### Mechanism and alternative explanations

The current dataset manifest specifies provider **0.5.12**, frame skip **2**,
sticky-action probability **0**, and no frame max-pooling. The
[pinned native implementation](https://github.com/tsilva/env-BreakoutAtari2600-turbo-native/blob/f06b9c81f4d717fd32782c5d877bb79a9042d266/src/lib.rs#L570)
first smooths paddle position toward the previous controller measurement, then
changes hidden charge using the action and repeat counter. Quantized thresholds
produce the following measurement. Charge, measurement, repeat and held state are
not represented directly by the RGB/action input. Recorded info fields expose
paddle position/width but omit those controller variables. Similar visible histories
can therefore conceal different next-step dynamics. These are native dynamics,
not ALE sticky actions or an unspecified stochastic action override.

For squared error, the optimal deterministic output is the conditional mean.
When possible targets differ by paddle position, the mean has faint edges.
[Mathieu et al.](https://arxiv.org/abs/1511.05440) study this general MSE video-blur
problem. The dataset collisions and checkpoint output above supply the evidence
specific to Gymemu. Deterministic sigmoid RGB values are not explicit confidence
estimates.

A supplemental probe used 1,024 randomly selected transitions across 32 held-out
episodes plus 81 consecutive screenshot-neighborhood transitions. Estimated center
MAE was 0.199 pixels and the 95th percentile 0.775. Reversal MAE was 0.134, versus
0.266 for repeated directional actions. Altering only the latest action changed
RIGHT-minus-LEFT predictions by 4.69 pixels on average. These small, non-independent
strata do not establish population rates or correct counterfactual physics, but do
not support ignored actions or reversal-specific failure as the leading explanation.
The red-profile center estimator clips to the interior strip and can be affected by
nearby red ball pixels; exact paired-input evidence does not rely on these estimates.

The ball-region objective gives the ball extra weight without analogous paddle
weight. That can affect residual fit, but increased paddle weight cannot resolve
contradictory targets for identical inputs. No loss-weight or retraining ablation
was performed, so its causal contribution remains unmeasured.

A focused next experiment is H8 context with R4 rollouts, keeping the rest fixed.
Persistent recurrent state or explicitly recorded controller state would address
the missing-state mechanism more directly. Predicting several possible futures can
represent ambiguity but still requires coherent state across rollout steps.

### Reproduction artifacts

Scripts, numerical receipts, frozen-input arrays and a labeled evidence plot are in
`logs/paddle-uncertainty-20260917/` (ignored diagnostic artifacts). The production
trainer, checkpoint and running experiment were not changed.

```bash
PYTHONPATH=. uv run python logs/paddle-uncertainty-20260917/prepare.py
PYTHONPATH=. uv run python logs/paddle-uncertainty-20260917/repro.py mps
PYTHONPATH=. uv run python logs/paddle-uncertainty-20260917/repro.py cpu
PYTHONPATH=. uv run python logs/paddle-uncertainty-20260917/measure.py
PYTHONPATH=. uv run python logs/paddle-uncertainty-20260917/aliases.py
PYTHONPATH=. uv run python logs/paddle-uncertainty-20260917/pairs.py
uv run python logs/paddle-uncertainty-20260917/plot.py
```

Both `repro.py` calls intentionally fail their diagnostic guard: outside-target
red mass exceeds 0.25 equivalent paddle columns. This threshold detects the supplied
symptom; it is not an established scientific quality threshold. Other scripts
complete and write JSON/NPZ evidence. `pairs.py` asserts byte-for-byte equality of
all paired image/action inputs. `measure.py` reuses its cached predictions when
present; remove that generated NPZ before testing a different checkpoint.

### Extending the history-length check

A follow-up retained the same 916,367 opening-prefix transitions and extended both
RGB and executed-action histories. Counts below are original H4 groups that still
contain at least one conflicting longer-input subgroup, not counts of all examples.

| RGB frames and actions | Original H4 groups still conflicting |
| ---: | ---: |
| 4 | 3,797 |
| 8 | 1,117 |
| 16 | 194 |
| 32 | 9 |
| 64 | 0 |
| 128 | 0 |
| 256 | 0 |
| Entire observed prefix, through target position 500 | 0 |

Thus 64 is the shortest **tested** length with no observed paddle conflicts in this
sample; intermediate lengths 33–63 were not tested. This does not prove universal
sufficiency, identify whether extra image or action history mattered, or guarantee
that a trained CNN will extract the relevant state. Longer histories distinguish
more recorded trajectories and reduce repeated inputs, so zero observed collisions
must not be treated as proof of a fully observable representation. The full-prefix
check also found no empirical counterexample to sufficiency since episode reset in
this bounded sample; the source-level hidden-state constructions above do not
establish that any such pair occurred here.

Reproduce with
`PYTHONPATH=. uv run python logs/paddle-uncertainty-20260917/history_lengths.py`.
The script extends only H4-conflicting groups, which is sufficient because a pair
with identical longer input necessarily has identical last four inputs. Receipts
are in `logs/paddle-uncertainty-20260917/history-lengths.json`.

## 2026-09-17: Separating visual representation from dynamics

Research note; no implementation change or new experiment.

[World Models](https://worldmodels.github.io/) trains a per-frame VAE to
reconstruct its input, then an action-conditioned recurrent model to predict the
next latent from current latent and memory. [IRIS](https://arxiv.org/html/2209.00588v2)
uses a single-image discrete autoencoder and a separate Transformer over frame
and action tokens on Atari. Its appendix also shows that excessive compression
can erase important sprites. These establish useful precedents, not evidence that
separation improves Gymemu.

[DreamerV3](https://arxiv.org/html/2301.04104v2) offers a different precedent:
its posterior combines the current observation with recurrent history, reconstructs
the current observation, and is trained jointly with a predictive prior using
reconstruction and KL objectives. It is not a frozen history-to-current-frame
autoencoder trained only for reconstruction.

Recommendation (inference): begin with the existing `approach=latent` split:
reconstruct individual frames, then predict their latents from encoded history
and actions. This separates visible appearance from temporal state estimation.
A history encoder trained only to reconstruct the newest frame can ignore history;
that objective alone does not require preserving velocity or hidden controller state.
Predicting the next frame from history should condition on the executed action;
it already trains dynamics, and can instead serve as an auxiliary objective when
jointly refining the representation. One-step prediction alone need not preserve
information needed later.

Check held-out codec reconstructions for ball and paddle fidelity before blaming
dynamics, then compare the same held-out next-frame RGB MSE and bounded recursive
rollouts. Keep `approach=direct` unchanged. Longer context or persistent memory may
address the missing-state evidence above; neither guarantees full observability.
## 2026-09-17: Deterministic brick-grid feasibility

`prototype_brick_grid.py` calibrated six color bands and a fixed 18-column grid
from one training frame in dataset revision
`676ff6388f4218d3c3a3ce9f2f33e075fa7314a3`. Bricks occupy 8-by-6 pixels at
x=[8,152), y=[57,93). The ball can adopt the brick row's color, so small matching
patches fitting within 2-by-4 pixels must not become bricks. An initial
conservative rule left these patches ambiguous; training examples established
the geometric exclusion. The rule was frozen before evaluating a fresh held-out
offset sample.

All 10,624 fresh held-out transitions matched recorded counts, including 286
single-brick destruction pairs. Of 42,496 training transitions, 42,178 matched;
the 318 mismatches occurred at episode steps 0 through 16, during startup wall
drawing despite an internal count of 108. Their visible grids are retained but
internal-state labels are masked unknown. All 1,097 sampled training single-brick
destruction pairs matched. Receipts and selected raw-RGB/matrix images are in
`logs/brick-grid-prototype/final/`.

Count agreement does not certify individual cells; synthetic cell-index tests,
visual inspection, and temporal removal checks add evidence. Sampling used
consecutive shard slices and does not estimate independent full-dataset accuracy.
No later-wall reset appeared in this sample. The result supports deterministic
offline augmentation with rejection masks; it does not establish a complete
rendering state.
## 2026-09-17: Complete brick-grid dataset augmentation

Published additive annotations to
[`tsilva/gradlab-breakout-trajectories` revision
`9a22e4c0b6b9796a1358f36a6854f2569cbee0af`](https://huggingface.co/datasets/tsilva/gradlab-breakout-trajectories/tree/9a22e4c0b6b9796a1358f36a6854f2569cbee0af).
The source was `676ff6388f4218d3c3a3ce9f2f33e075fa7314a3`.

The vectorized detector processed all 5,924,824 unique images. New columns cover
all 6,728,910 transitions and 2,320 episode-initial states. All original columns
were checked after Parquet roundtrip; image assets and split identities were
preserved. The build uses simulator counts only for quality checks, not to alter
the visible matrix.

There are 21,706 startup-animation successor states: 17,393 train and 4,313
held-out. All 2,320 initial frames also belong to the startup prefix, giving
24,026 of 6,731,230 captured states, or 0.3569%. Startup sometimes removes visible
bricks before the full wall appears. The prefix flag ends at the first complete
wall or gameplay evidence rather than assuming monotonic wall growth. In this
dataset its latest flagged transition step is 16.

Every one of the other 6,707,204 successor states matched its native brick count.
No unknown cells, unexpected colors, partial bricks, unexplained reappearances,
or later-wall resets were found. All native wall-clear counters are zero, so
later-wall behavior remains unvalidated. Every initial state's native count is
unavailable and flagged separately. These are complete count/consistency checks,
not independent native per-cell ground truth.

All 512 randomly selected real-image comparisons matched the original scalar
prototype. Tests covered decreasing startup animation, later-wall classification,
unexplained same-count cell swaps, noncontiguous frame IDs, and additive schema
roundtrips. The full suite passed 294 tests with two skips; Ruff passed. Receipts
are in `logs/brick-grid-augmentation-20260917-v1/`, and public code/provenance lives
under the dataset's `annotations/breakout-bricks-v1/`. Existing training revision
pins and filters were not changed.


## 2026-09-17: Trained paddle state history probes

An adaptive CPU study trained standalone RGB-derived MLPs on recorded revision
`676ff6388f4218d3c3a3ce9f2f33e075fa7314a3`. It used 128 training episodes,
32 separate training-split validation episodes, and 64 official held-out episodes.
All final comparisons share 182,746 test targets. Recorded native velocity is
supervised directly, not approximated by differences between two-tick captures.

For next-state prediction, one frame plus four executed actions was the smallest
visual context meeting the prespecified practical target: both MAEs <=0.25 native
units and >=95% jointly within one unit. Two frames plus two actions also met the
point-estimate criterion, with a weaker margin. Actions include the action executing
the predicted transition. Position is the paddle's left edge.

| Frames | Actions | Position MAE, pixels | Velocity MAE, pixels/native tick | Joint within 1 |
|---:|---:|---:|---:|---:|
| 1 | 2 | 0.647 | 0.342 | 78.65% |
| 1 | 3 | 0.297 | 0.206 | 94.70% |
| 1 | 4 | 0.225 | 0.157 | 96.77% |
| 1 | 8 | 0.228 | 0.160 | 96.91% |
| 2 | 1 | 0.880 | 0.379 | 77.10% |
| 2 | 2 | 0.242 | 0.170 | 96.40% |
| 2 | 4 | 0.193 | 0.139 | 97.51% |
| 4 | 2 | 0.196 | 0.147 | 97.44% |

A second seed reproduced the short-context results. F1/A4's episode-bootstrap
95% position-MAE interval is [0.211,0.238]; F2/A2's is [0.229,0.255], crossing
the target. Only 81% of F1/A4 predictions recover both rounded labels exactly.
These are measured practical minima, not exact sufficient histories or rollout
validation. Other architectures, thresholds, or distributions can change the result.

The fixed RGB front end reads the visible paddle at y=189:193, x=8:152 and
excludes a fixed red edge decoration. Accepted positions matched recorded labels
exactly; all rare unknown detections remain in evaluation. Final MLPs one-hot encode
integer position. This improved short-context fitting more than adding long scalar
histories. The probe consumes image-derived features, never recorded state inputs;
it does not establish how much context a full-image CNN will learn to use.

For current-state estimation, two observed frames plus their intervening action
achieved 0.006 position MAE and 0.166 native velocity MAE, with 99.92% jointly
within one unit. One frame plus four preceding actions is another passing option.
One frame plus three actions narrowly failed the current-velocity test threshold.

Reproduce with `probe_paddle_history.py`; see the training guide. Local checkpoints,
learning curves, frozen test plan, predictions, bootstrap intervals, inference smoke,
and the full report are in `logs/paddle-history-20260917/`. Production emulator
approaches and their RGB comparison metrics were unchanged.

## 2026-09-17: Supervised state and latent dynamics proposal

This is a proposed experiment, not a measured improvement to Gymemu.

[PlaNet, Hafner et al., 2019](https://planetrl.github.io/) separates an encoder
that infers state from observed images and actions, an action-conditioned state
transition, and an image decoder. It jointly trains the model with reconstruction
and latent consistency terms. Imagined trajectories advance directly in state
space; the decoder is unnecessary for planning. This demonstrates that joint
training and state-only rollout are practical, but does not establish the best
architecture or training schedule for this Breakout dataset.

PlaNet also explains why fitting one-step transitions with finite model capacity
does not necessarily optimize multi-step prediction. Its latent overshooting
objective trains multi-step predictions without decoding every prediction to
pixels. Its recurrent state carries information beyond the current image, while
stochastic state represents uncertainty. These are relevant options when recent
images and actions cannot identify the exact simulator state.

For this project, an explicit proposal is `state = encoder(observed history)`,
`next_state = transition(state, action)`, and `frame = decoder(state)`. During a
rollout, advance the predicted state and render it for display. By construction,
rendering errors then do not enter the next transition through re-encoding.
Initialization errors and transition errors still propagate. This is a causal
property of the proposed computation graph, not an empirical accuracy claim.

To preserve readable variables, supervise the exact coordinates consumed by the
decoder and transition with named targets such as position, velocity, brick
occupancy, and game phase. A separate prediction head attached to an unrestricted
latent does not give every latent coordinate those meanings. Reconstruction alone
does not identify semantic coordinates without assumptions or supervision;
[Locatello et al., 2019](https://proceedings.mlr.press/v97/locatello19a.html)
establish this limitation for unsupervised disentanglement.

A practical first comparison would train the encoder on annotated current state,
the decoder on annotated state and its matching image, and the transition on
consecutive annotated states and their intervening action. Evaluate the assembled
pipeline on predicted states too. Joint fine-tuning can combine normalized state,
reconstruction, and multi-step transition losses, with their weights selected on
validation episodes. Soft state supervision anchors meanings but does not prevent
small errors from carrying extra information. Missing persistent variables may
require added targets or explicitly separate recurrent memory. Exact simulator
state cannot be assumed recoverable from a fixed history; see the counterexample
and qualified history-probe results above.

## 2026-09-17: Full RGB CNN paddle history experiment

Full recorded RGB can support the short paddle histories found by the feature
probe, using the exact frame-autoencoder CNN encoder. The successful version adds
a learned position readout with auxiliary supervision; the plain CNN regression
head did not reach the same accuracy in this experiment.

Both models consume complete 210×160 RGB frames, divided by 255 and bottom-padded
to 216 as in `FrameCodec`. They share its three convolutions, channels 3→32→64→32,
kernel 4, stride 2, padding 1, with ReLU after the first two. Encoder weights are
shared across historical images. The experiment starts from scratch, then resumes
only its own checkpoints. Existing autoencoder weights were excluded because
their training episodes overlap this probe's validation episodes.

Targets are the original recorded `paddle_x_normalized` and
`paddle_vx_normalized`, read directly from revision
`676ff6388f4218d3c3a3ce9f2f33e075fa7314a3`. Inputs end at image t and action t;
targets describe state t+1. Labels are never predictor inputs. Reported native
units multiply normalized values by 160; velocity is pixels per native tick.

The study reuses the earlier episode split and target identities: 128 training
episodes, 32 validation episodes, and 64 official held-out episodes. It uses
65,536 training examples and 32,768 validation examples; final evaluation covers
all 182,746 eligible targets in the selected held-out episodes. This is less
training data than the original all-target feature study. An additional feature
control uses the same 65,536 training examples.

The adaptive sequence was: a 20-epoch, 32,768-example RGB spike; an 80-epoch,
65,536-example comparison of F1/A4, F2/A2, and F2/A4; 12 epochs of float32
refinement; then an explicit readout change. The last stage supervises a
160-class observed-position head with recorded labels for the input frame, trains
it jointly with next-state prediction for 24 epochs, and freezes learned perception.
Independent temporal/action heads then train on cached CNN probabilities for 80
epochs each. Each head takes approximately three seconds. Final evaluation runs
the full RGB models, with float32 and TF32 disabled. Training phases total about
24.6 minutes on one RTX 4090, excluding loading and evaluation.

The learned position classifier is 99.985% accurate on validation input frames.
No crop, color threshold, hand-written paddle detector, or recorded state is used
at inference. The auxiliary objective and position-probability bottleneck are
material parts of the successful model, not claims about an unmodified regressor.

| RGB frames | Actions | Test x MAE (px) | Test velocity MAE (px/tick) | Both errors ≤1 |
|---:|---:|---:|---:|---:|
| 1 | 3 | 0.307 | 0.217 | 94.43% |
| 1 | 4 | 0.236 | 0.171 | 96.35% |
| 1 | 8 | 0.267 | 0.199 | 96.26% |
| 2 | 1 | 0.864 | 0.370 | 78.21% |
| 2 | 2 | 0.237 | 0.173 | 95.75% |
| 2 | 4 | 0.200 | 0.149 | 96.63% |

Using the preset criterion (both MAEs ≤0.25 and at least 95% jointly within one),
F1/A4 and F2/A2 are the smallest passing histories among the tested configurations.
F1/A4 needs one CNN pass; F2/A2 trades another observed image for fewer actions.
F2/A4 offers the best accuracy tested. Eight actions did not improve the trained
one-frame model; this is an optimization result, not evidence that additional
information is harmful. Zero/one-action controls fail, as does F1/A2.

Episode-bootstrap 95% intervals for position MAE are [0.221,0.2504] for F1/A4,
[0.221,0.2514] for F2/A2, and [0.185,0.213] for F2/A4. The minimal candidates
pass on point estimates but sit close to the cutoff; F2/A4 has more margin.
The plain regression heads tested at F1/A4, F2/A2, and F2/A4 have held-out position
MAEs of 0.387, 0.456, and 0.389 respectively. Thus the successful RGB result
depends on the supervised visual readout and its training schedule.

These are one-step results from a single model seed and a selected subset of
held-out episodes. They do not establish a global minimum, exact recoverability,
or closed-loop rollout quality. Standalone checkpoints, complete comparisons,
episode-bootstrap intervals, source archives, launch receipts, and the report are
in `logs/paddle-rgb-20260917/`; see the full RGB probe in the training guide.

## 2026-09-17: Direct current paddle state from RGB and past actions

The preceding full-RGB study predicted the successor state. That did not match
the requested current-state estimation task. This corrected experiment uses
images through state t and executed actions through t−1 to directly regress
`paddle_x_normalized`, `paddle_vx_normalized`, and `paddle_width_normalized` at t.
There is no next-state loss, transition model, auxiliary classification objective,
position-probability bottleneck, or hand-written extraction in this model.

The same shared `FrameCodec` CNN processes each complete RGB image. Per-image
features and one-hot past actions feed an MLP with three continuous outputs.
All weights train jointly against the recorded normalized labels. Position and
velocity use native scale 160; width uses 16, verified against recorded native
width. Both recorded widths, 12 and 16 pixels, occur in train and held-out data.
The normalized velocity remains signed.

The experiment reuses the disjoint 128/32/64 train/validation/held-out episode
split and immutable dataset revision from the preceding study. Five 20-epoch
spikes compare F1/A0, F1/A4, F2/A0, F2/A1, and F2/A4 on 32,768 training examples.
F1/A4 and F2/A1 continue for 32 epochs on 65,536 examples and eight epochs in
float32. Checkpoints are selected by validation native MSE, with both refinement
phases evaluated in the same float32 precision before testing. Training totals
8.56 minutes on one RTX 4090, excluding loading and evaluation.

All 182,746 eligible targets in the 64 held-out episodes use the same identities.
Selected direct models achieve:

| Frames | Past actions | Current x MAE (px) | Current velocity MAE (px/tick) | Current width MAE (px) | All errors ≤1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 0.0752 | 0.3166 | 0.0602 | 98.41% |
| 2 | 1 | 0.0552 | 0.2148 | 0.0426 | 99.65% |

For F2/A1, the normalized MAEs are [0.0003449, 0.0013424, 0.0026602]. Both width
groups separately have all three native MAEs below 0.25. This direct regressor
passes the preset criterion of each native MAE ≤0.25 and at least 95% jointly
within one native unit. F1/A4 misses the velocity criterion in this run.

Use two observed frames and their intervening action as the compact working
configuration from this experiment. This is a measured result for the trained
models, not proof that a one-frame model with more optimization or a different
action history cannot succeed. Position and width are visible in the current
image; native velocity requires information about motion and may remain ambiguous
for a finite history. These are state-estimation results, not rollout tests.

Code, normalized outputs, native and width-stratified metrics, checkpoints,
source archives, launch receipts, and the report are under
`logs/paddle-current-rgb-20260917/`. The selected F2/A1 checkpoint is
`current-f2-a1-s47-n512-e8-fp32.pt`. The default direct RGB emulator is unchanged.

## 2026-09-17: Joint current paddle and ball state

The direct current-state regressor was extended to seven outputs using the
existing normalized paddle x, vx, width and ball x, y, vx, vy dataset fields.
No labels were extracted from images or derived from frame displacement. The
same autoencoder CNN processes two full RGB frames ending at t; one executed
action t−1 completes the input. All seven targets describe state t.

The data revision and disjoint 128/32/64 episode split were preserved. Two
24-epoch spikes compared fresh initialization with expanding the paddle-only
checkpoint. Fresh initialization was better. Two 32-epoch continuations
compared native-unit and inverse-training-standard-deviation loss weights.
The latter improved ball velocity learning. A final 24-epoch continuation used
float32, native-unit Huber loss, and worst-per-output validation MAE selection.
These final changes were combined, so their individual effects are not isolated.
Training used at most 65,536 examples, one seed, and 10.63 GPU minutes total,
excluding loading and evaluation. No larger context was tested in this extension.

Before testing, the selection rule was fixed to lowest worst per-output native
MAE on the same 32,768 validation targets in float32. The selected final model
achieved the following on all 182,746 eligible states in 64 held-out episodes:

| Recorded output | Normalized MAE | Native MAE |
|---|---:|---:|
| Paddle x | 0.002255 | 0.3608 px |
| Paddle vx | 0.001935 | 0.3096 px/native tick |
| Paddle width | 0.008136 | 0.1302 px |
| Ball x | 0.004453 | 0.7126 px |
| Ball y | 0.003196 | 0.8150 px |
| Ball vx | 0.130204 | 0.2604 px/native tick |
| Ball vy | 0.092038 | 0.3106 px/native tick |

All seven errors are within one native unit on 77.29% of states. This fails the
previous criterion of every native MAE ≤0.25 and joint within-one accuracy ≥95%.
Only width meets the per-variable MAE threshold. On exactly the same targets,
the earlier paddle-only model had x/vx/width MAEs of 0.0552/0.2148/0.0426, so
joint training also reduced paddle precision in these runs.

Native scales are [160, 160, 16, 160, 255, 2, 3.375]. These scales explain why
small coordinate errors in normalized units do not imply similarly small ball
velocity errors. All normalized labels and outputs retain their original
dataset representation; weighting only changes the training loss.

The compact setup learns useful estimates, but these experiments did not reach
uniformly low error across all seven outputs. This is not evidence that the
architecture cannot learn the task or that longer history is required. A
diagnostic of an intermediate model found substantial training-set errors too.
The minimum history for accurate joint estimation remains undetermined.

Checkpoints, normalized/native predictions, episode-bootstrap intervals, error
slices, immutable source archives, launch receipts, and the full report are in
`logs/paddle-ball-current-20260917/`. The selected checkpoint is
`current-f2-a1-s47-n512-e24-final-fp32.pt`. The direct emulator is unchanged.


## 2026-09-17: Recovering serve phase from recorded reset seeds

Audited annotated dataset revision `9a22e4c0b6b9796a1358f36a6854f2569cbee0af`
against the native reset/serve rules inspected at commit
`f06b9c81f4d717fd32782c5d877bb79a9042d266`. This is a data/source audit,
not a trained dynamics result or an implementation change.

The recorded manifest uses `noop_reset_max=30`, frame skip 2, and no FIRE reset.
Episode tables preserve reset seeds. The pinned Python reset implementation creates
`np.random.default_rng(seed)` and draws `integers(1, 31, dtype=np.uint64)` for the
reset no-op count. The native reset advances one tick per no-op. For source state
at transition index `t`, the source clock is therefore `reset_noops + 2*t` before
any episode boundary. Serve mode is `(source_tick + 2) % 4`; tick does not restart
after a life loss. No future state labels are used to calculate this phase.

The manifest's info filter omits `tick` and `serve_phase`. Per-transition
`elapsed_native_frames` is null for every audited serve. The phase reconstruction
uses the pinned fixed-cadence implementation and recorded reset metadata rather
than treating those missing values as measured durations.

All 830 transition shards and 2,320 episode records were inspected for explicit
`action_override_rule_id=auto_serve` transitions:

| Split | Auto-serves | Matching mode, x integer, vx and vy |
| --- | ---: | ---: |
| Train | 4,517 | 4,517 |
| Held-out | 1,150 | 1,150 |

The 5,667 cases include 2,320 initial serves and 3,347 later respawns. All four
modes occur. Every recorded executed serve action is FIRE; all three requested
policy actions occur among override cases. Every case has configured frame skip 2,
and its record seed agrees with episode metadata.

A second pass retrieved the preceding recorded state for all 3,347 later serves;
every source has `ball_y=0`. Native serving preserves the source ball x fraction.
Including that fraction, all 5,667 recorded raw fixed-point successor x values
match exactly; all recorded successor y values are 114. After the two native
frames, integer x values by mode are `[17, 77, 81, 141]`, horizontal velocities are
`[+1, -1, +1, -1]` pixels/native frame, and vertical velocity is +1. The native
spawn itself occurs at integer x `[16, 78, 80, 142]` before the remaining tick.
729 audited successors have nonzero fractional x.

Scope: this checks explicit override transitions, not unflagged loss/re-serve
inside one repeated action. It does not establish sufficiency of the other state
variables for collisions, paddle dynamics, or arbitrary new provider settings.

Recommendation: supply a four-category persistent clock phase to learned dynamics.
For dataset scenes, initialize it from episode seed and frame position, save it
with the scene, and advance it by the known native-frame duration. Do not reset it
on ball loss or replace it with the provider's -1 sentinel during active play.
The provider's optional `serve_phase` field is masked outside waiting states, so
recording `tick` or an always-valid phase is preferable for future datasets.

If a fresh start lacks phase metadata, sampling one initial phase and retaining
its evolution preserves timing correlations better than independently sampling a
new serve mode at every respawn. Exact replay of an arbitrary visual history may
still require phase inference or external provenance. A recurrent phase estimator
is an alternative, but history need not uniquely determine the initial phase.

Scripts and JSON receipts are in `logs/serve-phase-audit-20260917/`.
Primary code: [native serve rule](https://github.com/tsilva/env-BreakoutAtari2600-turbo-native/blob/f06b9c81f4d717fd32782c5d877bb79a9042d266/src/lib.rs#L450)
and [reset RNG](https://github.com/tsilva/env-BreakoutAtari2600-turbo-native/blob/f06b9c81f4d717fd32782c5d877bb79a9042d266/python/env_breakoutatari2600_turbo_native/env.py#L922).

## 2026-09-17 Single-ball state dynamics

The state-only experiment predicts seven normalized ball/paddle variables, 108
brick cells, and a terminal flag from semantic state and requested policy actions.
Predicted state supplies subsequent context. Rendering does not enter the feedback
loop. Every ball loss ends a virtual episode; lives identify loss boundaries but
are not model inputs or outputs. Source records remain unchanged. See
[the training contract](training.md#single-ball-state-dynamics).

The annotated snapshot is revision `9a22e4c0b6b9796a1358f36a6854f2569cbee0af`.
The bounded cache uses disjoint original episodes, split seed 917:

| Split | Original episodes | Virtual segments | Transitions | Terminal transitions |
| --- | ---: | ---: | ---: | ---: |
| Train | 128 | 348 | 359,313 | 227 |
| Validation | 32 | 79 | 88,638 | 48 |
| Reserved test | 64 | 171 | 182,079 | 111 |

Training and validation episodes come from the original training split. Test
episodes come from the original held-out split. Quality gaps and recording ends
are censored boundaries, not death labels. Of 359,086 nonterminal training
transitions, 9,655 change brick occupancy and 28,808 change recorded ball velocity.

### Recorded state is still incomplete

An exact-input audit used only nonterminal training examples. Keys contain every
float32 state value, explicit history-validity masks, and the requested actions,
including the current action. Repeated examples are compared with the first
successor seen for that input. Counts below are examples, not unique conflicting
groups or estimates of the fraction of prediction error caused by ambiguity.

| States | Past actions | Repeated inputs | Different successors |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 10,698 | 999 |
| 1 | 1 | 10,525 | 957 |
| 1 | 7 | 6,280 | 463 |
| 2 | 1 | 8,965 | 575 |
| 2 | 3 | 8,685 | 527 |
| 4 | 7 | 5,605 | 190 |
| 8 | 7 | 3,813 | 93 |

For a tensor-verified example, original episodes 811 and 139 at source step 45
have identical eight-state histories, full validity masks, and action inputs
`[0, 0, 0, 1, 0, 1, 1, 1]`. The last entry is the current requested action.
Their successor paddle positions are 16 and 17 native pixels, and successor
paddle velocities are 4 and 5 in the label's native units. Their ball successors
match. Across the eight-state audit, maximum conflicting paddle positions differ
by about seven pixels. No conflicting brick successors were found in these groups.

This establishes ambiguity for these model inputs, even after removing serving.
It does not establish that all prediction errors are irreducible, or that longer
history cannot help. The earlier native-controller audit documents hidden charge,
measurement, repeat, and held state as relevant omitted variables. A short-history
model winning a bounded fit comparison would not establish complete observability.
The receipt is `logs/state-dynamics-training-audit-20260917.json`.

### Search budget and selection

The search screens all 16 combinations of state history `[1, 2, 4, 8]` and past
action history `[0, 1, 3, 7]`, using seeds 47 and 91. Every input includes the
current requested action in addition to that past-action count. Screening uses a
two-hidden-layer MLP with width 128 and four one-step epochs. Each epoch samples
100,000 training starts. Evaluation uses all 88,638 validation transitions and
128 fixed rollout starts at horizons 1, 8, 32, and 128.

The two leading contexts by mean seed score proceed to 64- and 128-unit MLPs and
two-layer GRUs, again with both seeds. Each finalist receives ten one-step epochs,
three generated-state epochs at horizon four, and three at horizon eight. Each
stage begins from the best validation checkpoint so far. Checkpoint selection uses
the fixed rollout/termination score documented in the training guide. This gives
32 screening runs and 16 finalist runs. A separate three-epoch baseline preceded
the search. These are matched training-example budgets, not equal wall-clock budgets.

Only validation chooses the configuration, seed, checkpoint, and terminal threshold.
The selected checkpoint receives one reserved-test evaluation, including all
182,079 test transitions and 512 fixed rollout starts. The search does not train
on test trajectories. Two seeds and a short screen are limited evidence about
architecture or minimum sufficient history.

Reproduce with new destination paths:

```bash
uv run gymemu dynamics prepare \
  dataset=logs/brick-grid-augmentation-20260917-v1/dataset \
  cache=data/state-dynamics-20260917 \
  data.train_episodes=128 data.validation_episodes=32 data.test_episodes=64
uv run gymemu dynamics search \
  cache=data/state-dynamics-20260917 output=runs/state-search-20260917 \
  trainer.train_samples=100000 trainer.validation_samples=0 \
  trainer.rollout_starts=128 evaluation.starts=512 'search.widths=[64,128]'
```

All runs save their resolved configuration, original episode identities, cache
identity, evaluation identities, metrics, and checkpoints locally.

Screening mean validation scores across the two seeds, lower is better:

| State history | 0 past actions | 1 past action | 3 past actions | 7 past actions |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 54.011 | 45.554 | 51.274 | 50.089 |
| 2 | 55.644 | 62.867 | 54.508 | 51.789 |
| 4 | 60.276 | 58.109 | 61.640 | 59.817 |
| 8 | 73.768 | 82.566 | 79.041 | 70.602 |

The one-state contexts with one and seven past actions advance. Larger input
histories fit worse within four epochs; this does not contradict the exact-input
audit showing that extra history resolves some ambiguity. Optimization difficulty
and information availability are separate questions.

### Final comparison and reserved test

Finalist mean and population standard deviation of the validation selection score
across two seeds. Each entry uses that run's best validation checkpoint.

| Model | Width | Past actions | Parameters | Mean score | Seed SD |
| --- | ---: | ---: | ---: | ---: | ---: |
| state_mlp | 128 | 7 | 50,548 | 36.822 | 1.947 |
| state_mlp | 64 | 7 | 21,236 | 36.936 | 0.856 |
| state_mlp | 128 | 1 | 47,476 | 38.034 | 0.821 |
| state_mlp | 64 | 1 | 19,700 | 38.470 | 1.038 |
| state_gru | 128 | 1 | 210,036 | 51.503 | 0.063 |
| state_gru | 128 | 7 | 210,036 | 53.554 | 2.854 |
| state_gru | 64 | 1 | 68,212 | 55.984 | 5.806 |
| state_gru | 64 | 7 | 68,212 | 59.614 | 1.895 |

The selected configuration is a two-hidden-layer, 128-unit MLP with one current
state and seven prior actions, plus the current action. It has 50,548 parameters.
Seed 91, epoch 16 supplies the saved checkpoint. The 64-unit model with the same
context is nearly tied; this experiment does not establish a reliable advantage
for width 128. GRUs rank worse within this training budget.

The reserved test contains 181,968 nonterminal state targets and 111 terminal
transitions. Its one-step native-position MAEs are:

| Predictor | Ball x | Ball y | Paddle x |
| --- | ---: | ---: | ---: |
| Selected MLP | 0.648 | 1.025 | 1.093 |
| Copy current state | 3.411 | 4.493 | 4.405 |
| Constant motion diagnostic | 0.190 | 0.603 | 2.385 |

Constant motion is an evaluation-only extrapolation using recorded velocity and
the two-native-frame interval, without learned or hand-written collision handling.
It has better aggregate one-step ball MAE than the selected network. The selected
network has better paddle MAE. Neither result establishes reliable collision
prediction or playable rollouts.

Recursive errors at each horizon use the same 512 sampled starting positions.
Available nonterminal targets decrease near segment boundaries. These diagnostics
continue prediction after an early predicted stop so it cannot conceal later errors.

| Horizon in model steps | Targets | Ball x MAE | Ball y MAE | Paddle x MAE |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 0.654 | 1.066 | 1.186 |
| 8 | 509 | 10.213 | 13.648 | 4.528 |
| 32 | 498 | 33.550 | 26.597 | 9.327 |
| 128 | 458 | 34.721 | 30.631 | 17.742 |

Mean reliable rollout length is **3.74 model steps**, versus 121.23 available
reference steps. A failure means ball error above eight native pixels, paddle
error above four pixels, any brick mismatch, or a termination mismatch. This is
a diagnostic threshold, not a claim of acceptable gameplay.

The aggregate losses conceal severe event failures:

- One-step brick-removal recall is **7.55%**, with **0.73% precision**. There are
  367 correctly predicted removals, 50,151 false removals, and 4,496 missed removals.
  Brick-cell accuracy is still 99.72%, illustrating why that metric alone is weak.
- At the validation-selected terminal threshold of 0.45, the model detects **7 of
  111 losses**, misses 104, and falsely terminates 130 live transitions. Recall is
  **6.31%**, precision 5.11%, and F1 0.0565.
- On recorded ball-velocity changes, one-step ball x/y MAE increases to
  **2.37/5.13 native pixels**.
- Across the 512 recursive starts, one rollout terminates early and 31 miss a
  reference terminal event.

The pipeline and finite search are complete; the selected model is not ready for
accurate simulation. These results motivate controlled experiments on collision,
brick-change, and terminal supervision, alongside more informative controller
state or longer memory. The exact-input audit proves some ambiguity, but it does
not explain away the large fit errors or rare-event failures. Further tuning should
use validation, with fresh held-out episodes for the next final comparison.

Artifacts:

- `runs/state-search-20260917/summary.json` contains every finalist and final test.
- `runs/state-search-20260917/candidates.json` contains all 48 search runs.
- `runs/state-search-20260917/final-state_mlp-h1-a7-w128-s91/best.pt` is selected.
- `logs/state-dynamics-playback-20260917/playback.json` contains 128 recursive
  predictions from validation episode 1103, source step 11. It stops at the step
  limit, not a predicted loss. This checks playback execution, not accuracy.
- `data/state-dynamics-20260917/manifest.json` records splits and segmentation.

Verification: the full suite passed with 309 tests and two skips, including the
existing direct and multi-stage train/play tests. The subsequent Hydra config and
CLI checks passed all 12 targeted tests. Ruff and whitespace checks passed.

## 2026-09-17 Loss scaling and isolated state targets

The user requested a cheap loss-scaling test first, then independent target models
if scaling did not produce good results. Joint architecture search remained paused.
All new fits use the previous train/validation episodes; the reserved test was not
reevaluated or used for selection.

### Loss scaling helped one task but did not solve the emulator

An 8,192-example training batch at the selected MLP's one-step epoch 10 contained
five terminal examples. Weighted shared-body gradient norms were motion 28.63,
bricks 0.01265, and terminal 0.04979. This is a single-batch diagnostic, not a causal
measurement of how much total error each loss causes.

Two matched continuations started from that same checkpoint. Both used seed 91,
ten one-step epochs, 100,000 sampled starts per epoch, learning rate 0.0005, and
the same validation targets. The control retained brick/terminal weights 1/0.2.
The alternative used 1,000/100 to bring those measured shared gradients closer
to the motion gradient. All other weights and the positive-class weighting stayed
unchanged. Both best composite-score checkpoints were their final epoch.

| Metric on validation | Control | Increased event weights |
| --- | ---: | ---: |
| Terminal F1 | 0.150 | 0.427 |
| Brick-removal recall | 0.16% | 3.33% |
| Brick-removal precision | 40.0% | 5.45% |
| Ball x MAE, native pixels | 0.514 | 0.746 |
| Ball y MAE, native pixels | 1.038 | 1.720 |
| Mean reliable rollout steps | 4.12 | 2.78 |

Scaling affected learning, but these particular fixed weights did not produce a
good joint model. The run is a bounded continuation test, not an exhaustive test of
adaptive balancing or training from scratch. Artifacts are
`runs/state-loss-scaling-{control,balanced}-20260917` and
`logs/state-loss-scaling-comparison-20260917.json`.

### Independent target models

Eighteen independent MLPs use nine target families and two contexts. Each has its
own weights and exactly one target loss, with no generated feedback. Brick occupancy
remains a 108-cell family; its cells are not separate models. The contexts use one
or eight state observations plus seven prior requested actions and the current
action. Each fit uses width 128, depth two, seed 91, twenty epochs, and 100,000
training starts per epoch. Validation uses all eligible targets. This is a single-seed
diagnostic, not a new architecture ranking.

| Scalar target | One-state validation MAE | Eight-state validation MAE |
| --- | ---: | ---: |
| ball_x | 0.450 | 0.481 |
| ball_y | 0.958 | 0.885 |
| ball_vx | 0.176 | 0.219 |
| ball_vy | 0.471 | 0.377 |
| paddle_x | 0.727 | 0.738 |
| paddle_vx | 0.550 | 0.580 |

Position errors are native pixels and velocities use the recorded native units.
Similar errors on the 20,000-example training diagnostic show that these models
also fit training data imperfectly. Several best checkpoints occur near epoch 20;
the budget does not establish convergence.

| Event metric | One state | Eight states |
| --- | ---: | ---: |
| Brick-removal F1 | 0.125 | 0.189 |
| Terminal F1 | 0.299 | 0.262 |
| Width-change accuracy | 0.000 | 0.000 |

The width predictors exceed 99.95% aggregate accuracy but miss all 36 recorded
width changes. Isolating targets does not remove within-target event imbalance.
Poor isolated fit alone does not establish that a variable is undiscoverable.

### Which inputs actually contradict their targets?

Exact-input fingerprints include every supplied float32 state, validity flag, and
requested action. IDs are used only to retrieve witnesses. The following counts
are conflicting input groups among all 359,313 training transitions. Undefined
state targets at terminal transitions are excluded from state comparisons.

| Context | Ball x | Ball vx | Paddle x | Paddle vx | Other state targets / terminal |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 states, 7 prior actions | 3 | 1 | 272 | 245 | 0 |
| 8 states, 7 prior actions | 1 | 1 | 53 | 47 | 0 |
| 16 states, 15 prior actions | 0 | 0 | 9 | 8 | 0 |
| 32 states, 31 prior actions | 0 | 0 | 3 | 3 | 0 |
| Complete available segment prefix | 0 | 0 | 3 | 3 | 0 |

With 32 states and 31 past actions, the remaining three paddle groups contain
nine examples, all with incomplete warm-up history. Restricting the audit to
348,577 examples with all 32 states available produces no observed conflicts.
This diagnostic restriction is not a training-data exclusion. Absence of observed
conflicts is not proof that these histories determine every possible successor.

### Recovered startup observations repair the known input problem

The remaining witnesses are original episode/step pairs 811:20 versus 909:20,
1362:20 versus 796:20, and 551:19 versus 699:19. Their available virtual segments
begin at steps 17, 15, and 16, respectively. Their earlier recorded paddle values
or requested actions differ during startup. The adapter had removed that useful
history because startup brick labels were not trustworthy. This is missing context
introduced by target eligibility, not proof that the original dataset lacks it.

Retaining 32 recorded paddle observations and actions resolves all three groups
without dropping the nine samples or crossing any life loss. The repair then
received an explicit implementation in `RecordedPaddleContext`. It treats paddle
history validity separately from brick-target eligibility, validates source
provenance/current-state alignment, and clears at life losses, lifecycle resets,
and missing paddle measurements. It neither edits the dataset nor changes the
eligible supervised targets. Unknown reset measurements remain masked.

A second audit uses the actual augmented probe inputs: one current full state,
seven cached past actions, and 32 retained paddle/action observations. Across all
359,313 training examples, it finds 478 repeated-input groups covering
1,158 examples and **no observed contradictory targets**.
This removes the counterexamples found in this sample; it does not certify full
observability on arbitrary scenes or unexamined trajectories.

Two independently trained paddle probes use those repaired inputs. Both retain
the same twenty-epoch budget and eligible target identities:

| Target | Original one-state MAE | Retained-context MAE |
| --- | ---: | ---: |
| paddle_x | 0.727 | 0.487 |
| paddle_vx | 0.550 | 0.486 |

The new input has more features and parameters, so the fitted error improvement
does not isolate startup recovery from longer context or increased capacity. The
exact-input refinement independently shows that recorded extra context distinguishes
the previously contradictory targets. Joint training and playback have not been
switched to this probe-only input contract.

### A small-fit check separates capacity from held-out accuracy

Three additional independent models fitted 256 training examples each: 128 event
examples and 128 other examples, for 1,000 full-batch updates. Brick removal and
terminal classification reached precision and recall 1.0 on those same examples.
Ball-y MAE reached 0.139 native pixels, including 0.129 on velocity-change cases.
This demonstrates that the architecture and gradient paths can fit these small
sets. It is memorization evidence, not held-out performance or proof of sufficient
inputs. Event sampling, input structure, and optimization remain candidates for
the broad-fit failures.

The next joint experiment should wait until the isolated targets achieve acceptable
validation event accuracy with the retained context. No further joint search was
run after the unsuccessful scaling test. In particular, there is no evidence here
that brick-change or terminal targets are inherently undiscoverable.

Artifacts include `logs/state-observability-20260917/` for contradiction witnesses,
startup traces, the retained-context audit, and small-fit results;
`runs/state-isolated-targets-20260917/` for all 18 original probes; and
`runs/state-restored-paddle-context-20260917/` for the two repaired-input probes.
Commands and checkpoint contracts are in
[the training guide](training.md#isolated-state-targets-and-input-sufficiency).

Verification: 313 tests passed with two skips before the context extension. After
the extension, all 17 targeted state, context, checkpoint, and CLI tests passed.
Ruff passed.

## 2026-09-17 Event errors and isolated convergence

The user requested event-specific diagnosis, longer isolated fits, inspection of
possible missing inputs, and an event-sampling experiment if collision errors
remained large. All experiments use the existing 128 training and 32 validation
episodes. The reserved test was not opened. The direct RGB approach, original
source dataset, target eligibility, and joint state player remain unchanged.

### Ordinary movement and collision errors

The existing eight-state isolated ball models were worse than constant motion on
ordinary movement. Validation contains 88,590 nonterminal targets: 81,292 with
unchanged velocity/bricks and 7,298 with velocity changes. The latter comprise
2,432 brick changes, 1,519 inferred paddle collisions, and 3,347 inferred wall
collisions. These categories use recorded successor labels and geometry; they are
not native collision flags, and no event labels enter predictor inputs. Two
cancelling native collisions could evade the unchanged-velocity proxy.

| Old isolated model, MAE in pixels | Ball x | Ball y |
| --- | ---: | ---: |
| Ordinary flight | 0.359 | 0.615 |
| Constant motion on those same targets | 0.000002 | 0.146 |
| Velocity-change transitions | 1.836 | 3.886 |
| Constant motion on those same targets | 2.333 | 5.688 |

Thus the models learned some collision response but also introduced avoidable
error during ordinary flight. The old ball-y model's largest errors were missed
brick reflections, approaching 15 pixels in one transition. Restoring paddle
history had not removed large errors near the paddle's travel limits.

### Longer fits help, but do not establish convergence

Three fresh independent models use eight full states, seven prior requested
actions plus the current action, and 32 retained paddle observations/actions.
Architecture is the registered context MLP, width 128 and depth two. Each uses
seed 91 and 100,000 training samples per epoch. Snapshots preserve the best
validation-RMSE checkpoint through epochs 20, 40, and 60, so comparisons within
this table hold inputs, initialization, and sampling sequence fixed.

Each selected 60-epoch checkpoint then initializes twenty more epochs at learning
rate 0.0003, seed 591, with a fresh AdamW optimizer. These are continuations from
selected weights, not exact optimizer-state resumes.

| Target, validation MAE in pixels | Through 20 epochs | Through 60 epochs | Lower-LR continuation |
| --- | ---: | ---: | ---: |
| Paddle x | 0.580 | 0.450 | 0.432 |
| Ball x | 0.408 | 0.370 | 0.313 |
| Ball y | 0.867 | 0.708 | 0.709 |

Selection minimizes RMSE, which is why the final ball-y MAE does not improve
although RMSE decreases from 1.548 to 1.507 pixels. Final training-diagnostic MAEs
are 0.399, 0.302, and 0.630 respectively. Error is therefore not solely a
validation generalization problem. The ball models still exceed the overall
constant-motion MAEs of 0.192 and 0.603 pixels. This is a single-seed diagnosis,
not a model ranking or evidence of accurate recursive gameplay.

### Oversampling collisions trades ordinary accuracy for event accuracy

Two ball-y continuations start from the exact same selected checkpoint, both with
fresh optimizers, learning rate 0.0003, seed 591, twenty epochs, and 100,000 draws
per epoch. The control samples uniformly. The alternative samples half from
velocity/brick-change transitions and half from ordinary transitions, repeating
rare events as necessary. It does not importance-correct the resulting objective.
Both select checkpoints by RMSE on the same full validation distribution.

| Ball-y validation MAE, pixels | Uniform | Event-balanced |
| --- | ---: | ---: |
| All nonterminal transitions | 0.709 | 1.050 |
| Ordinary flight | 0.506 | 0.942 |
| Velocity changes | 2.966 | 2.249 |
| Brick changes | 4.658 | 3.263 |
| Inferred paddle collisions | 3.630 | 2.181 |

The selected event-balanced checkpoint is epoch 2, versus epoch 20 for the
control. Oversampling helps rare events but is not an overall accuracy fix.
The balanced run is retained as an experiment, not promoted into joint training.

### Native-state inspection and information checks

Native-code inspection uses commit
`f06b9c81f4d717fd32782c5d877bb79a9042d266`, pinned by the earlier source audit.
Three diagnostics distinguish actual memory requirements from a failed fit:

- Ball y is an integer RAM coordinate even though the simulator moves it in fixed
  point. A diagnostic enumerates eighth-pixel phase candidates and filters them
  using source/past y and unchanged vertical velocity. Among 81,292 inferred
  ordinary-flight validation transitions, one observation uniquely determines
  49,164 next-y values; four determine 73,088, and eight do not increase that count.
  All 73,088 uniquely determined predictions match exactly. The remaining 8,204
  are ambiguous under this restricted constant-velocity calculation, not proven
  ambiguous under the full model input. The diagnostic selects free-flight targets
  for evaluation using successor labels, so it is not a deployable event detector.
- The native paddle controller uses charge, measured controller value, repeat
  state, and held-input state. A diagnostic reproduction of the pinned controller
  from reset metadata and original executed actions exactly matches recorded paddle
  x and velocity on **449,929 transitions across 160 episodes**, including startup
  and later lives. This is source-code replay with the full original action prefix,
  not a learned model or evidence that 32 observations suffice. It never enters
  learned playback. The code also confirms that controller state persists through
  life loss, while our learning contexts reset there.
- Native ball speed depends on cumulative paddle hits, with changes at hits 4, 8,
  and 12 before the fastest breakthrough mode. A source-prefix count reconstructed
  from inferred paddle bounces agrees with the recorded next-speed groups. All
  examined segment starts have the initial serve velocity, supporting a zero
  initial hit count for this audit. In all 3,338 training and 827 validation
  non-breakthrough paddle bounces, the previous paddle hit lies outside the
  eight-state window. Current velocity exposes a speed group but not the precise
  count within it. This identifies a memory requirement to test explicitly; it
  does not prove identical complete neural inputs have conflicting targets.

A separate exact-input audit removes ball/brick observations from the paddle
comparison because `update_paddle` never reads them. Among training samples,
eight retained paddle observations/actions produce 916 conflicting position groups;
32 produce ten groups covering 32 examples. Extending the requested context to
64 or 128 leaves those ten groups. All occur with incomplete retained context;
the 350,051 examples with all 32 paddle observations available have no observed
paddle-target conflicts. Adding the reconstructed native controller state
to the 32-observation inputs removes all ten conflicts. For example, episode
577 step 2056 and episode 1993 step 1315 have the same retained paddle input but
next positions 8 and 10 pixels; hidden charge/measurement differ. Both are at the
first usable step of a later life. This is not a contradiction of the earlier
full-input audit: their ball/brick inputs differ. Full-model observability remains
unproven, and near-unique full inputs can conceal missing causal memory.

A counterfactual check also swaps ball/brick histories while preserving paddle
history and actions. The earlier corrected paddle model changes its prediction
by 0.139 pixels on average over 20,000 validation examples, although native paddle
motion is independent of those swapped fields. This measures sensitivity to
irrelevant inputs, not in-distribution validation error or a proof of the main
error source.

The next useful input ablation is reconstructed controller state and cumulative
paddle-hit count, with explicit initialization at a life start, before combining
losses again. Their use in a learned rollout would require predicting/carrying the
additional state or a recurrent representation, not consulting future dataset
labels. Separately, the ordinary-flight errors show there is still avoidable
fitting error. Extra history length or larger event weights alone is unsupported
as the next fix.

Artifacts:

- `logs/state-event-diagnosis-20260917/`: event reports, worst-error witnesses,
  matched-budget comparisons, source snapshot, phase/controller/count diagnostics,
  and scripts that reproduce these checks.
- `runs/state-convergence-20260917/`: three 60-epoch isolated fits and budget snapshots.
- `runs/state-event-continuations-20260917/`: three uniform lower-LR continuations
  and the matched ball-y event-balanced continuation.

Implementation adds event diagnostics and training-only sampling to isolated
probes, safe initialization checks, checkpoint hashes, and budget snapshots.
Verification: `uv sync --frozen`; 317 tests passed and two skipped; Ruff and
`git diff --check` passed; a real existing joint checkpoint completed a 32-step
headless playback smoke. The seven real isolated fits also exercise training,
checkpoint saving/loading, and validation with corrected context. No joint-model
accuracy or playback improvement is claimed.

## 2026-09-17 Known internal-state inputs

The user requested isolated-model comparisons with reconstructed paddle-controller
state and cumulative paddle-hit count. Twelve fits were completed. All use the
existing training/validation episode splits; the reserved test and original dataset
were not changed or used for selection. These remain one-step diagnostic models,
not a joint emulator or a learned implementation of internal-state updates.

### Source-time inputs and controls

`state_hidden_context.py` writes a separate input cache with controller charge,
measurement, input-repeat state, held-input state, and paddle-hit count. Native
source commit `f06b9c81f4d717fd32782c5d877bb79a9042d266` is checked against its
snapshot SHA-256. Reset metadata and past executed actions reproduce the controller;
every recorded successor paddle position and velocity is checked. All 449,929
source transitions from the selected 160 episodes match. The saved inputs precede
the current action. The counter uses only earlier inferred paddle bounces and is
capped at twelve; every segment start has the initial serve velocity required to
initialize it at zero. It is a reconstructed count, not a newly recorded native
collision signal. Controller state continues through life loss. Normal observation
histories still stop at their existing boundaries.

Known normalization bounds are charge 3856, measurement 235, repeat 60, held 1,
and hit count 12. Preparation does not fit a scaler on validation. The separate
manifest records source/dataset/state-cache identities, checksums, and verification
counts. Loading checks array hashes, dimensions, and state-cache identity.

The registered `state_hidden_probe` has five extra input positions in every arm.
The `none` mode zeros them; `controller`, `hits`, and `both` expose the indicated
values. This holds parameter count fixed. All arms still have eight full states,
seven prior requested actions plus the current action, and 32 retained paddle
observations/actions.

Eight primary continuations start from the previously selected isolated paddle-x,
ball-x, and ball-y checkpoints. Prior learned weights are copied exactly and the
added first-layer columns start at zero. A numerical check confirms initial
prediction equivalence. The new weights remain trainable; their nonzero fitted
norms were checked. Each arm uses a fresh AdamW optimizer, seed 1591, learning rate
0.0003, thirty epochs, and the same 100,000 uniformly sampled starts each epoch.
Checkpoint selection minimizes native RMSE over all 88,590 validation nonterminal
targets. Evaluation identities and sampling seeds match within each comparison.

### Additional variables did not provide a useful improvement

| Target and added inputs | Validation MAE | Validation RMSE |
| --- | ---: | ---: |
| Paddle x, zero control | 0.424609 | 0.625517 |
| Paddle x, controller | 0.423349 | 0.621932 |
| Ball x, zero control | 0.255448 | 0.666323 |
| Ball x, controller and hits | 0.254923 | 0.666547 |
| Ball y, zero control | 0.708720 | 1.492494 |
| Ball y, controller | 0.708744 | 1.492505 |
| Ball y, hits | 0.708709 | 1.492522 |
| Ball y, controller and hits | 0.708771 | 1.492542 |

Positions use native pixels. More continuation training helped ball x relative to
its starting checkpoint, but the paired control improved by the same amount.
That gain cannot be attributed to the extra inputs. On the 114 non-breakthrough
validation paddle hits at count thresholds 4, 8, and 12, ball-y MAE changes only
from 2.479 to 2.469 pixels with the count alone. This is also not a useful fix.

A further pair tests vertical velocity directly. Both start from the same earlier
20-epoch eight-state ball-vy checkpoint, adding the retained paddle context and
internal-input columns with zero initial weights. They use the same thirty-epoch
continuation settings as above. Native velocity MAE/RMSE are 0.338542/0.956230 for
the zero control and 0.345531/0.962226 with hit count. Threshold-hit velocity MAE is
2.103 versus 2.230. This bounded fit also supplies no evidence of an improvement.
It does not establish that hit count is irrelevant to native dynamics.

### Fresh training and an exact paddle sufficiency check

To test whether continuation obscured the benefit of new inputs, a fresh pair of
paddle-x models uses matched random weights, seed 1591, sixty epochs, learning rate
0.001, and 100,000 samples per epoch. The same model/input dimensions and validation
selection apply. The selected checkpoints are both epoch 60.

| Fresh paddle fit | Validation MAE | Validation RMSE | Training diagnostic MAE |
| --- | ---: | ---: | ---: |
| Zero control | 0.468240 | 0.684134 | 0.446352 |
| Controller inputs | 0.462476 | 0.659333 | 0.449034 |

The small validation gain does not solve paddle prediction. Similar training error
also rules out a purely held-out generalization explanation for this fit.

A separate diagnostic initializes the pinned controller computation from only the
inputs supplied to the model: current paddle x, four reconstructed source-time
controller values, and the current requested action. It advances two native frames
and then compares its result with recorded targets. It uses neither future labels
nor the old action prefix inside this one-step calculation. Position and velocity
match exactly on **359,086 training and 88,590 validation nonterminal transitions**.

This is stronger evidence than the earlier absence of duplicate-input conflicts:
the supplied variables suffice for the tested paddle transition mapping. With
those values included, the neural paddle model's remaining error is a fitting
problem, not missing input information. The exact diagnostic is handwritten native
logic and never enters learned playback. No equivalent sufficiency claim is made
for ball dynamics, which also contain subpixel state and collision memory.

The next controlled paddle experiment should simplify the learned mapping to
current paddle/controller state plus action, removing unrelated ball/brick inputs
and redundant history. This would test whether the input representation and
optimization are preventing the model from fitting a known sufficient state.
A different output objective may also merit testing later; these runs do not
identify a unique architecture or loss defect. Larger history windows or another
joint-model search are not supported by these results.

### Artifacts and verification

- `data/state-hidden-inputs-20260917`: separate normalized input arrays and provenance.
- `runs/state-hidden-inputs-20260917`: eight primary continuations, expanded starting
  checkpoints, fixed plan, run configurations, validation metrics, and diagnostics.
- `runs/state-hidden-speed-20260917`: two direct vertical-velocity comparisons.
- `runs/state-hidden-scratch-20260917`: two fresh paddle fits.
- `logs/state-hidden-inputs-20260917`: reproduction scripts, the exact one-step
  controller check, comparison receipt, and existing joint-player smoke.

`uv sync --frozen` passed. The full suite reports 321 passed and two skipped;
Ruff and `git diff --check` passed. Tests cover source-time hit counting, boundary
resets, controller source-state snapshots, input-array alignment/checksums, and
feature masks with fixed parameter counts. A real existing joint checkpoint
completed 32 headless playback steps. The twelve real fits exercise model
construction, training, compatible checkpoint initialization, save/load, and
validation with the new input adapter. These are single-seed controlled diagnostics,
not evidence of robust architecture superiority or improved recursive gameplay.

## 2026-09-17 Accurate paddle learning from sufficient inputs

The user requested accurate learning with known sufficient inputs before returning
to missing-history questions. The experiment therefore isolates only the paddle-x
transition. It receives current paddle x, controller charge, measurement, repeat,
held-input state, and the current requested action. These are the same variables
whose sufficiency was independently checked against the pinned native controller.
There is no ball/brick input, old history, or native update formula inside the neural
predictor. Correct controller values are provided at each evaluated transition.

### Encoding and output objective resolve the fitting problem

A new registered MLP has two 128-unit SiLU hidden layers. A 2-by-2 experiment compares
scaled scalar inputs versus the same inputs supplemented with binary digits, and
native displacement regression versus integer-displacement classification. Binary
encoding is generic integer representation, not an extra hidden variable or a
handwritten controller feature. The scalar model receives eight features including
one-hot action; the hybrid model receives 43. The classifier has 23 displacement
classes, -11 through +11, determined from training data. No class bounds are fitted
on validation or test targets.

Every run uses all 359,086 nonterminal training transitions each epoch, batch size
1024, one hundred epochs, AdamW at 0.001 with no weight decay, cosine decay to
0.00005, and gradient clipping at norm five. The four primary runs share seed 1591
and identical shuffled sample order. The successful hybrid classifier is repeated
with seed 2026. Checkpoints minimize the number of validation integer-pixel errors,
then raw RMSE. For regression, exact-pixel accuracy uses rounded predictions;
regression raw MAE remains separately reported.

| Input encoding / output | Validation exact pixel accuracy | Final-test exact pixel accuracy |
| --- | ---: | ---: |
| Scalar / regression | 79.2731% | 78.8547% |
| Scalar / classification | 91.6277% | 90.7901% |
| Hybrid / regression | 99.9052% | 99.7346% |
| Hybrid / classification, seed 1591 | 99.9977% | 99.9808% |
| Hybrid / classification, seed 2026 | 100.0000% | 99.9863% |

Validation has 88,590 transitions from 32 episodes. The seed-1591 classifier makes
two validation errors, both one pixel and sharing the same input tuple. The second
seed makes zero. Both selected classifier checkpoints occur at epoch 21. Their
training errors are one and nine respectively; later epochs are not substituted
based on test results. The hybrid regression model makes 84 rounded validation
errors, versus 18,362 for scalar regression. Thus input representation contributes
strongly even with MSE, and classification provides a further improvement. These
results do not establish that MSE intrinsically cannot learn the mapping.

### Separate final test after checkpoint selection

The seed-2026 hybrid classifier is selected using validation only. Checkpoint paths
and SHA-256 values for all five models are frozen before preparing test inputs.
No architecture, loss, input, checkpoint, or training changes use final-test metrics.
The final test has **181,968 nonterminal transitions from 64 original held-out
episodes**. Each model is evaluated once on that fixed set.

The selected 25,111-parameter model is exact on **181,943 transitions**, making
25 errors with mean absolute position error **0.000159 pixels** and maximum error
three pixels. The independent seed makes 35 errors, with mean absolute error
0.000231 pixels. The selected model is exact on 99.9846% of moving-paddle targets
and 99.9938% of stationary targets, so the result is not explained by copying a
stationary paddle. Its accuracy on the first eight usable steps of each segment
is 99.9268% over 1,367 targets.

Final-test input reconstruction initially stopped at episode 400, step 2521 before
any neural test evaluation. Its final action terminated the last life after one
native frame, whereas preparation had assumed two frames for every action. The
pinned native code checks loss at screen y >= 217, corresponding to RAM y >= 208.
Preparation now determines that one-frame stop from the preceding source record's
ball-y and lives values. It does not inspect the successor to choose the frame
count. This repair verifies all **182,810 source transitions** in the selected test
episodes, including four one-frame final actions, with zero controller mismatches.
All these terminal transitions remain excluded from paddle targets. The models
stay frozen during this data-alignment repair. The failed incomplete cache is
retained separately from the completed final-test cache.

### Supported conclusion and limits

A neural network can learn the tested paddle mapping extremely accurately when
provided a sufficient current state. The earlier failure was not evidence that the
paddle was unlearnable. For this experiment, simplifying the input and representing
integer structure explicitly made optimization much more effective; selecting an
integer output improved the result further. The controlled 2-by-2 comparison
supports both effects within this model/training budget. Different input/output
sizes mean parameter counts differ; this is not a pure equal-parameter causal
estimate or a proof of the uniquely optimal architecture.

The result is one-step paddle position with **correct controller state supplied**.
No controller-update model, history encoder, ball predictor, brick predictor, or
full learned rollout is established by this experiment. The classifier calls no
native transition code. Native code is used only outside it to reconstruct and
verify the sufficient-input labels. A later experiment must learn or infer those
internal values if they will not be available during simulation.

Artifacts:

- `runs/paddle-sufficient-{scalar,hybrid}-{regression,classification}-20260917`
  and `runs/paddle-sufficient-hybrid-classification-seed2026-20260917` contain
  configurations, per-epoch metrics, summaries, and safe model checkpoints.
- Selected checkpoint:
  `runs/paddle-sufficient-hybrid-classification-seed2026-20260917/best.pt`.
- `logs/paddle-sufficient-20260917/frozen-selection.json` and `final-test.json`
  contain the pre-test selection hashes, per-episode errors, and provenance.
- `data/paddle-sufficient-final-test-20260917-v2` contains the separate verified
  test-input arrays. Original datasets and source train/validation inputs are unchanged.

Verification includes five real bounded fits, checkpoint save/load equivalence,
full final-test inference, generic bit-encoding/action tests, source/target alignment,
split separation, and the native-frame early-stop regression. The existing joint
player also completed a 32-step headless smoke. After the final data-alignment repair, the full suite reports **324 passed and
two skipped**. Ruff and `git diff --check` pass. `uv sync --frozen` passed; no
dependencies, original dataset files, or shared RGB baseline behavior changed.

## 2026-09-18 Controller state from history

The preceding paddle predictor was accurate with correct internal controller
inputs supplied. This experiment asks whether current charge and measurement can
instead be inferred from observed paddle/action history. These are changing state
variables, not neural-network parameters.

### What the history contains

Inputs contain H paddle x, native-frame velocity and width observations, validity
flags, and H preceding requested actions. There are no current/future actions,
future observations, ball/brick inputs, or internal values in the neural input.
Current charge and measurement are labels from the provenance-checked native
reconstruction. All 359,086 nonterminal training targets and 88,590 validation
targets are retained, in the existing disjoint original-episode splits. No test
split is read. Valid startup paddle observations remain available, but life loss
clears history, as in the existing experiment contract.

Requested and executed actions differ on 287 of 449,929 original train/validation
transitions: 160 RIGHT and 127 LEFT requests become FIRE during automatic serving.
The probes deliberately use requested actions, matching the prior history models.
The complete-reset reconstruction claim uses executed actions and correct native
frame counts. The original dataset is unchanged.

### Exact ambiguity audit

The read-only audit groups exactly equal model-visible histories, then checks for
different current controller labels. Counts below exclude incomplete windows.
A conflicting group has identical inputs and at least two different target values.

| Observations | Complete training windows | Conflicting charge groups | Conflicting measurement groups |
| --- | ---: | ---: | ---: |
| 8 | 357,404 | 13,072 | 1,321 |
| 16 | 355,103 | 425 | 67 |
| 32 | 349,609 | 0 | 0 |

The zero at 32 observations is not a sufficiency proof: only 133 complete input
groups repeat at that length; most histories appear once. At 32, 64 and 128
observations, including incomplete windows gives 65 conflicting charge groups and
19 conflicting measurement groups. The same groups remain when using the entire
retained life prefix. Thus more padding/history capacity cannot resolve all the
later-life startup cases after prior controller information has been discarded.
These are paddle-only input findings, not claims of identical full RGB or full-game
state histories.

A native-controller counterexample establishes the general finite-window limit
more directly. Two controller states observed at episode/step 143/197 and 594/197
have paddle x=144 and charge/measurement pairs (603,38) and (655,42). Both use repeat
60. Extend each with 128 no-movement FIRE actions. Both produce x=144 and vx=0 at
every observation while retaining their different charge and measurement values.
One subsequent LEFT action yields x=144 and x=142 respectively. Wall clamping
makes the no-movement observation sequence a fixed point, so it can be extended
arbitrarily in the controller subsystem. This is a deterministic diagnostic using
the pinned native rules, not neural inference or a full-game rollout. It proves
that a finite paddle/action window starting from an unknown internal state cannot
always identify either variable exactly.

Conversely, a known reset plus complete executed-action history determines the
controller through its deterministic updates. The existing reconstruction verified
449,929 recorded paddle transitions without a mismatch. That establishes an
available label-generation path, not a learned recurrent model.

### Four independent learned probes

Each variable receives a separate registered `controller_history_mlp`. Inputs use
scaled scalars and ordinary binary digits of the same visible values, two 128-unit
SiLU layers, and cross-entropy over integer values observed in training. The charge
vocabulary has 383 values and measurement has 76. Sixteen validation charge labels
are outside the training vocabulary and are counted as errors; measurement has
none. No hidden labels enter the model input.

Each fit runs 50 epochs over all training samples, with batch size 2048, seed 2026,
AdamW at 0.001 without weight decay, cosine decay to 0.00005, and gradient clipping
at five. Selection uses validation exact-integer errors, then MAE. These are bounded
initial probes, not an exhaustive architecture or optimization search.

| Target | Observations | Train exact | Validation exact | Validation native MAE |
| --- | ---: | ---: | ---: | ---: |
| Charge | 8 | 68.6304% | 61.0216% | 13.2901 |
| Charge | 32 | 73.0616% | 59.9481% | 7.1849 |
| Measurement | 8 | 96.6342% | 95.7332% | 0.2657 |
| Measurement | 32 | 97.9178% | 95.2354% | 0.1641 |

Longer history reduces average error in these fits but does not improve exact
validation accuracy. Charge remains much harder to learn exactly than measurement.
The 32-observation charge model also has substantial training error, so its
validation error cannot be attributed entirely to missing information. Neither a
failed fit nor the empirical duplicate-input audit gives a population accuracy
limit for this dataset distribution. Wider/longer-trained or recurrent models
remain untested here.

### Effect on the existing paddle predictor

The frozen successful paddle-transition classifier is evaluated on the same
88,590 validation targets. Replace its correct charge and/or measurement with the
32-observation probe outputs, keeping current paddle x, repeat, held and requested
action correct. This is one-step inference only.

| Inferred fields | Exact next paddle position | Errors | MAE in pixels |
| --- | ---: | ---: | ---: |
| None, correct controller | 100.0000% | 0 | 0 |
| Charge | 96.9048% | 2,742 | 0.03959 |
| Measurement | 97.0245% | 2,636 | 0.04635 |
| Charge and measurement | 94.7488% | 4,652 | 0.07142 |

Both inferred fields together are not accurate enough to replace the known-state
inputs for faithful simulation. The independent classifiers can also produce
inconsistent charge/measurement combinations; the experiment does not quantify
that contribution separately. Correct repeat/held inputs remain supplied, and no
controller updates or recursive rollouts are learned. The next design worth
investigating is persistent learned controller state initialized from a known
start, with a separate policy for arbitrary-scene initialization.

Artifacts are `logs/controller-history-20260918/{audit,training,downstream}.json`,
`witnesses.json`, `wall-counterexample.json`, `action-audit.json`, the corresponding
local driver scripts, and `runs/controller-history-{charge,measurement}-h{8,32}-20260918`.
The registered model and source-time input builder are covered by boundary,
current-action exclusion, invalid-observation masking and safe checkpoint roundtrip
tests. All four real fits also verify raw/encoded inference and checkpoint reload
agreement. The existing joint player completed a bounded 32-step smoke; this does
not represent a rollout of the new inference probes.

Verification: `uv sync --frozen` and Ruff pass; the full test suite reports
**326 passed, two skipped**. `git diff --check` passes. No dependency, source dataset,
shared RGB runner, or player behavior changes were needed.

## 2026-09-18 Paddle controller values in the complete dataset

Added the four reconstructed controller values directly to a new local dataset
version at `data/breakout-paddle-controller-20260918`. Every transition contains
source and successor charge, measurement, repeat, and held-button values. Episode
rows also contain the initial controller state after seeded reset no-ops. Raw
integer/boolean columns have explicit timing and normalization metadata.

The annotator replays the pinned native controller using recorded executed actions.
Controller memory continues across life losses. It handles the last-life early
stop using the preceding source labels. Successor labels verify each replayed
paddle position and velocity; they do not determine the source controller values.
The inferred paddle-hit count is excluded.

| Split | Episodes | Verified transitions | Replay mismatches | One-frame final-life stops |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,856 | 5,385,620 | 0 | 68 |
| Held-out | 464 | 1,343,290 | 0 | 23 |
| Total | 2,320 | 6,728,910 | 0 | 91 |

Generation, replay checks, Parquet readback, and source-file hash verification
completed in **262.2 seconds**, about 4.4 minutes. All original columns compare
unchanged after readback. The 832 rewritten transition/episode files occupy
565,345,180 bytes. Images are unchanged and linked from the existing local asset
store. Original data and split memberships remain unchanged. The output has not
been uploaded to the Hub; earlier runs still retain their original dataset identity.

The builder is `gymemu/augment_paddle_dataset.py`. Schema, timing, native source,
builder code, validation counts, and file hashes are under
`annotations/breakout-paddle-controller-v1/` in the new dataset. Earlier brick
annotation receipts remain historical input receipts. The new receipts describe
the rewritten files. Tests cover source/successor/initial alignment, episode
continuity across shard boundaries and life losses, executed-action selection,
final-life frame counts, preservation of original columns, and rejection of
missing transitions or replay mismatches. Full verification: **329 tests passed,
two skipped**; `uv sync --frozen`, Ruff, and `git diff --check` passed.

A bounded compatibility smoke prepared a cache from the new dataset, trained the
existing state model on 128 examples for one epoch, reloaded its checkpoint, and
completed 32 playback steps. It did not evaluate the test split or train a new
controller model. Artifacts and the reproduction script are in
`logs/paddle-annotation-20260918/`.

## 2026-09-18 Minimum controller inputs for paddle position

Tested whether the successful one-step paddle predictor still learns accurately
when controller inputs are removed. Current paddle x and requested action remain
in every fit. Input selection now occurs before scalar/binary encoding in the
registered `paddle_transition_mlp`; omitted values cannot affect its output. Older
checkpoints default to their original four controller fields.

Four fresh fits use the same 359,086 nonterminal training transitions, 88,590
validation transitions from 32 separate episodes, width 128, two hidden layers,
100 epochs, batch size 1,024, optimizer schedule, and seeds 1591/2026 as the earlier
experiment. They predict integer displacement with cross-entropy. No final-test
data were read and no controller-update model was trained.

| Controller inputs | Seed | Validation errors | Exact position accuracy | Selected epoch |
| --- | ---: | ---: | ---: | ---: |
| Original four, frozen reference | 1591 | 2 | 99.9977% | 21 |
| Original four, frozen reference | 2026 | 0 | 100.0000% | 21 |
| Charge, repeat, held | 1591 | 8 | 99.9910% | 52 |
| Charge, repeat, held | 2026 | 12 | 99.9865% | 68 |
| Charge only | 1591 | 7 | 99.9921% | 78 |
| Charge only | 2026 | 4 | 99.9955% | 67 |

Every reduced model predicts all 67 validation transitions with an unsaturated
repeat counter correctly. The charge-only seed-2026 model's four errors are each
one pixel. Its seed-1591 repeat has maximum error two pixels. Charge-only models
have 25 encoded inputs and 22,807 parameters; charge/repeat/held has 34 inputs and
23,959 parameters. Thus input sizes and parameter counts differ slightly.

An exact-input audit of all 16 controller subsets on the combined development
sample finds no position conflicts when charge is included. Removing all controller
inputs creates 411 conflicting groups and at least 251,012 unavoidable errors over
447,676 examples. Measurement alone also has conflicting targets. These are
finite-sample findings, not proof of complete observability or a minimal state
capable of updating itself recursively. Repeat is already 60 in 358,774 training
and 88,523 validation examples, so its rare startup behavior needs separate care.

Artifacts: `logs/paddle-minimal-inputs-20260918/comparison.json`,
`reference-validation.json`, `audit.json`, and the reproduction scripts in that
directory. Checkpoints are in `runs/paddle-minimal-{charge,charge-repeat-held}-s{1591,2026}-20260918`.
The new input-exclusion and checkpoint tests pass, including old checkpoint specs
without an explicit field list. All four fits verify checkpoint reload agreement.
The full suite reports **333 passed, two skipped**; Ruff and `git diff --check`
pass. The existing joint player completes 32 steps. This is a compatibility check,
not a rollout of the reduced paddle model.

### Full training partition and reachable controller states

The wider audit covers 5,385,552 original training transitions with exactly two
native frames, including startup. It excludes 68 final-life actions that stop after
one frame. Mixing those durations initially produced 40 conflicts even with all
controller inputs; the fixed-duration audit removes all 40. Neither audit reads
the held-out partition. With x, charge, and executed action there are **zero
conflicting next positions and zero conflicting next charges** in the fixed-duration
sample. With only x and action there are 411 conflicting position groups and at
least 3,162,890 unavoidable errors. Held matches the previous executed direction
flag on every original training row.

A stronger diagnostic enumerates the pinned paddle controller from every allowed
reset no-op count, 1 through 30, then explores every sequence of FIRE, RIGHT, and
LEFT applied for two native frames per action. There are seven distinct initial
controller states, **3,279 reachable controller states**, and **9,837 outgoing
transitions**. Enumeration runs to exhaustion. When full controller states share
x and charge, each action always produces the same next x and next charge.
Thus x plus charge is a closed predictive state for this pinned controller under
the specified resets and action timing. This includes its reachable startup cases;
keeping repeat and held separately is not required under this contract. This is
an offline native-rule sufficiency check, not learned charge prediction or neural
rollout evaluation. Different frame skips or arbitrary manually restored controller
states are outside the enumeration.

This corrects the earlier conservative advice to retain repeat during startup.
The raw internal variables can differ without requiring different predictions at
the recorded action cadence. The learned one-step model needs only current x,
charge, and action; learning next charge and testing repeated predictions remain
separate work. Full audit results and reproduction scripts are
`full-training-fixed-audit.json`, `charge-update-audit.json`, and
`reachable-controller.json` in the diagnostic directory. The initial mixed-duration
report is retained separately as `full-training-audit.json`.
All 3,262 distinct full controller source states observed across the original
training partition are contained in that reachable-state enumeration.

## 2026-09-18 Learned charge updates and removal of paddle velocity

New state-model configs set `predict_paddle_velocity=false`. The MLP/GRU and
retained-context models exclude velocity from their encoded inputs and learned
heads. The state objective and prediction metrics omit that target, and default
isolated-target sweeps exclude it. The existing 115-slot cache layout remains
unchanged; new generated states reserve slot 5 as zero. Original dataset columns
and historical checkpoint behavior remain available. A model spec without the
new option retains the legacy architecture. Native paddle motion and ball
collision code do not consume paddle velocity; it is a recorded measurement.

Two independent compact classifiers learn the missing charge update using only
current paddle x, charge, and requested action. Labels come directly from the
newly annotated transition tables, joined by episode and step with provenance,
checksum, duplicate, missing-row, and source-charge checks. Every selected
successor charge was also compared against the earlier reconstructed cache with
exact agreement. Death targets are excluded. There are 359,086 training targets
and 88,590 validation targets, on the existing disjoint episode split. No reserved
test data is used.

Each model uses scalar-plus-binary encoding, two 128-unit SiLU layers, 21,259
parameters, and cross-entropy over 11 training-derived charge-change classes:
-120, -60, -9, -5, -1, 0, 1, 5, 9, 60, 120. Prediction adds the selected change to
current charge. This is learned inference; it does not execute controller rules.
Training uses the previous 100-epoch, batch-1024 AdamW/cosine budget with seeds
1591 and 2026. Validation integer-error count selects the checkpoint, then RMSE;
ties retain the earlier checkpoint.

| Seed | Selected epoch | Training errors / 359,086 | Validation errors / 88,590 |
| --- | ---: | ---: | ---: |
| 1591 | 19 | 0 | 0 |
| 2026 | 16 | 0 | 0 |

Both selected checkpoints are exactly correct on all 55,600 changing-charge
validation targets, 32,990 unchanged targets, 632 early-segment targets, and 67
unsaturated-counter targets. Charge MAE and maximum error are zero on this
validation set. These figures do not claim perfect predictions on every possible
controller input.

### Predicted position and charge fed back together

Each frozen charge model is paired with the earlier frozen charge-only position
classifier from seed 2026. The models both read the same current x/charge/action,
then both outputs supply the next state. Only initialization uses recorded x and
charge. Checkpoint hashes and 256 sampled validation starts are saved before this
evaluation. Requested actions are recorded; scoring stops at reference life or
segment boundaries. This is a paddle subsystem test, not learned full-game
termination or ball/brick dynamics.

Each charge seed makes **zero charge errors over 30,909 scored recursive
predictions**, with up to 128 steps per start. There are no out-of-range charge
predictions. The shared position model makes one transient one-pixel error across
those predictions. There are 226 starts with 128 available steps; all have exact
x and charge at step 128, while 225 have an entirely exact joint prediction prefix.
The position error recovers without a recorded-state correction. All charge
prefixes remain exact. A first rollout-script attempt encountered padding action
3 on already inactive rows; inactive rows now receive a harmless FIRE input and
remain excluded from scoring. Model weights and scene selection were unchanged.

Checkpoints are `runs/paddle-charge-s{1591,2026}-20260918/best.pt`. Configs save
target, sparse class values, input fields, and annotation provenance. Experiment
scripts, selection hashes, recursive metrics, and summaries are in
`logs/paddle-charge-20260918/`. The diagnostic models are separate from the joint
state-player checkpoint format.

Verification: the full suite passed **338 tests, two skipped**. After adding the
charge-label join/checksum regression, all **30 targeted state/charge tests** pass.
Ruff, `uv sync --frozen`, and `git diff --check` pass. A bounded new velocity-free
state fit completed one-step and recursive training, reloaded its checkpoint, and
played 32 steps. The earlier joint checkpoint also played 32 steps unchanged.

## 2026-09-18 Horizontal-velocity input and learning audit

The user requested an assessment of missing information and prediction errors,
with approval required before any dataset additions. This investigation reads
the existing state and controller caches and writes only diagnostic artifacts.
It does not modify datasets, training defaults, playback, or the direct baseline.
The reserved test is not read.

The saved one-state horizontal-velocity regressor reproduces validation MAE
0.175807, increasing to 2.123237 on the 3,083 transitions where horizontal
velocity changes. Training MAE is similarly poor at 0.172812 overall and 2.098019
on changes. Only 3.48% of validation transitions change horizontal velocity;
on unchanged transitions the predictor still introduces MAE 0.105591.
This is an isolated target fit, so competition between different target losses
cannot explain this particular result.

### Native-rule diagnostic

A diagnostic transcription of the pinned native source advances two native
frames, including collision order, paddle movement, speed changes, and brick
contact memory. It uses current recorded state and existing controller inputs.
Paddle-hit count comes from the previously prepared source-prefix diagnostic,
not from the transition being predicted. All eight fractional-y possibilities
and both brick-contact flags are considered. Each source candidate set is filtered
only by earlier transitions within the life; current successor labels enter only
the scoring and the next source's candidate set. Life starts are checked for serve
velocity and position below the bricks, with unknown fractional y retained.

| Diagnostic | Train | Validation |
| --- | ---: | ---: |
| Nonterminal transitions | 359,086 | 88,590 |
| Correct next horizontal velocity uniquely determined from the source prefix | 359,084 | 88,588 |
| Transitions still allowing different next horizontal velocities | 2 | 2 |
| Recorded successors inconsistent with every surviving candidate | 0 | 0 |

The two ambiguous validation witnesses are episode 173, step 2930, and episode
1552, step 2076. Fractional-y candidates 4/8 and 5/8 produce opposite horizontal
directions near the paddle. This does not prove irreducible ambiguity under all
original episode information: the diagnostic deliberately starts each life with
unknown fractional y. With controller and hit-count inputs held available,
one observed state leaves 161 ambiguous validation predictions; eight states
leave 112. The full source prefix reduces this to two.

Relevant input findings:

- Paddle controller information affects the paddle position on the intermediate
  native frame. It is already available in the augmented dataset, but the old
  horizontal-velocity regressor does not consume it.
- Exact paddle-hit count affects speed changes. It is absent from the dataset
  columns and old neural input, but is reconstructible from prior bounces under
  the checked life-start contract. Replacing it by the start of its current speed
  group introduces 69 errors among the 88,588 uniquely determined validation
  predictions. Replacing it by zero introduces 586.
- Integer RAM ball y omits the fractional coordinate. Forcing the fraction to
  zero introduces 107 errors on that same identified subset. Clearing brick-contact
  memory introduces nine. These are diagnostic interventions, not measured error
  lower bounds for a learned model with recorded history.

### Controlled wall-bounce fits

A source-only subset selects RAM ball y from 100 through 160. Over the next two
native frames the ball cannot reach the paddle or bricks. Current ball x and
horizontal velocity alone determine its next horizontal velocity. The native wall
calculation matches all 186,970 training and 44,856 validation targets exactly;
validation contains 870 wall reflections.

Two small diagnostic MLPs use the same two scalar inputs, two 128-neuron SiLU
hidden layers, uniform full-subset sampling, AdamW, 60 epochs, and seeds 91/2026.
The regressor predicts an additive velocity change. The classifier chooses keep
or reverse and retains the recorded speed magnitude. Both select by validation
MSE. Classification therefore changes the allowed outputs as well as the loss;
this is not evidence that replacing MSE alone causes the improvement.

| Predictor | Validation MAE | Wall-reflection MAE | Exact classifier errors |
| --- | ---: | ---: | ---: |
| Existing full-input regressor, same subset | 0.129760 | 1.786900 | n/a |
| Two-input regressor, seed 91 | 0.053120 | 0.825310 | n/a |
| Two-input regressor, seed 2026 | 0.047987 | 0.825327 | n/a |
| Keep/reverse classifier, seed 91 | 0.000223 | 0.011494 | 8 / 44,856 |
| Keep/reverse classifier, seed 2026 | 0.000245 | 0.012644 | 9 / 44,856 |

The classifiers are 99.9822% and 99.9799% exact overall, and 99.0805% and 98.9655%
exact on reflections. These results establish avoidable learning error where
inputs are sufficient. They do not establish full-game horizontal-velocity
accuracy or recursive reliability. No sampling ablation was run for this target,
so class imbalance remains a plausible contributor rather than a demonstrated
sole cause. All eight observed nonterminal horizontal velocities are discrete:
plus/minus 0.5, 1, 1.5, and 2.

The next experiment should retain isolated horizontal-velocity supervision and
test discrete outputs with the already available controller input and explicitly
tracked source-time collision memory. Resolve fractional-y initialization before
claiming complete input sufficiency. Do not add dataset columns without approval.

Artifacts and reproduction commands are in `logs/ball-vx-audit-20260918/`:
`check_baseline.py`, `audit.py`, `wall_fit.py`, split audit reports, and four
diagnostic checkpoints. These models are experiment-only and are not registered
for application playback.

## 2026-09-18 Discrete horizontal velocity with sufficient source inputs

The user authorized the discrete-output experiment and timing investigation while
retaining the requirement to ask before adding dataset content. No dataset or
state-cache files were changed. The six fits use the existing 359,086 training and
88,590 validation nonterminal transitions. The reserved test was not opened.

### Resolving the two timing witnesses

The earlier candidate audit initialized each life with an unknown fractional
vertical position. The native source instead preserves the fraction through loss
and serve. Reset starts at an integer position. The first retained segment in each
checked episode starts by recorded step 18, before the initially descending ball
can reach the paddle; its fraction is therefore zero. At later lives, replay carries
the fraction through the last life-loss transition, including the first native
frame's movement when loss happens on the second frame. Training targets and
history windows still stop at each life boundary.

This source-only replay exactly matches the four recorded successor ball fields
and all brick cells on every training and validation transition. Its source
paddle-hit counts also agree with the existing diagnostic cache. It uses no
successor to select current memory. There are 220 fraction carries between lives
in training and 47 in validation, never between recorded episodes. Both earlier
validation witnesses have fraction 5/8 and next horizontal velocity -2. The
information ambiguity is resolved; this does not mean a neural model predicts
those witnesses correctly.

### Matched training comparison

The registered `ball_velocity_mlp` receives 118 source values: six ball/paddle
measurements excluding paddle velocity, charge, previous paddle-hit count,
fractional y, brick-contact memory, and 108 brick cells. Scalar encoding uses
118 inputs. Hybrid encoding adds 64 binary features for the ten scalar fields,
giving 182 inputs. It adds no native collision or transition rules to inference.
No images or actions are inputs: within this fixed two-native-frame contract,
the relevant intermediate paddle position depends on the source controller state.
That simplification is not claimed for other frame skips.

All fits have two 128-neuron SiLU hidden layers, batch size 1,024, 100 full-data
epochs, AdamW with learning rate 0.001 and weight decay 0.0001, cosine decay to
0.00005, and gradient clipping at five. Classifiers predict one of the eight valid
horizontal velocities using cross-entropy. The regression control predicts a
continuous residual using native-velocity MSE. Best checkpoints minimize
validation native MSE, retaining the earlier epoch on ties. The memory ablation
zeros only hit count, fractional y, and brick contact; charge and architecture
remain unchanged. The two scalar seed-91 classifiers share initialization and
batch order. Scalar classification has 32,776 parameters, scalar regression
31,873, and hybrid classification 40,968.

| Model | Seed | Selected epoch | Validation MAE | Validation RMSE | Exact validation | Exact when vx changes | Exact on paddle hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Earlier regressor, historical budget/input reference | 91 | historical | 0.175807 | 0.497463 | n/a | n/a | n/a |
| Scalar classifier, collision memory masked | 91 | 97 | 0.051202 | 0.421244 | 98.0551% | 70.1265% | 51.5471% |
| Scalar classifier, collision memory included | 91 | 93 | 0.050835 | 0.418337 | 97.9332% | 72.6889% | 52.7979% |
| Scalar classifier, collision memory included | 2026 | 80 | 0.053132 | 0.426034 | 97.8564% | 72.9484% | 53.7196% |
| Scalar regression, collision memory included | 91 | 95 | 0.120823 | 0.392112 | n/a | n/a | n/a |
| Hybrid classifier, collision memory included | 91 | 80 | 0.017186 | 0.249873 | 99.4311% | 90.1070% | 75.3785% |
| Hybrid classifier, collision memory included | 2026 | 57 | 0.016407 | 0.243814 | 99.4582% | 91.7937% | 76.9585% |

Scalar classification improves MAE but has worse RMSE than the matched scalar
regressor; committing to a wrong velocity can produce a larger error. Hybrid
classification improves both metrics. The earlier checkpoint is a historical
reference, not a matched-budget attribution of the improvement to classification.
Snapping the new scalar regressor to the nearest valid velocity yields MAE
0.077503 and 93.0861% exact, so quantizing its existing outputs does not match
the categorical fits.

The best hybrid model makes 480 validation errors. Its exact accuracy is 99.5927%
on 3,437 wall-event transitions, 99.2599% on 2,432 brick-event transitions, and
76.9585% on 1,519 paddle-event transitions. Events are overlapping native-rule
diagnostic labels, not neural inputs. The other hybrid seed makes 504 errors.
The seed-91 model fits all 359,086 training targets exactly; the selected seed-2026
checkpoint has 73 training errors. Paddle-hit accuracy is 100% and 99.1951% on
training but only 75.3785% and 76.9585% on validation. The remaining gap is therefore
not explained by absent source information on these checked samples.

### Irrelevant-input sensitivity

A source-only region with RAM y above 100 is too far from the bricks for a brick
contact within two native frames. Shuffling only the brick layout in this region
leaves the native next horizontal velocity unchanged on every checked example.
The best classifier nevertheless changes 310 of its 1,519 paddle-event predictions;
the other hybrid seed changes 333. The best model's paddle errors increase from
350 to 438 under the shuffle. This establishes sensitivity to irrelevant inputs
for this region. It is a counterfactual diagnostic, not ordinary held-out accuracy
or proof that every original error is caused by brick inputs.

Both timing witnesses remain wrong in the best-MSE neural checkpoint despite
having the correct reconstructed fraction. The other hybrid checkpoint gets one
right. Thus resolving state ambiguity and learning the correct use of that state
are separate achievements. The next focused investigation is paddle-collision
generalization and enforcing independence from distant brick layouts, rather than
adding dataset columns or increasing every loss weight.

These results are one-step predictions from recorded state and reconstructed
source memory. No full-game recursive accuracy is claimed; the model does not
predict its own collision memory or other game variables. Checkpoints are under
`runs/ball-vx-discrete-20260918/`. Reproduction scripts, memory checks, aggregate
comparisons, counterfactual results, and training/test logs are under
`logs/ball-vx-discrete-20260918/`. Frozen reload reproduces all six validation
reports exactly. Verification: **344 tests passed, two skipped**; Ruff,
`uv sync --frozen`, and `git diff --check` passed.

### Continued horizontal-velocity generalization experiments

After the user asked to continue, two matched interventions addressed the
irrelevant-brick sensitivity. Each uses the same two seeds, 100 full-data epochs,
optimizer, batch order, and validation selection as the hybrid controls. Neither
changes dataset records or writes reconstructed/augmented feature arrays.

First, for training sources with RAM y above 100, replace the brick inputs with
the layout from a randomly chosen training row. The ball cannot reach the bricks
within the two native frames in this region. A separate augmentation RNG keeps
sample order identical to the control. The native calculation confirms target
invariance on a fixed shuffle of all 238,495 training and 57,161 validation sources
in the region. Validation metrics always use the original layouts. Second, retain
that augmentation and add the scalar difference `ball_x - paddle_x` plus twelve
bits encoding it in eighth-pixel units. This relation uses existing coordinates,
not an additional state variable or a native collision branch. It changes the
input width from 182 to 195 and parameter count from 40,968 to 42,632; hidden
layers remain 128/128 with SiLU and the output remains eight velocity classes.

| Hybrid classifier variant | Seed | Best epoch | Validation errors / 88,590 | MAE | RMSE | Exact on paddle hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original layouts | 91 | 80 | 504 | 0.017186 | 0.249873 | 75.3785% |
| Original layouts | 2026 | 57 | 480 | 0.016407 | 0.243814 | 76.9585% |
| Irrelevant layouts varied during training | 91 | 98 | 383 | 0.012530 | 0.210823 | 81.6985% |
| Irrelevant layouts varied during training | 2026 | 94 | 388 | 0.012631 | 0.211398 | 81.1060% |
| Varied layouts plus relative offset | 91 | 82 | 296 | 0.009346 | 0.180460 | 85.4510% |
| Varied layouts plus relative offset | 2026 | 62 | 320 | 0.010170 | 0.189014 | 84.9243% |

Both interventions improve validation MAE and paddle-event accuracy in both
seeds. The final models are 99.6659% and 99.6388% exact overall, with training
errors four and 52 respectively. The best-MSE final checkpoint is
`runs/ball-vx-discrete-20260918/hybrid-classification-memory1-s91-brick-aug-relative/best.pt`.
It is 94.2588% exact when horizontal velocity changes, 99.5927% on wall events,
and 99.4243% on brick events. Its 221 paddle-event errors remain the largest
event-specific problem. These are one-step validation results, not final-test or
recursive-emulation accuracy.

The irrelevant-brick diagnostic changes only 31 and 37 paddle predictions after
augmentation, compared with 333 and 310 before it. The relative-offset variants
change 27 and 48. The best final model's paddle errors rise from 221 to 226 under
the counterfactual shuffle, so sensitivity is reduced but not eliminated.
Native-frame timing separates 284 paddle events on the first frame from 1,235 on
the second. The best model misses 37 and 184, respectively. Their error rates are
13.0% and 14.9%; the larger second-frame count mostly reflects its greater
frequency, rather than evidence of an exclusively second-frame failure. It still
misses the episode-173 timing witness and gets the episode-1552 witness correct.

A separate sufficiency check verifies that charge deterministically gives the
controller measurement used by the native audit on every checked source. Thus
the neural model does not need an unprovided measurement input to reproduce
that calculation. The remaining errors concern learning and generalization on
the checked states, not an identified need for new dataset columns.

All ten checkpoints reload with identical saved validation metrics. Results and
scripts are in the same diagnostic directory, including
`brick-augmentation-comparison.json`, `relative-comparison.json`,
`frozen-evaluation.json`, and `controller-sufficiency.json`. After adding the
relative encoding, all six focused model-contract tests pass, including joint
translation invariance of the offset and checkpoint compatibility. Ruff and
`git diff --check` remain clean. The earlier full run passed 344 tests with two
skipped; it preceded this optional encoding extension.


## 2026-09-18: Focused paddle-region velocity learning

The proposed focus on paddle approaches did not improve held-out horizontal
velocity accuracy in either seed. Removing brick layout and contact memory
reduced the network size but did not close the generalization gap. Dataset and
cache files were not changed.

Eligibility uses only current RAM ball y in [160, 183] and downward velocity,
within the existing nonterminal split. It retains every checked paddle collision:
6,336 in 30,204 training sources and 1,519 in 7,164 validation sources. The other
23,868 / 5,645 sources contain no paddle hit in the next two native frames.
Within these, 228 / 48 pass below the paddle without hitting it. Future event
labels are evaluation strata only. Life boundaries and the reserved test remain
unchanged. Native diagnostics verify the existing source memory and target
invariance after replacing bricks and contact memory in this region.

The compact model receives nine state values: ball x/y/vx/vy, paddle x/width,
charge, prior hit count, and fractional y. Hybrid encoding plus the derived
horizontal offset gives 85 inputs, two 128-unit SiLU layers, and eight output
classes, for 28,552 parameters. The full-input control has 195 encoded inputs
and 42,632 parameters, retaining train-only random brick-layout augmentation.
Neither neural model executes native collision rules.

An initial 100-epoch screen produced paddle accuracies of 77.55% / 78.21% for
full inputs and 80.45% / 80.91% for compact inputs. Training still had many errors.
However, each focused epoch has only 30 minibatches, versus 351 globally. The
screen therefore provided 3,000 updates, not the previous 35,100. Four fresh
fits stretched training and the cosine schedule to 1,170 epochs, giving 35,100
updates each. This matches optimizer updates, not unique-example coverage or
validation-checkpoint opportunities. All other optimizer settings and both
seeds were retained. Checkpoint selection uses minimum validation native MSE.

The previous models were rescored on exactly the same selected examples:

| Training / input | Seed | Selected train errors / 30,204 | Validation errors / 7,164 | Paddle accuracy / 1,519 | Paddle direction / speed / both errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous global / full | 91 | 4 | 252 | 85.45% | 179 / 59 / 17 |
| Previous global / full | 2026 | 52 | 277 | 84.92% | 190 / 55 / 16 |
| Focused / full | 91 | 74 | 305 | 84.13% | 196 / 60 / 15 |
| Focused / full | 2026 | 0 | 299 | 83.94% | 198 / 60 / 14 |
| Focused / compact | 91 | 181 | 304 | 83.61% | 202 / 60 / 13 |
| Focused / compact | 2026 | 0 | 307 | 83.15% | 206 / 65 / 15 |

Direction and speed error counts overlap; subtract the “both” count to obtain
the number of incorrect paddle predictions. Direction accounts for about 80%
of incorrect paddle predictions in all four focused fits. The compact seed-2026
checkpoint gets all 30,204 selected training examples correct but misses 256 of
1,519 held-out paddle hits. Final training cross-entropy is near zero in all
four runs, while the best validation checkpoints occur much earlier, at epochs
164 / 469 for full inputs and 115 / 247 for compact inputs. More repetition of
these examples does not solve the held-out errors under this training setup.

The compact seed-2026 model has slightly lower selected-region velocity MSE
than compact seed 91, despite fewer exact matches. Its paddle direction accuracy
is 86.44% and absolute-speed accuracy is 95.72%. Among 692 fast paddle hits it
makes 103 mistakes, all direction errors. Native-frame breakdowns do not show a
consistent failure restricted to second-frame collisions across seeds. These
results support isolating direction prediction next, with the same sufficient
inputs, before increasing model size or collecting new fields. They do not
identify a single architectural cause or prove every remaining input necessary.
The previous global model remains the better predictor on this validation set.

Artifacts are under `logs/ball-vx-paddle-focus-20260918/` and
`runs/ball-vx-paddle-focus-20260918/`: both drivers, selection receipts, initial
and update-matched comparisons, global baselines, per-epoch logs, registry
checkpoints, and largest-error source examples. All eight new checkpoints reload
with identical saved validation metrics. The nine focused model-contract tests
pass; the full suite passes 348 tests with two skipped. Ruff and whitespace checks
pass. These results concern one-step velocity from recorded source state only;
they do not establish a full-game or recursive emulator.


## 2026-09-18: Isolating next horizontal direction

Direction-only learning gives a small, consistent improvement in this two-seed
validation experiment. Both direction models get 87.89% of the 1,519 paddle
bounce directions right, compared with 86.77% for both matched eight-class
controls. This removes 17 of 201 paddle direction errors per seed, a 1.12
percentage-point accuracy increase. Full velocity accuracy is not measured for
the binary model because it does not predict speed.

The experiment retains the focused source eligibility, all 30,204 training and
7,164 validation sources, nine current-state inputs, hybrid encoding and derived
horizontal offset, two 128-unit SiLU layers, seeds 91 and 2026, minibatch order,
AdamW settings, and 35,100 updates. Only the output and cross-entropy targets
change: eight signed speeds versus two directions. The binary network has 27,778
parameters; the eight-class network has 28,552. The two hidden layers have the
same initialization for a given seed. No native transition rule runs in either
predictor. Source-index hashes match the preceding experiment.

Both fresh controls and direction-only models select by minimum direction errors
on all selected validation sources, keeping the earliest tie. This avoids
comparing a direction-selected binary model only against velocity-MSE-selected
controls. All 1,170 training losses for each eight-class control exactly reproduce
its previous compact run, confirming the control training itself is unchanged.
A second eight-class decoding sums probabilities for the four speeds in each
direction and selects its own checkpoint using the same direction-error criterion.
Summing probabilities does not improve paddle accuracy in these runs.

| Model / decoding | Seed | Training direction errors / 30,204 | Validation direction errors / 7,164 | Paddle direction errors / 1,519 | Paddle direction accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous MSE-selected eight-class | 91 | 166 | 249 | 202 | 86.70% |
| Previous MSE-selected eight-class | 2026 | 0 | 243 | 206 | 86.44% |
| Eight-class, best class | 91 | 0 | 248 | 201 | 86.77% |
| Eight-class, summed directions | 91 | 276 | 247 | 210 | 86.18% |
| Eight-class, best class | 2026 | 74 | 237 | 201 | 86.77% |
| Eight-class, summed directions | 2026 | 74 | 237 | 201 | 86.77% |
| Direction-only | 91 | 0 | 222 | 184 | 87.89% |
| Direction-only | 2026 | 0 | 228 | 184 | 87.89% |

Both selected direction models fit every training source correctly. Their
selected epochs are 242 and 485; continuing to epoch 1,170 leaves 188 and 191
paddle direction errors, versus 184 at the selected checkpoints. Thus more
repetition has not eliminated the generalization gap. The selected direction
models make 222 and 228 errors across all validation approaches. Preserving the
incoming direction without a learned update makes 917 overall errors and 834
paddle errors, giving only 45.10% paddle direction accuracy.

The improvement is not a uniform correction of the eight-class model. For seed
91, the binary model fixes 88 paddle errors and introduces 71; 113 errors are
shared. For seed 2026 it fixes 81, introduces 64, and shares 120. The eight-class
objective contributes modestly to error under this setup, but isolating direction
leaves roughly 12% of paddle cases wrong. Input coverage and how the network
learns the contact geometry remain candidates for the remaining gap. This
experiment does not establish either as the cause or justify adding hidden
fields. A useful next diagnostic is training coverage near the mistaken paddle
configurations, followed by a controlled learning curve using more existing
training examples if coverage is sparse.

All results use the existing validation episodes and selected checkpoints, not
a fresh test set; two seeds do not establish broad statistical significance.
No recursive simulation was evaluated. The compact nine inputs remain ball
x/y/vx/vy, paddle x/width, charge, prior paddle-hit count, and fractional y.
Dataset and cache files were not modified.

Artifacts: `logs/ball-direction-20260918/train.py`, `comparison.json`,
`baselines.json`, `paired.json`, `report.py`, `train.log`, and this report;
checkpoints, per-epoch metrics, final-epoch evaluations, and all validation error
examples are under `runs/ball-direction-20260918/`. Each saved checkpoint reloads
with identical validation metrics, including its explicit decoding mode. The
full test suite passes 350 tests with two skipped; all 11 focused model-contract
tests pass, as do Ruff and whitespace checks. README, training instructions, and
model-extension documentation describe the new direction output contract.


## 2026-09-18: Coverage and boundaries in direction errors

Errors are associated with both weaker training coverage and sharp local
changes in the true next direction. The coverage association survives excluding
exact training-state matches and appears in all three distance representations.
It does not explain all the mistakes or prove that more data will fix them.
No new input variable was identified, and no dataset or cache was modified.

The audit freezes both direction-only checkpoints and reproduces their saved
validation metrics exactly: 184 errors each among 1,519 paddle hits. All 196
validation paddle sources with an exact nine-variable training match are correct
in both models. The remaining 1,323 unseen sources have 184 errors each, or
13.91%. No complete source-state key has conflicting direction labels within
training or across the matched training/validation rows. Neighboring states with
different labels are not evidence of missing inputs, because those states differ.

Density uses the distance to the 25th nearest training source. Its reference
threshold is the training 75th percentile, measured on 2,048 seeded training
paddle-hit anchors while excluding each anchor's entire recorded episode.
Neighbors come from all 30,204 selected training approaches, including misses.
The principal geometric representation preserves all source information through
relative ball/paddle x, integer plus fractional y, vx, vy, paddle x/width, charge,
and prior hit count. Every dimension is standardized using training inputs only.
Raw nine-state standardization and the existing 85-value hybrid model encoding
provide sensitivity checks; 1st/5th-neighbor radii are also retained in the report.

Among unseen paddle sources, the geometric definition marks 330 as sparse and
993 as better-covered. Error rates are 17.88% / 18.48% in the sparse group versus
12.59% / 12.39% elsewhere, for seeds 91 / 2026. Sparse sources contain only
59 / 61 of the 184 errors, about one third. The other distance representations
also show higher error rates with weaker coverage; none puts a majority of the
errors in its sparse group. Descriptive episode-bootstrap intervals for the
unseen geometric sparse-minus-dense difference are +0.78 to +9.88 and +1.53 to
+10.78 percentage points. These are associations on development validation data,
not independent-test confidence bounds or causal estimates.

A separate native-rule probe shifts current ball x by every eighth pixel up to
one pixel in either direction, holding other source variables fixed. It marks
cases where any such shift changes next-vx sign. This includes contact/miss
changes and wall interactions as well as the paddle's direction-selection rule.
It uses the pinned native source and reproduces the original target before
perturbation. The perturbed states are diagnostic counterfactuals; their physical
reachability is not asserted and they are never fed into training.

The one-pixel probe marks 303 unseen paddle sources. The models miss 81 / 77,
or 26.73% / 25.41%, compared with 103 / 107 errors among 1,020 other unseen
sources, or 10.10% / 10.49%. Crossing the coverage and boundary groups gives:

| Coverage and local direction behavior | Unseen validation paddle states | Seed 91 errors | Seed 2026 errors |
| --- | ---: | ---: | ---: |
| Better coverage, no direction flip within 1 px | 759 | 67 / 8.83% | 70 / 9.22% |
| Sparse coverage, no direction flip within 1 px | 261 | 36 / 13.79% | 37 / 14.18% |
| Better coverage, direction can flip within 1 px | 234 | 58 / 24.79% | 53 / 22.65% |
| Sparse coverage, direction can flip within 1 px | 69 | 23 / 33.33% | 24 / 34.78% |

The same pattern appears before removing exact matches. Scans at 1/8, 1/2,
and two pixels are retained as sensitivity checks. Coarse geometry cells omit
charge and hit count deliberately, so they do not establish complete-state
coverage: 52 validation paddle sources have no matching coarse training cell,
but only 9 / 10 model errors occur there. Cells with at least 20 training examples
still contain 103 / 98 errors. Thus a blanket explanation of unseen collision
geometry is insufficient.

As a diagnostic, copying the nearest training direction in standardized relative
geometry gets 87.16% of all 1,519 paddle cases right, versus the neural models'
87.89%. Five- and 25-neighbor voting are worse. Raw-state and hybrid-encoding
neighbor copying are also worse. These are fixed descriptive probes, not a
selected replacement model. Mixed-direction neighborhoods are common, and do
not imply contradictory labels for identical state inputs.

A concrete error in both neural models is episode 1103, step 1080. Its source
is `[102, 175, -2, 2, 92, 16, 1213, 12, 0]`, with target direction -1. The nearest
geometric training source is `[102, 175, -2, 2, 93, 16, 1211, 12, 0]` at episode
2283, step 989, also with direction -1. Its coarse geometry cell has 44 training
examples, and a one-pixel horizontal ball perturbation does not change its native
direction. This is an example of error despite nearby supporting data, not a
missing-state or sparse-only witness.

The next useful intervention is a controlled training-data learning curve with
more training episodes, fixed validation episodes and state inputs, and matched
optimizer-update budgets. Report the sparse/boundary groups separately and fit
all sampling choices from training data. This would test whether greater
coverage causes an accuracy improvement. The present audit only establishes
associations; it does not warrant adding dataset columns or declaring the
remaining gap solved.

Artifacts are `logs/ball-direction-coverage-20260918/audit.py`, `unseen.py`,
`summary.json`, `unseen-summary.json`, `examples.json`, and the run logs. The
examples file records every paddle source, both model errors, nearest training
witnesses, support counts, and native-boundary flags. Source-index hashes match
the earlier experiment. Float32 distance cancellation was caught by a direct-norm
assertion; the completed audit uses float64 distances and passes that check.
There is no new production code or model fit in this audit. Repository Ruff and
whitespace checks pass. The reserved test remains unused.


## 2026-09-18: Direction learning curve with more recorded episodes

More training episodes substantially improve next-direction accuracy with the
same model and optimizer-update budget. Paddle-direction accuracy rises from
87.95–88.02% with 128 episodes, to 92.10–93.55% with 256, to 95.79–95.98% with
512. The 512-episode models make 64 and 61 paddle errors versus 182 and 183 for
the corresponding 128-episode controls, reductions of 64.8% and 66.7%.
This is evidence that training coverage was a major contributor to the previous
gap under this experimental setup. It is not a complete velocity or rollout result.

All six fits use the same compact direction model: nine source variables, 85
hybrid/relative encoded inputs, two 128-unit SiLU layers, two direction logits,
and 27,778 parameters. They use the same two seeds, AdamW settings, cosine
schedule, gradient clipping, and 35,100 updates. Every batch has exactly 1,024
examples, for 35,942,400 example presentations per fit. Shuffled full passes
continue across batch boundaries. Larger datasets receive fewer repetitions per
example. Validation occurs every 30 updates; the checkpoint with the fewest
errors across all 7,164 fixed validation approaches wins, with earliest ties.
Fresh 128-episode controls account for the change from the preceding approximately
1,007-example batches. The larger runs do not receive more updates or validation
selection opportunities.

The existing episode order from split seed 917 defines nested training sets.
The original 128 training and 32 validation episode IDs are reproduced exactly;
the additional episodes come only from the original dataset's training partition.
The original held-out partition and reserved test records remain unread. The
three sizes contain 30,204 / 61,179 / 123,563 selected source states and
6,336 / 13,007 / 26,157 paddle hits. Source eligibility remains descending ball
with RAM y between 160 and 183, and includes approaches with no hit. Training
state segmentation retains the existing life-loss boundaries.

The existing controller-annotated dataset supplies source charge and the other
controller values used only for native diagnostics. Derived collision memory
and hit count remain in RAM. Native replay exactly checks all 1,481,303 eligible
nonterminal transitions across the 512 episodes, including four ball outputs
and brick layout, with zero source hit-count mismatches. The old 128 episodes'
focused model inputs, targets, events, episode IDs, and steps exactly match the
previous experiment. No dataset or feature-cache files were created or changed.
Native rules are not part of neural inference.

| Training episodes | Paddle-hit training examples | Seed | Training errors / all selected sources | Validation direction errors / 7,164 | Paddle accuracy / 1,519 | Old sparse accuracy / 347 | Boundary accuracy / 349 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 6,336 | 91 | 0 / 30,204 | 226 | 88.02% | 83.86% | 77.65% |
| 128 | 6,336 | 2026 | 1 / 30,204 | 230 | 87.95% | 83.29% | 78.51% |
| 256 | 13,007 | 91 | 0 / 61,179 | 130 | 93.55% | 88.76% | 85.96% |
| 256 | 13,007 | 2026 | 256 / 61,179 | 148 | 92.10% | 85.59% | 83.95% |
| 512 | 26,157 | 91 | 180 / 123,563 | 83 | 95.79% | 90.49% | 89.40% |
| 512 | 26,157 | 2026 | 4 / 123,563 | 80 | 95.98% | 90.49% | 90.26% |

All scores use the same validation targets. Sparse and boundary groups retain
the previous audit's membership rather than being recomputed to favor larger
runs. In the 347 previously sparse cases, accuracy improves from 83.29–83.86%
to 90.49% in both 512-episode fits. Among the 349 cases where a source ball-x
perturbation of at most one pixel can change native direction, accuracy improves
from 77.65–78.51% to 89.40–90.26%. The remaining errors therefore still
concentrate in the harder groups.

The 1,323 validation states without an exact match in the original 128-episode
training set improve from 86.17–86.24% to 95.16–95.39%. This group is defined
relative to the original small training set; it is not proof that every state
remains unseen after expansion. The experiment does not separate gains from
new exact state coverage versus improved interpolation.

Selected checkpoint updates are 18,240 / 5,520 for 128 episodes, 24,690 / 4,800
for 256, and 7,530 / 14,160 for 512. Training errors in the table refer to those
selected checkpoints, not necessarily the final update. Final-update paddle
accuracy also improves with dataset size: 87.36–87.62%, 91.64–92.96%, and
95.39–95.66%. Thus the size trend is not limited to the best-checkpoint selection.

The primary conclusion is specific to this nested episode order, these two
initialization seeds, and the development validation set. It supports testing
further training-set expansion, keeping the validation partition and input
contract fixed. It does not establish that all missing accuracy is a data issue,
that the nine inputs are minimal, or that the model can run recursively. Speed
is not an output in this experiment.

Artifacts are `logs/ball-direction-scale-20260918/prepare.py`, `train.py`,
`split-plan.json`, `expanded-preparation.json`, `comparison.json`, `summarize.py`,
and `train.log`; every checkpoint, configuration, per-update-block log, and
validation error list is under `runs/ball-direction-scale-20260918/`. All six
saved checkpoints reload with identical validation metrics. Split-disjointness,
old-input equality, native successor parity, and matched budget assertions pass.
Repository Ruff and whitespace checks pass. No production model or player code
was changed for this experiment.


## 2026-09-18: Near-perfect direction development accuracy and reserved test

Expanding to all 1,824 available training episodes and improving the learned
geometry representation reaches zero errors on the fixed development validation
set. A model frozen before reading the reserved test reaches 99.9380% on its
3,226 paddle hits and 99.9602% across its 15,069 selected approaches. This is
short of the requested approximately 99.99% independent paddle accuracy. The
result concerns next-horizontal-direction only, not speed, full-game state,
image reconstruction, or recursive play.

Preparation retains life boundaries and all later-life segments. It selects
440,833 training approaches, including 93,389 paddle hits, using the same
source-only descending-ball region as before. Native replay exactly checks
5,271,950 nonterminal training transitions. The old 128-episode inputs and targets
remain identical, and validation remains the same 32 episodes, 7,164 approaches,
and 1,519 paddle hits. Dataset files and feature caches remain unchanged. Derived
features and auxiliary targets exist only in RAM. The prepared arrays were
streamed over SSH to the user's configured RTX 4090 host because local memory
pressure made CPU training slow. No packages were installed.

The full-data original direction architecture reaches 10 validation paddle errors,
99.3417%, with seed 91. The second CPU baseline was interrupted after 33,990
updates; its recovered best checkpoint has 13 paddle errors. Its interrupted
status is recorded, so it is not a matched completed run. A partial spatial CPU
run was also recovered before moving the subsequent fits to the GPU.

The new `ball_direction_geometry` model first learns the paddle's displacement
during the intermediate native frame. Its inputs are current paddle position
and charge. The auxiliary label is derived from the recorded current controller
measurement using the audited native rule. This is additional native-derived
supervision, not an added dataset column or an inference-time rule. Training
uses 2,895 distinct observed position/charge configurations and 13 displacement
classes. The 25→128→128→13 SiLU classifier reaches zero errors on those training
configurations and all 7,164 validation sources after 368 epochs. Its 21,517
parameters are then frozen, included in every geometry checkpoint, and kept in
evaluation mode during direction training.

The direction network sees the same nine source variables through geometric
encodings. It gets relative ball/paddle coordinates, constant-velocity motion
proposals, and the learned intermediate paddle position. No collision branch or
controller update runs inside inference. Scalar encoding has 16 inputs; hybrid
adds relative-coordinate and y-position bits for 62 inputs. The final 84-input
encoding also exposes binary digits of absolute ball x and its motion proposal.
This addresses a validation finding: all three errors of the selected wide
62-input model were near a side wall. The actual output remains two learned
left/right logits.

| GPU variant | Hidden layers | Seed | Validation errors / 7,164 | Paddle errors / 1,519 | Training errors / 440,833 |
| --- | --- | ---: | ---: | ---: | ---: |
| Original hybrid plus spatial features | 2 × 128 SiLU | 91 | 19 | 16 | 126 |
| Original hybrid plus spatial features | 2 × 128 SiLU | 2026 | 12 | 10 | 3 |
| Learned paddle, scalar geometry | 2 × 128 ReLU | 91 | 11 | 4 | 1,064 |
| Learned paddle, scalar geometry | 2 × 128 ReLU | 2026 | 10 | 6 | 933 |
| Learned paddle, hybrid geometry | 2 × 128 ReLU | 91 | 6 | 4 | 150 |
| Learned paddle, hybrid geometry | 2 × 128 ReLU | 2026 | 10 | 7 | 360 |
| Learned paddle, wide hybrid geometry | 3 × 256 ReLU | 91 | 4 | 1 | 191 |
| Learned paddle, wide hybrid geometry | 3 × 256 ReLU | 2026 | 3 | 2 | 195 |
| Learned paddle, absolute-position bits | 2 × 128 ReLU | 91 | 1 | 1 | 14 |
| Learned paddle, absolute-position bits | 2 × 128 ReLU | 2026 | 1 | 1 | 29 |
| Learned paddle, wide absolute-position bits | 3 × 256 ReLU | 91 | 0 | 0 | 30 |
| Learned paddle, wide absolute-position bits | 3 × 256 ReLU | 2026 | 0 | 0 | 6 |

The base GPU fits use 35,100 updates and the wider fits allow 105,300, with batches
of 1,024. AdamW starts at 0.001 with weight decay 0.0001, cosine decay to 0.00005,
and gradient clipping at 5. Validation is checked every 30 updates and selects
the earliest minimum-error checkpoint. Fitting stops when all validation errors
reach zero. The two final wide fits stop at 15,720 and 7,440 updates. They each
contain 153,858 trainable direction parameters plus the frozen 21,517 paddle
parameters, 175,375 total. These are adaptive development experiments with
unequal budgets, not a controlled attribution of every gain to one change.

Three additional fine-tunes use half uniform samples and half from the highest
5% of training losses, refreshed every 600 updates. None improves its parent's
best validation error count. An equal-probability ensemble of the first six fits
has four validation errors. An ensemble of the four absolute-position fits is
also considered, with no validation-fitted combination weights. The frozen
selection prefers a single model on ties, then smaller models, and retains the
first fit on remaining ties. It chooses wide absolute-position seed 91. Seed
2026 is not selected using its training error count or test results.

Before reading test trajectories, `frozen-selection.json` records the selected
checkpoint hash and selection rule. The reserved 64 test episodes are disjoint
from every training and development validation episode. Their existing state
cache and recorded controller annotations are read without writing new caches.
Source collision memory is reconstructed by the same forward replay method;
recorded successor states check parity and never choose current hidden values.

| Frozen single model evaluation | Paddle direction | All selected directions |
| --- | ---: | ---: |
| Development validation | 1,519 / 1,519, 100% | 7,164 / 7,164, 100% |
| Reserved test | 3,224 / 3,226, 99.9380% | 15,063 / 15,069, 99.9602% |

The reserved test has zero errors on 760 first-native-frame paddle hits and two
on 2,466 second-frame hits. Four errors are on no-hit approaches, including two
of the 111 trajectories that pass the paddle. The intermediate paddle predictor
has two test errors, but neither overlaps a direction error. A post-test audit
replaces its prediction with the correct intermediate paddle position and still
finds all six direction errors. This diagnostic is not a model improvement or a
reported learned-model score. No fitting or checkpoint selection follows the
test result. Future use of these test examples to develop another model must
acknowledge that they have now been inspected.

Zero validation errors do not establish 99.99% population accuracy, especially
after repeated validation-based selection. The independent test provides the
stronger estimate, and still contains rare mistakes. The nine source values are
ball x, integer RAM y, vx, vy, paddle x, paddle width, charge, prior paddle-hit
count, and fractional y. Some are reconstructed diagnostic inputs. Their joint
recursive prediction remains untested.

Drivers, preparation receipts, transfer hash, and local reload checks are under
`logs/ball-direction-highaccuracy-20260918/`. CPU runs are under
`runs/ball-direction-highaccuracy-20260918/`; all GPU checkpoints, configs,
metrics, error witnesses, the frozen selection, and independent test reports
are under its `gpu-artifacts/` directory. Each selected GPU checkpoint is
reloaded before scoring. The final selected checkpoint is also checked through
the production model registry on CPU. The direct image baseline and player
behavior are unchanged.


Verification: the complete local suite reports 358 passed, two skipped, and one
90-second subprocess timeout in the separate brick-dataset augmentation test.
That timeout repeats in isolation under local memory pressure. All 21 focused
ball-model checks pass, including all three geometry encodings, frozen paddle
weights, memory masking, and checkpoint round trips. A remote attempt to check
the brick test cannot collect because the existing training environment lacks
PyArrow; no packages were installed to work around it. Ruff and whitespace
checks pass. CPU reload reproduces zero validation errors and six test errors,
matching CUDA despite the two environments using different PyTorch versions.
`runs/ball-direction-highaccuracy-20260918/result.json` ties together the dataset
manifest hash, preparation and transfer receipts, frozen selection, test result,
and verification limitations.


## 2026-09-18: Learned horizontal speed with frozen direction

Trained a separate speed classifier and combined it with the previously frozen
near-paddle direction predictor. On 64 fresh held-out episodes, speed is exact
on 3,255 of 3,263 paddle hits, **99.7548%**. Combined signed horizontal velocity
is exact on 3,254 of those hits, **99.7242%**, and on 15,336 of all 15,346 selected
approaches, **99.9348%**. These are recorded-source, one-step, descending
near-paddle results. Full-game integration and recursive prediction remain
untested. The approximately 99.99% validation aggregate does not transfer to
paddle-hit test accuracy.

The registered `ball_horizontal_velocity` stores a complete frozen
`ball_direction_geometry` and a trainable four-class magnitude head. The four
magnitudes, 0.5, 1, 1.5, and 2 native pixels per native frame, are checked against
the training vocabulary. Prediction multiplies the frozen learned direction by
the learned magnitude. The same nine source values and 84 encoded geometry
features are used. Neither direction nor its intermediate-paddle network is
fine-tuned. Training snapshots and all selected checkpoint reloads assert exact
frozen-weight equality. All inference weights are stored together; loading does
not require the parent checkpoint file.

The parent direction checkpoint is the previously selected wide absolute-position
seed-91 model, SHA-256
`78e7b3c0105214e8011bff2a1977ebbe4384cc68ca394edda3b1c7e3fa296ce8`.
Its validation directions remain exact. Consequently, validation speed and
combined-velocity error counts are identical in this experiment. Both networks
still rely on reconstructed current collision memory; those memory updates are
not learned here.

Preparation reproduces the earlier 1,824 training episodes, 440,833 selected
training approaches, 93,389 paddle hits, and fixed 32-episode validation set.
Episode memberships and source-index hashes match the direction experiment.
Native successor checks pass on all 5,271,950 nonterminal training transitions.
The dataset and feature caches are unchanged. Source arrays remain in RAM and
are streamed to the existing RTX 4090 environment; encoded features are cached
only in GPU memory. No dependencies are added.

Two initial 84→128→128→4 ReLU classifiers each have 27,908 trainable parameters
and receive 35,100 updates. Because both retain validation errors, two wider
84→256→256→256→4 classifiers receive up to 105,300 updates. The wider speed head
has 154,372 trainable parameters. Together with the frozen 175,375 direction and
intermediate-paddle parameters, its standalone checkpoint contains 329,747
parameters. These adaptive fits change both width and update budget; the result
is not an isolated estimate of the benefit of additional layers.

All fits use cross-entropy, batches of 1,024, AdamW at 0.001 with weight decay
0.0001, cosine decay to 0.00005, and gradient clipping at 5. Shuffled passes
continue across batch boundaries. Every 30 updates, validation exact speed
errors select the checkpoint, retaining the earliest tie. Zero validation errors
would stop a fit; none reaches zero. No speed-change labels or held-out errors
influence training sampling.

| Speed head | Seed | Selected update | Training speed errors / 440,833 | Validation speed errors / 7,164 | Paddle errors / 1,519 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 × 128 ReLU | 91 | 7,410 | 103 | 2 | 1 |
| 2 × 128 ReLU | 2026 | 22,470 | 1 | 2 | 1 |
| 3 × 256 ReLU | 91 | 30,120 | 58 | 1 | 1 |
| 3 × 256 ReLU | 2026 | 9,120 | 58 | 2 | 1 |

The selected wide seed-91 model scores 99.9860% on all validation approaches and
99.9342% on validation paddle hits. It correctly predicts 171 of 172 actual
validation speed changes. Zero errors on 6,992 unchanged-speed validation cases
show that it does not pay for that event accuracy with false speed changes on
this split.

Before any new test transitions are read, `plan_holdout.py` reserves 64 episodes
from the remaining 400 original held-out episodes, excluding the previously
inspected 64. A fixed seed, 190918, selects IDs using episode metadata only.
`freeze.py` then records the selected model hash before test preparation. The
fresh test contains 185,949 native-verified nonterminal transitions, of which
15,346 meet the same descending near-paddle source selection. All 3,263 native
paddle hits are retained. No model fitting or selection follows this test.

| Fresh test group | Samples | Speed errors | Direction errors | Combined vx errors | Combined exact accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| All selected approaches | 15,346 | 8 | 3 | 10 | 99.9348% |
| Paddle hits | 3,263 | 8 | 2 | 9 | 99.7242% |
| No paddle hit | 12,083 | 0 | 1 | 1 | 99.9917% |
| Actual speed changes | 388 | 4 | 1 | 4 | 98.9691% |
| Unchanged speed | 14,958 | 4 | 2 | 6 | 99.9599% |
| Fast paddle hits | 1,542 | 0 | 0 | 0 | 100% |

One error affects both speed and direction, so their counts cannot simply be
added. Four speed errors miss an actual change; four incorrectly change a speed
that should stay the same. All eight speed errors occur on paddle hits. Keeping
the incoming magnitude with the same frozen direction model makes 389 paddle
velocity errors, versus nine for the learned combination. That baseline misses
all 388 speed changes; the learned speed model handles 384 correctly. Its gain
therefore is not explained by copying unchanged speed.

The frozen intermediate-paddle predictor makes two fresh-test errors. Neither
overlaps a speed or direction error. Supplying the correct intermediate paddle
position in a post-test attribution audit leaves eight speed and three direction
errors. This is privileged-input diagnosis, not a learned-model improvement or
a new selection criterion. The remaining errors belong to the learned speed
and direction decisions on these tested examples.

Artifacts are under `logs/ball-speed-20260918/` and
`runs/ball-speed-20260918/gpu-artifacts/`. The selected combined checkpoint is
`speed-w256-s91/best.pt`, SHA-256
`bfaf12fe519e03a9b8045d5a33cbdee53d15ab193240bb134889ac571ce01230`.
`result.json` links preparation and transfer receipts, dataset identity, the
fresh held-out episode plan, frozen selection, test metrics, and CPU reload.
The full suite passes **364 tests, two skipped**; the focused frozen-model and
checkpoint checks pass. `uv sync --frozen`, Ruff, and whitespace checks pass.
The direct image baseline, shared runner, dataset schema, and player are unchanged.

## 2026-09-18: Full-game vx integration with a frozen paddle model

Combined the selected near-paddle direction/speed model with the existing
full-field vx classifier. The source-only gate is integer RAM y in [160, 183]
and vy > 0. Event labels are used for evaluation, never for routing. The paddle
component remains byte-for-byte unchanged. The complete registered model is
`ball_horizontal_router`: 372,379 parameters, of which only the full-field
195→128→128→8 SiLU network's 42,632 parameters were updated.

Simply routing the existing models reduced validation errors from 296 to 45
out of 88,590 transitions. Outside the paddle gate, the old model had zero
training errors but 44 validation errors: 25 ordinary-flight, five wall, and
14 brick errors. That motivated a matched coverage experiment. Controls and
expanded fits used the same starting weights, seeds 91/2026, 30,000 updates,
batch size 2,048, and optimizer/augmentation settings. Control data contained
328,882 outside-gate sources from 128 episodes. Expanded data contained
4,831,117 outside-gate sources from all 1,824 training episodes. Preparing those
sources checked all 5,271,950 nonterminal training transitions against native
next-state and brick-layout outcomes. Derived inputs stayed in RAM; dataset
files and columns were unchanged.

| Fit | Outside-gate validation errors / 81,426 | Total validation errors / 88,590 | Selected update |
| --- | ---: | ---: | ---: |
| Routed existing models | 44 | 45 | Parent |
| Original-data fine-tune, seed 91 | 31 | 32 | 14,400 |
| Original-data fine-tune, seed 2026 | 35 | 36 | 2,700 |
| Expanded-data fine-tune, seed 91 | **15** | **16** | **20,400** |
| Expanded-data fine-tune, seed 2026 | 18 | 19 | 18,600 |

The expanded seed-91 model was selected and hashed before reading 64 previously
unused held-out episodes. Seed 290918 selected their IDs from metadata, excluding
both earlier 64-episode test sets. These episodes supplied 185,517 nonterminal
transitions, all exactly matched by the input-reconstruction/native audit.
No weights or model choices changed after test inspection.

| Frozen model on the same fresh test | Total vx errors / 185,517 | Outside-gate errors / 169,994 |
| --- | ---: | ---: |
| Old full-field model | 751 | 104 |
| Routed existing models | 119 | 104 |
| Best original-data fine-tune | 84 | 69 |
| Selected expanded-data fine-tune | **50** | **35** |

| Fresh-test group | Samples | Errors | Exact vx accuracy |
| --- | ---: | ---: | ---: |
| All transitions | 185,517 | 50 | **99.9730%** |
| Ordinary flight | 170,515 | 16 | 99.9906% |
| Wall contacts | 6,931 | 4 | 99.9423% |
| Brick contacts | 5,056 | 21 | 99.5847% |
| Paddle contacts | 3,203 | 10 | 99.6878% |
| Actual vx changes | 6,387 | 32 | 99.4990% |

Collision categories overlap. The frozen near-paddle region contributes 15
errors / 15,523 approaches on this new test, including ten paddle-hit errors.
Its results differ from the speed-stage test because the episode set changed.

The dominant remaining brick failure is missed acceleration: all 21 brick
errors preserve incoming magnitude when the target magnitude is 2. Their signs
are correct. Only **31 / 52 actual brick-triggered accelerations are predicted
correctly (59.6154%)**. Validation likewise has nine missed accelerations among
29, plus one other brick error. Thus high aggregate accuracy still conceals a
weak rare transition. A useful next experiment is isolated brick-speed-change
learning, with recall and false speed changes reported explicitly. The current
input audit found no mismatch requiring additional dataset fields under this
fixed transition contract; it does not prove generalization to arbitrary states.

The selected checkpoint is
`runs/ball-vx-router-20260918/global-finetune-s91/best.pt`, SHA256
`8de3dbf0987642cd29ff969a7e6950443d2a6c5e1455019a0d26e7dccf306591`.
`result.json`, `provenance.json`, and `remaining-error-audit.json` beside the run
record metrics, episode identities, preparation receipts, freeze records, source
hashes, and the acceleration breakdown. CUDA and local CPU reloads reproduce the
same exact validation/test metrics; all frozen paddle weights match their parent.
The full suite passed 367 tests with two skips, plus targeted router tests after
a dtype-preservation correction; Ruff, frozen dependency sync, and whitespace
checks passed. This remains one-step vx prediction with supplied current memory,
not a complete recursive state emulator.

## 2026-09-18: Shared spatial brick features resolve most missed accelerations

The preceding vx integration missed 21 of 52 brick-triggered horizontal speed
increases on its reserved test. A new experiment freezes that entire predictor
and isolates the keep/increase decision for source RAM y <= 100 and |vx| < 2.
Its observed outcomes are the incoming magnitude or 2. The learned decision
sets magnitude; the frozen parent supplies sign and all predictions outside the
domain. Native rules audit inputs and outcomes offline but never decide the
learned model's output.

The six numeric inputs are current ball x, RAM y, vx, vy, fractional y, and
brick-contact memory, alongside the 108 brick cells. Three interventions on
paddle/controller values preserve all 5,117 eligible validation vx targets.
Training/validation have 178,725 distinct reduced input vectors and zero groups
with conflicting acceleration labels. All 5,271,950 nonterminal training
transitions still pass source reconstruction/native outcome checks. Filtering
all 1,824 training episodes yields 304,107 eligible sources and 1,694 positives;
validation has 5,117 sources and 29 positives. Dataset columns remain unchanged,
and derived arrays are retained only in RAM.

Flat heads use scalar/binary features; geometry variants add current and
constant-velocity proposed coordinates. The spatial model instead applies one
shared learned network to each cell's geometry relative to the ball, masks absent
bricks, and max pools its features. Grid centers are an explicit layout prior;
there are no hand-written overlap or acceleration decisions. All heads use
cross-entropy and validation selection on speed errors, breaking ties by missed
accelerations. Uniform and 25%-positive sampling are compared with two seeds.

| Fit | Training speed errors at selected checkpoint | Validation misses / 29 | Validation false accelerations | Total validation speed errors |
| --- | ---: | ---: | ---: | ---: |
| uniform-s91 | 17 | 8 | 7 | 15 |
| balanced-s91 | 2 | 8 | 11 | 19 |
| uniform-s2026 | 65 | 8 | 6 | 14 |
| balanced-s2026 | 31 | 7 | 10 | 17 |
| geometry-uniform-s91 | 4 | 6 | 6 | 12 |
| geometry-balanced-s91 | 4 | 4 | 5 | 9 |
| geometry-uniform-s2026 | 57 | 7 | 1 | 8 |
| geometry-balanced-s2026 | 7 | 5 | 5 | 10 |
| spatial-balanced-s91 | 56 | 1 | 1 | 2 |
| spatial-balanced-s2026 | 90 | 1 | 2 | 3 |
| spatial-uniform-s91 | 130 | 1 | 1 | 2 |
| spatial-wide-s91 | 68 | 1 | 1 | 2 |
| spatial-uniform-s2026 | 77 | 1 | 1 | 2 |
| spatial-wide-s2026 | 4 | 2 | 0 | 2 |

The raw heads receive 149 features and have two 128-unit ReLU hidden layers;
geometry heads receive 179. Both use 30,000 updates and batches of 1,024. Small
spatial fits use 13→64→64 shared cell layers, occupancy-masked max pooling, and
70→128→2 global layers; they receive only 15,000 updates with batches of 512.
Wide spatial fits use three shared 128-unit layers and 30,000 updates. All use
AdamW 0.001, weight decay 0.0001, cosine decay to 0.00002, and gradient clipping
at 5. Every 100 updates evaluates the fixed validation set. The selected small
balanced seed-91 checkpoint is from update 6,900: 14,402 learned parameters
around a frozen 372,379-parameter parent, totaling 386,781.

Extra positive sampling alone did not resolve the raw-vector model's error gap.
Both sampling strategies work much better with shared spatial features; widening
the spatial model does not improve the best validation error count. This supports
representation of ball/brick relationships as the useful intervention here.
It does not attribute every gain to one factor: spatial features, weight sharing,
architecture, and training budget differ from the flat controls.

The SSH log stream disconnected during the final sweep. Recovered summaries,
checkpoints, comparison file, and the pre-test selection record show that all
14 fits and selection completed remotely. No partially completed fit was scored.
The chosen checkpoint was copied and its hash verified before reading 64 new
held-out episodes, selected using metadata seed 390918 and excluding the 192
previously inspected episode IDs. Weights did not change after test inspection.

| Same new test episodes | Previous integrated model | New spatial acceleration model |
| --- | ---: | ---: |
| Full vx errors / 184,160 | 57 | **16** |
| Full vx exact accuracy | 99.9690% | **99.9913%** |
| Actual horizontal accelerations caught / 48 | 32 | **47** |
| Acceleration recall | 66.6667% | **97.9167%** |
| False accelerations in 11,197 eligible sources | 29 | **3** |
| Acceleration precision | 52.4590% | **94.0000%** |
| Ordinary-flight vx errors / 169,294 | 32 | **8** |
| Wall-event vx errors / 6,813 | 2 | **0** |
| Brick-event vx errors / 4,966 | 18 | **2** |
| Paddle-event vx errors / 3,269 | 6 | **6** |

The new head has four magnitude-decision errors: one missed acceleration and
three false accelerations. One frozen-parent direction error brings its region's
full-vx errors to five. Eleven errors remain outside the region. The six paddle
errors and every outside-region prediction are unchanged. Event categories can
overlap. Validation has three full-field errors / 88,590 (99.9966%); test and
validation percentages must not be interchanged.

The source-only inputs remain sufficient for every audited outcome; no new
missing field was discovered. This remains one-step vx prediction with supplied
fractional position and collision memory, not closed recursive full-state
simulation. Vertical velocity, positions, brick layout, and memory updates still
need accurate learned transitions.

Selected checkpoint:
`runs/ball-acceleration-20260918/gpu-artifacts/spatial-balanced-s91/best.pt`, SHA256
`d7a4099725c1d21c1e9063b8758c78752179a07905d0a76dcff21f69f1029080`.
The run's `result.json`, `provenance.json`, `fresh-test-errors.json`, and archived
comparison/configuration files preserve the measurements and evaluation identity.
All parent tensors match the prior checkpoint exactly. CPU and CUDA fresh-test
prediction hashes match exactly, as do validation metrics. The final suite passed
370 tests with two skips; Ruff, frozen dependency sync, and whitespace checks
passed.

## 2026-09-18: Discrete vertical velocity with a frozen horizontal model

Next-vy learning reuses the existing compact ball/controller state and spatial
brick representation. Native forward replay checks all 5,271,950 nonterminal
training sources and 88,590 validation sources. Paddle measurement is recovered
exactly from current charge on those sources. This audit finds no missing input
for these recorded transitions; fractional y and contact memory still come from
reconstruction, so it does not establish an independently closed learned state.

A new `ball_vertical_velocity` registry model keeps the complete horizontal
predictor frozen and adds source-routed upper-field, paddle, and ordinary-flight
heads. The upper network shares 13→64→64 ReLU cell features, max pools occupied
cells, then applies 135→128→128→8 layers. The paddle network uses
97→256→256→256→8 ReLU layers, including raw scalar/binary charge alongside the
existing learned paddle geometry. Flight uses a learned 1→32→8 classifier.
The eight signed classes are derived from training targets. Collision routing
never uses event or successor labels; native rules run only in the offline audit.

Both collision heads train together with summed cross-entropy and extra sampling
of actual vy changes. Their parameters are disjoint, allowing independent
validation checkpoint selection before composition. The parent contains 386,781
frozen parameters; 199,064 vertical parameters are fitted. Derived arrays remain
in local/remote RAM, with no dataset edits or persistent feature cache.

| Development fit | Updates | Validation errors / 88,590 | Interpretation |
| --- | ---: | ---: | --- |
| Seed 91, balanced sampling | 12,000 | 26 | Early upper-field errors dominate. |
| Seed 2026, balanced sampling | 12,000 | 22 | Similar initial result. |
| Seed 91, longer fit | 60,000 | **1** | Upper head is exact on validation; one missed paddle bounce. |
| Seed 2026, longer fit | 60,000 | 3 | Confirms low error with the same inputs and architecture. |

Select seed 91 with upper weights from update 30,800 and paddle weights from
17,200. Hash the complete checkpoint before reading 64 previously unused test
episodes, selected with metadata seed 490918 and excluding all four prior test
sets. No subsequent fit uses their errors.

| Fresh-test subset | Sources | Exact errors | Accuracy |
| --- | ---: | ---: | ---: |
| All transitions | 182,441 | **13** | **99.9929%** |
| Actual vy changes | 10,441 | 4 | 99.9617% |
| Ordinary transitions, no audited event | 167,702 | 9 | 99.9946% |
| Paddle events | 3,335 | 1 | 99.9700% |
| Brick events | 4,895 | 3 | 99.9387% |
| Ceiling events | 2,211 | 0 | 100% |
| Side-wall events | 4,527 | 1 | 99.9779% |

Event groups overlap. Two of the 527 magnitude-changing targets are predicted
incorrectly. The unchanged-vy baseline makes 10,441 errors on these same sources.
The trained model therefore learns the rare changes as well as ordinary motion.
CPU and CUDA checkpoint reloads produce identical fresh-test predictions, and
horizontal predictions and weights remain exactly identical to the parent.

The result is a successful one-step vertical-velocity model. Position,
fractional-coordinate and contact-memory updates, termination, and full-state
recursive evaluation remain outstanding. The dataset is unchanged. Evidence,
checkpoint provenance, four development fits, and evaluation results are under
`runs/ball-vy-20260918/` and `logs/ball-vy-20260918/`. Verification: 372 tests
passed, two skipped; Ruff, frozen dependency sync, and whitespace checks passed.

## 2026-09-18: Discrete ball displacements and coherent fractional y

The next experiment isolates x displacement, freezes that model, then learns
vertical displacement including its fractional coordinate. Native forward replay
matches recorded positions on all 5,271,950 nonterminal training sources and
88,590 validation sources. The existing 118 current-state values are sufficient
on these audited transitions. No dataset columns or persistent feature caches
are added. Y-fraction labels come from audited native replay, with integer y
checked against recorded successors.

Velocity prediction alone does not establish position: adding twice the true
successor velocity still gives 1,669 x errors and 3,330 combined-y errors on
validation. The game can collide during either native frame. These oracle
baselines only support diagnosis; the models never receive true next velocities.

The training vocabulary has 22 x and 19 y displacements. A `ball_position`
registry model keeps the successful vy/vx parent frozen and fits separate
upper-field, paddle, and remaining-region heads. The upper head shares
13→64→64 ReLU cell features and applies 135→128→128→K layers after masked max
pooling; the paddle head is 97→256→256→256→K. K is the axis vocabulary size.
Remaining-field x uses 31→128→128→22 layers; y uses 1→32→19. All 2,316 distinct
training (x, vx) pairs outside the collision regions have unambiguous x targets;
all remaining-region y targets equal twice incoming vy.

| Development fit | Updates | Validation exact errors / 88,590 |
| --- | ---: | ---: |
| X, seed 91 | 30,000 | 4 |
| X, seed 2026 | 30,000 | **3**, selected and frozen before y fitting |
| Y including fraction, seed 91 | 60,000 | **0** |
| Y including fraction, seed 2026 | 60,000 | **0**, selected by predefined name tie-break |

Both models reuse the same training arrays in RAM. Disjoint region heads are
selected independently using validation exact errors. The x model fits 227,586
parameters and the y model fits 203,673, each preserving a complete frozen
585,845-parameter parent. Y predicts a single continuous eighth-pixel coordinate,
then separates integer RAM y and remainder, preserving carry/borrow consistency.

Before reading any new test transitions, freeze and hash both complete models.
The shared fresh test reserves 64 previously unused episodes using metadata seed
590918 and excludes all five earlier test sets. No training follows inspection.

| Fresh-test subset | Sources | X errors | Y-including-fraction errors |
| --- | ---: | ---: | ---: |
| All transitions | 187,251 | **21 (99.9888% exact)** | **7 (99.9963% exact)** |
| Ordinary transitions, no audited event | 171,983 | 6 | 1 |
| Paddle events | 3,251 | 12 | 2 |
| Brick events | 5,122 | 3 | 3 |
| Ceiling events | 2,624 | 0 | 0 |
| Side-wall events | 4,516 | 3 | 1 |

Event groups overlap. X MAE is 0.000283 px; combined-y MAE is 0.000156 px.
Integer y has seven errors; its fractional component has three (99.9984%).
On displacements differing from twice incoming velocity, x makes 15 errors /
6,652 and y makes five / 10,997. The remaining six x errors and two y errors
occur where constant-velocity displacement would have been correct.

Both positions are simultaneously exact on 99.9850% of transitions (28 errors).
Frozen vx and vy make 17 and 18 errors on this same new test; all four ball
quantities are simultaneously exact on 99.9701% (56 errors). Those velocity
scores use a different test set from the original velocity reports. Both stored
velocity components remain identical to their parent, and CPU/CUDA checkpoint
reloads agree exactly.

This establishes accurate one-step positions, including a learned next-fraction
update. It does not establish a closed recursive emulator: current layout,
contact memory, prior paddle-hit count, width, and paddle state are supplied.
X errors remain concentrated around paddle interactions. Perfect y validation
still leaves seven fresh-test errors. Further improvements need development
experiments without fitting this held-out test.

Artifacts, provenance, the four development fits, and frozen test results are
under `runs/ball-position-20260918/` and `logs/ball-position-20260918/`.
Verification: 375 tests passed, two skipped; Ruff, frozen dependency sync, and
whitespace checks passed. No shared runner/player or baseline approach changed.

## 2026-09-18: Coherent brick-layout prediction

The next isolated target is the full successor brick layout. Native replay
matches the recorded transitions using the existing state. Across all 5,271,950
nonterminal training sources, 142,307 change the layout; each removes exactly one
brick. There are no added cells, multiple removals, empty source layouts, or
complete clears. The smallest source layout has three bricks. Validation has
2,432 removals in 88,590 sources. All 108 cells are represented in training, with
877–1,800 removals per cell.

The native game has a separate wall-refill phase, but the audited data never
reaches it. The learned output contract therefore covers unchanged layouts or
one occupied-cell removal and cannot represent refills. No new state columns
or persistent feature caches are written. The targets are recorded brick masks;
native rules support the audit only.

A new `brick_layout` model scores 109 outcomes: no change and each possible cell
removal. Absent cells are masked from removal choices. The model copies the
successful y model's 13→64→64 spatial encoder into a trainable cell encoder,
then combines each local 64-feature vector with a pooled 64-feature context and
71 global current-state features. A shared 199→64→64→1 ReLU network scores
removals, while 135→128→128→1 layers score no change. The model uses ball x/y,
vx/vy, fractional y, contact memory, and current bricks. Paddle/controller
inputs do not affect this head. It learns collision decisions without native
rules, true successor values, or event-based routing in inference.

There are 56,130 fitted parameters and 789,518 frozen y/velocity-parent
parameters. Compare two seeds for 30,000 AdamW updates with batch 512 and 25%
removal samples, selecting minimum validation whole-layout errors every 500
updates. Separately test an analytical +2.48619236 correction to the no-change
bias to undo the sampled class-prior shift. The correction is derived from
training frequencies only; no threshold sweep or additional fit is performed.

| Development candidate | Best training update | Validation errors / 88,590 | Breakdown |
| --- | ---: | ---: | --- |
| Seed 91 | 18,000 | 1 | One false removal; all true removals correct. |
| Seed 2026 | 20,000 | **0** | All layouts and removals exact; selected. |
| Seed 91 with prior correction | 18,000 | 1 | One missed removal; no false removals. |
| Seed 2026 with prior correction | 20,000 | **0** | Same validation result; uncorrected model wins name tie-break. |

Freeze and hash the complete seed-2026 checkpoint before reading 16 previously
unused held-out episodes (metadata seed 690918). Exclude all six earlier test
sets and leave 64 episodes untouched for later full-state evaluation.

| Fresh-test measure | Result |
| --- | --- |
| Complete next layout | **45,245 / 45,245 exact (100%)** |
| Actual removals, correct cell | **1,184 / 1,184 (100%)** |
| Unchanged layouts | **44,061 / 44,061 exact** |
| Missed removals / false removals / wrong-cell removals | **0 / 0 / 0** |
| Removal-cell precision / recall / F1 | **1.0 / 1.0 / 1.0** |

An always-unchanged baseline reaches 97.3831% aggregate exactness while missing
all 1,184 removals. The learned result is therefore not an unchanged-layout
shortcut. The fresh test has no clears or refills and a minimum of six bricks
in a source layout. These finite results do not establish universal accuracy.

CPU and CUDA checkpoint reloads produce identical layouts and preserve the
parent's y and velocity predictions exactly. The separate x checkpoint is
unchanged. No test errors are used for fitting. Contact memory is still supplied
as a current input; its learned update is the next dependency before recursive
full-state simulation. Wall-refill behavior remains outside this model's output
contract and observed training coverage.

Artifacts, audit and split provenance, development comparisons, and frozen-test
results are under `runs/brick-layout-20260918/` and `logs/brick-layout-20260918/`.
Verification: 377 tests passed, two skipped; focused decoder/freeze tests,
Ruff, frozen dependency sync, and whitespace checks passed.

## 2026-09-19: Learned brick-contact memory and isolated feedback

Learn the next contact flag while preserving the selected brick-layout, y,
velocity, and separate x models. Native replay reconstructs contact labels in
RAM from known resets and matches all 5,271,950 nonterminal training successors
against recorded ball coordinates, velocities, and bricks. Contact labels are
derived supervision, not independently recorded observations. No dataset columns
or persistent feature caches are written. All later lives remain eligible as
separate segments.

A first head receives 71 current ball features and two probabilities from the
frozen brick model, no removal versus any removal. The 73→128→128→2 ReLU
classifier plateaus at 102 validation errors, mostly false activations near the
brick area's vertical edges. Grouping exactly identical float32 inputs finds
no contradictory validation targets, so these results do not prove missing
information or an irreducible accuracy limit.

Expose predicted collision geometry next. Weight each cell's 13 geometric
features by its frozen removal probability and append their sum. The resulting
86→128→128→2 ReLU classifier uses 27,906 trainable parameters and keeps 845,648
parent parameters frozen. Inputs are current ball x, RAM y, vx, vy, fractional y,
contact memory, and current bricks. Neither recorded successor events nor native
collision rules enter inference. The additional features describe the model's
predicted collision, so they add no dataset fields.

Use cross-entropy and AdamW for 20,000 updates per seed, batch 512. Each batch
contains 128 examples from each current/next contact pair. Learning rate starts
at 0.001 and decays by cosine to 0.00002, with weight decay 0.0001 and gradient
clipping at 5. Select the earliest minimum-error validation checkpoint every 250
updates, then compare runs by errors and name. Training has 121,374 activations
and 121,223 clearings; validation has 2,053 and 2,050 respectively.

| Candidate | Best update | Validation errors / 88,590 |
| --- | ---: | ---: |
| Removal probability only, seed 91 | 11,250 | 104 |
| Removal probability only, seed 2026 | 18,500 | 102 |
| Predicted collision geometry, seed 91 | 14,500 | **0** |
| Predicted collision geometry, seed 2026 | 18,000 | **0**, selected by name |

Both GPU addresses were unreachable. Run locally on CPU, reusing frozen features
in RAM. A narrow Arrow read recovers source layouts for the geometry comparison;
all ordered eligible ball states and 71 encoded source values match the earlier
audit exactly. Probability differences due to batch size are at most 1.2e-7.

Freeze and hash the selected checkpoint before evaluating the same 16 held-out
episodes used for brick-layout testing. This is explicitly a reused test and
preserves 64 untouched episodes for eventual full-state evaluation. No test rows
participate in contact training or selection.

| Reused-test contact transition | Examples | Errors |
| --- | ---: | ---: |
| Inactive remains inactive | 41,899 | 0 |
| Activates | 996 | 0 |
| Clears | 993 | 0 |
| Active remains active | 1,357 | 0 |
| All contact predictions | **45,245** | **0** |

Feed only predicted contact into subsequent transitions, resetting at each life,
episode, or step gap. Other state fields remain recorded or reconstructed. Across
52 life segments, contact stays exact and all predicted layouts stay exact. The
frozen ball models retain 8 x errors, 3 combined-y errors, 5 vx errors, and 5 vy
errors. Their union covers 15 transitions, unchanged from supplied-contact
inference. Joint accuracy across contact, layout, and four ball quantities is
99.9668%. Of 1,181 complete seven-step windows around brick removals, 1,178 have
every output exact; the other three contain existing ball errors. Validation
feedback likewise preserves zero contact/layout errors and five joint ball
errors across 88,590 transitions.

This establishes contact-only feedback on the evaluated lives, not full-state
recursive simulation. Hit-count and width updates, termination, rare ball errors,
and wall-refill coverage remain unresolved. Contact accuracy measures agreement
with replay-derived labels and does not prove universal correctness.

The complete selected checkpoint is
`runs/brick-contact-20260919/geometry-s2026/best.pt`, SHA-256
`4f23f5fc4d2dea3159a15c813bd98e9303883079447a661a3cac692c5ee297b0`.
Audit receipts, comparisons, and evaluation results are under the corresponding
`logs/brick-contact-20260919/` and run folders. CPU reload reproduces the selected
raw-input predictions, every nested parent weight is unchanged, and the separate
x checkpoint retains its previous hash. Verification passes 379 tests with two
skipped, Ruff, frozen dependency sync, and whitespace checks. No GPU parity result
is claimed for this experiment.

## 2026-09-19: Learned capped paddle-hit count

Learn a binary paddle-hit event and decode the next count as the current count
plus that event, capped at 12. Native replay supplies supervision and verifies
all 5,271,950 nonterminal training successors against recorded ball and brick
state. Keep all eligible lives, separating their boundaries. No dataset columns
or persistent feature caches are written. The training set contains 440,833
sources in the established paddle region and 93,389 actual hits.

The new `paddle_hit_count` model freezes the selected vertical-velocity parent
and reuses its paddle encoding. Remove the prior-count scalar, leaving 96
features from current ball x, RAM y, vx, vy, fractional y, paddle x, width and
charge, including the frozen learned intermediate paddle position. The event
classifier excludes count, contact memory, and bricks. Its decoder alone uses
current count. This makes event predictions independent of accumulated count
errors. A broad source-only region, RAM y 160 through 183 with positive vy,
covers every audited hit; sources outside it retain the count.

The head is 96→256→256→256→2 with ReLU and cross-entropy. It has 156,930 fitted
parameters plus 585,845 frozen parent parameters. Initialize the hidden layers
from the vertical paddle head, dropping its count column; initialize the two
outputs by averaging positive-vy and negative-vy output weights respectively.
The decoder enforces a zero-or-one increment and saturation, but collision
decisions remain learned. Native game rules never run during inference.

Train two seeds for 20,000 AdamW updates, batch 512 with equal hits and non-hits
from the paddle region. Use learning rate 0.0005, cosine decay to 0.00001,
weight decay 0.0001, and gradient clipping at 5. Select by validation hit errors,
then count errors, retaining the earliest update and breaking run ties by name.
Validation contains 1,519 hits, but only 699 increment the count. The other 820
occur at saturation, so count accuracy alone would conceal event mistakes.

| Candidate | Best update | Validation count errors / 88,590 | Hit-event error |
| --- | ---: | ---: | --- |
| Seed 91 | 2,250 | 0 | One missed hit at count 12 |
| Seed 2026 | 3,750 | 0 | One false hit at count 12; selected by name |

Freeze the selected checkpoint before reading the count-test transitions. Reuse
the same 16 episodes previously inspected for brick layout, contact, and ball
errors, keeping them out of count fitting and selection. This is not a fresh
test; 64 episodes remain untouched for eventual full-state evaluation.

| Reused-test measure | Result |
| --- | --- |
| Exact next count | 45,244 / 45,245, **99.9978%** |
| Actual count increments detected | **377 / 377** |
| Actual paddle hits detected | **801 / 801** |
| Hits detected after saturation | **424 / 424** |
| False hits and false count increments | **1** each |
| Hit precision / recall | **99.8753% / 100%** |
| Count accuracy with count/contact feedback | 45,241 / 45,245, **99.9912%** |

The false hit occurs at episode 1570, step 2228 and changes count 4 to 5. The
frozen intermediate-paddle predictor gives 61 px, matching the offline
controller calculation. This is a remaining event-classification error, also
seen in the existing ball velocity models for that source. It is not explained
by an incorrect intermediate paddle prediction. No test-directed refitting is
performed.

Count/contact feedback starts from known life resets and supplies every other
state field. The erroneous count persists for four transitions until the
reference segment boundary. Across 52 life segments, one has count errors;
contact and layout stay exact. Every ball prediction is unchanged. Joint state
errors rise from 15 to 18 because of three additional counter-only errors,
giving 99.9602% joint accuracy. Validation feedback has zero count/contact errors
across 79 segments, retaining five existing joint ball errors. These checks do
not demonstrate autonomous error recovery, learned termination, or full-state
recursive simulation.

The selected CPU checkpoint is
`runs/paddle-hit-count-20260919/count-s2026/best.pt`, SHA-256
`19d70377d9ba6db8e02d3720a27a6f4f3abe160339f4ed4959d3d1d815e533cf`.
Audit, comparison, selection, and feedback records are under that run folder
and `logs/paddle-hit-count-20260919/`. CPU reload preserves raw-input predictions
and every frozen parent weight. The separate x and contact checkpoint hashes
are unchanged. Verification passes 381 tests with two skipped, Ruff, frozen
dependency sync, and whitespace checks. Paddle width, termination, rare collision
errors, and wall-refill coverage remain unfinished.

## 2026-09-19: Learned paddle width and three-field feedback

Audit the width transition before fitting. The pinned native code narrows the
paddle on a ceiling collision and restores full width at serve or life loss.
Those lifecycle resets are outside the within-life dynamics scope. Every one of
5,271,950 nonterminal training sources matches the native width rule: 2,159
narrowings, 4,093,205 wide retentions, and 1,176,586 narrow retentions. There are
no within-life widenings. Training targets are recorded successor widths;
native replay checks the transition rule and reconstructs fractional y in RAM.
The dataset and its columns remain unchanged.

Use current width, RAM ball y, fractional y, and vertical velocity. Integer and
fractional y form one eighth-pixel coordinate. The initial encoding combines
three normalized scalars, 11 coordinate bits, six signed-velocity bits, and a
current-narrow indicator. A 21→64→64→2 ReLU classifier predicts width 12 or 16.
It does not impose a monotonic update or run a native collision rule.

Width changes are rare. The validation persistence baseline scores 99.9594%
while missing every one of 36 changes. Fit with equal representation of wide
retention, narrowing, and narrow retention, 128 examples per category per batch.
For each seed, run 10,000 AdamW updates with cross-entropy, learning rate 0.001,
cosine decay to 0.00002, weight decay 0.0001, and gradient clipping at 5. Select
minimum validation width errors, then changed-width errors, retaining the
earliest checkpoint at 250-update checks and breaking run ties by name.

The initial heads catch all true changes but narrow early on two and one
validation sources. Append the normalized constant-velocity proposal `y + vy`
and its 11 coordinate bits. The new 33→64→64→2 head has 6,466 parameters and
uses the same four input fields. It exposes simple motion arithmetic without
executing a collision decision or taking a true successor as input. Cached and
direct feature construction agree exactly on 359,086 original training sources.

| Candidate | Best update | Validation errors / 88,590 | Missed changes |
| --- | ---: | ---: | ---: |
| Initial encoding, seed 91 | 6,750 | 2 false narrowings | 0 |
| Initial encoding, seed 2026 | 7,250 | 1 false narrowing | 0 |
| Motion proposal, seed 91 | 4,750 | 2 false narrowings | 0 |
| Motion proposal, seed 2026 | 7,500 | **0**, selected | **0** |

Freeze and hash the selected checkpoint before reading the same 16 reused test
episodes. They remain outside width training and selection, but their earlier
brick, contact, count, and ball results have already been inspected. Preserve
64 untouched episodes for eventual full-state testing. Do not present the reused
result as fresh-test generalization.

| Reused-test width transition | Examples | Errors |
| --- | ---: | ---: |
| Wide remains wide | 36,007 | 0 |
| Narrows | 18 | 0 |
| Narrow remains narrow | 9,220 | 0 |
| All next widths | **45,245** | **0** |

Change precision and recall are both 100%. There are no false changes or false
widenings. The small number of actual changes is an important limit on the
coverage of this finite evaluation.

Feed width, count, and contact forward together from known life starts, using
recorded or reconstructed values for the other state fields. Width remains exact
across all 52 test life segments. Every older model output matches the preceding
count/contact feedback evaluation byte for byte. The existing count error still
covers four transitions; joint state errors remain 18, or 99.9602% joint accuracy.
Validation feedback has zero width errors across 79 segments and retains five
existing joint ball-state errors. These are partial-state checks with reference
boundaries, not complete recursive simulation or learned termination.

The selected CPU checkpoint is
`runs/paddle-width-20260919/proposal-s2026/best.pt`, SHA-256
`dd1cf25b8e049c5f858f2b183d64f4511bb1cfe2589a3a835e39440c8a09208d`.
Audit, comparison, selection, and feedback records are under that run folder
and `logs/paddle-width-20260919/`. CPU reload reproduces the selected predictions,
and all frozen x, contact, and count checkpoint hashes are unchanged. No dataset
columns or persistent feature caches are written. Learned life-loss termination,
rare collision errors, wall-refill coverage, and full-state integration remain
unfinished.

Verification for the width model passes 385 tests with two skipped, Ruff,
frozen dependency sync, and whitespace checks.


## 2026-09-19: Learned life-loss termination

The older history-based terminal probe had validation F1 0.299. Tackle this
state target independently with the sufficient current-state representation,
including the death transitions excluded by the position and velocity probes.

Restore terminal sources in a RAM-only training view, leaving the dataset
unchanged. Their fractional y comes from the previous source's causal replay
output within the same life. Preserve later lives and distinguish death from
censoring. The original game's pre-movement loss check at the fixed two-frame
cadence agrees with every included recorded boundary across training, validation,
and reused test. This is an offline audit, not the neural inference rule.

Fit a 31→64→64→2 ReLU classifier with 6,338 parameters, using current RAM y,
fractional y, and vy. Inputs include scalar and bit encodings of the combined y,
vy, and the current-motion proposal `y + vy`. No lives or actions are required
by this tested predictor. Native rules do not execute at inference.

Use all 1,824 training episodes: 5,275,072 sources, including 3,122 deaths.
Balance deaths, surviving sources near the loss boundary, and other survivors
with 128 examples of each per batch. Run two seeds for 8,000 AdamW updates each,
with cross-entropy, learning rate 0.001 decayed to 0.00002, weight decay 0.0001,
and gradient clipping at 5. Select by validation errors, then missed deaths,
earliest checked update, and finally run name. Both seeds reach zero validation
errors; seed 2026 is selected by the declared run-name tie break. Each training
run takes about five CPU seconds after preparation.

| Check | Sources | Actual deaths | Missed deaths | False stops |
| --- | ---: | ---: | ---: | ---: |
| Validation, seed 91, update 250 | 88,638 | 48 | 0 | 0 |
| Validation, seed 2026, update 500, selected | 88,638 | 48 | 0 | 0 |
| Reused 16-episode test, frozen selected checkpoint | 45,283 | 38 | 0 | 0 |

Death precision and recall are 100% on both validation and reused test. Scan for
the first predicted stop in each recorded life segment: all 48 validation and
38 test deaths stop on the exact transition. There are no predicted stops in
31 censored validation segments or 14 censored test segments. These are source
state checks, not recursive ball prediction. Only 38 test deaths were observed,
and the 16 test episodes were previously inspected for other targets. The
remaining 64 held-out episodes are still untouched.

The model is saved at `runs/life-termination-20260919/stop-s2026/best.pt`, SHA-256
`ff2bad6f68b5701699684537186ba225b171ae271f65ffbf1587a1a68493f37d`.
The result is in `runs/life-termination-20260919/result.json`; preparation,
training, audit and selection scripts/receipts are under
`logs/life-termination-20260919/`. CPU checkpoint reload is exact. Checked prior
model hashes are unchanged; no dataset fields or persistent feature caches are
written. Full-state integration and the effect of accumulated ball errors on
termination remain untested.

Verification passes 387 tests with two skipped, Ruff, frozen dependency sync,
and whitespace checks. Focused tests cover the input dependency contract,
learning, checkpoint reload, and network-controlled stop decisions.


## 2026-09-19: Paired vertical position and velocity

Combine the existing y predictor, including fractional y, with its velocity
parent and test their feedback independently before adding other state heads.
Each transition computes both predictions from one source state, then applies
the selected feedback together. Use all eligible starts at horizons 1, 8, 32,
and 128; never cross reference life boundaries. Keep other state supplied.

The first incompatibility is a validation paddle bounce at episode 1918,
step 2461. Predicted next y is correctly 176.375, but vy remains +3.375 instead
of switching to -3.375. This one velocity error becomes 128 failed 128-step
windows with joint feedback. Y-only feedback adds no errors, while vy-only
feedback produces 18 incorrect endpoints at that horizon. This distinguishes
a velocity disagreement from failure of the original y model on validation.

Freeze both original predictors and train a 10→64→64→8 ReLU correction to vy.
Its inputs are current vy, predicted y displacement and eight original vy
probabilities. The 5,384-parameter head can use evidence of a bounce in the
position prediction without executing collision rules or receiving true next y.
Use 359,086 sources from the original 128 training episodes, with equal batch
sampling of original velocity errors, correct velocity changes and other correct
predictions. Only three training sources have original vy errors, so they are
sampled repeatedly. Both seeds run 6,000 AdamW updates. Their earliest zero-error
validation checkpoint is update 250, and both become exact through all 78,806
eligible 128-step validation windows. Select seed 2026 by the declared name tie
break. This smaller correction makes joint rollout fine-tuning unnecessary for
the validation target at this stage.

Freeze the selection, then evaluate the same 16 reused test episodes, keeping
64 untouched episodes reserved. The correction reduces one-step vy errors from
five to one; y retains its three errors. Joint errors fall from seven to three.

| Joint-feedback check | Original pair | Corrected pair |
| --- | ---: | ---: |
| One-step exact, 45,245 sources | 99.9845% | 99.9934% |
| Incorrect 8-step endpoints / 44,881 | 45 | 24 |
| Incorrect 32-step endpoints / 43,640 | 165 | 96 |
| Incorrect 128-step endpoints / 39,203 | 422 | 224 |
| Fully exact 128-step windows / 39,203 | 38,692 | 38,979 |
| Fully exact 128-step percentage | 98.6965% | 99.4286% |
| Mean 128-step endpoint y error | 1.9200 px | 1.5996 px |
| Maximum 128-step endpoint y error | 782.25 px | 897.75 px |

Windows overlap, so 224 failed windows are seeded by three distinct transitions,
not 224 independent collision mistakes. All three remaining one-step errors
involve y: two paddle interactions and one brick interaction, with one also
having incorrect vy. Some failures drift more severely after correction, and
mean y error is higher at horizons 8 and 32. This is a reduction in failure
frequency, not a guarantee of bounded error. No held-out examples were used for
fitting. All other fields remain supplied and reference boundaries stop the
runs; complete ball or full-state feedback remains untested.

The selected checkpoint is
`runs/vertical-pair-20260919/coupling-s2026/best.pt`, SHA-256
`db047b3a50ba075aba9238240b86109b9c1a209fe19b71234daae3ccdc60e748`.
It saves both original models and the correction. All original model weights
remain identical, and no dataset columns or persistent feature caches were
written. Scripts and audit receipts are under `logs/vertical-pair-20260919/`;
all four feedback comparisons are in `runs/vertical-pair-20260919/result.json`.
The evaluator's reuse of exact reference-input predictions matches brute-force
feedback in tests. Checkpoint reload also reproduces the full reused-test
rollout results after eliminating duplicate position computation.

Verification passes 390 tests with two skipped, Ruff, frozen dependency sync,
and whitespace checks. The three focused pair/evaluator tests also pass after
the inference optimization.


## 2026-09-19: Collision displacement refinement rejected after held-out evaluation

Attempt a focused improvement to the remaining vertical pair failures. The
original validation set has zero errors, so reserve 256 development episodes
from the training pool before fitting new heads. These episodes were seen by
the frozen parent models. They provide a refinement comparison, not independent
generalization evidence. Fit on the remaining 1,568 episodes and preserve the
original 32 validation episodes as a zero-regression gate.

A full causal audit covers 5,271,950 nonterminal sources. The fitting subset has
23 y mistakes across 4,529,910 sources, all in the upper or paddle regions.
Development has five y errors and six joint errors across 742,040 sources.
This supplies training analogues of collision errors without fitting any of the
known held-out witnesses.

Add two frozen-feature residual y heads, 162→64→64→19 and 124→64→64→19 ReLU,
29,222 parameters. The original y/vy pair remains frozen. The correction sees
existing spatial/paddle features and original predicted probabilities. Corrected
y displacement then supplies the existing velocity coupling. Keep ordinary
flight unchanged and initialize residual logits to zero.

Train two seeds for 6,000 updates each with categorical displacement loss,
small residual-logit regularization, and balanced sampling of original errors,
correct nonlinear movements, and correct linear movements. The full-strength
heads introduce validation errors. Neither seed yields an eligible checkpoint.

As a bounded follow-up, calibrate the last seed's correction strength using
only development and original validation. Scale 0.3 preserves exact original
validation and reduces development joint errors from six to three. It repairs
22 of 23 fitting y errors without introducing fitting errors. Full 128-step
development failures drop from 357 to 89, mean y error from 0.2143 to 0.0240
pixels, and maximum y error from 911.25 to 459 pixels. Freeze the candidate.

| Stage | Original validation joint errors | Development joint errors |
| --- | ---: | ---: |
| Current pair | 0 | 6 |
| Unscaled seed 91, final update | 4 | 13 |
| Unscaled seed 2026, final update | 4 | 12 |
| Seed 2026 at 0.3 correction strength | 0 | 3 |

The reused held-out test does not confirm the improvement. The candidate keeps
all three prior one-step errors and introduces one new paddle displacement error.
The ball bounces, but the predicted displacement reflects the wrong timing within
the two-native-frame transition. Next vy remains correct for that new mistake.

| Reused-test metric | Current pair | Candidate |
| --- | ---: | ---: |
| Joint errors / 45,245 transitions | 3 | 4 |
| Failed 128-step windows / 39,203 | 224 | 352 |
| Fully exact 128-step windows | 99.4286% | 99.1021% |
| Mean endpoint y error at 128 steps | 1.5996 px | 2.3451 px |
| Worst endpoint y error at 128 steps | 897.75 px | 897.75 px |

Reject the candidate and retain the existing combined pair. This experiment
improves fitting and development behavior but fails to transfer that improvement
to the reused held-out episodes. It does not establish missing state information
or justify more fitting to the test cases. No further fitting follows the test,
and 64 untouched episodes remain reserved. Other state fields and stopping
boundaries remain supplied throughout these partial-state rollouts.

The retained checkpoint is
`runs/vertical-pair-20260919/coupling-s2026/best.pt`, SHA-256
`db047b3a50ba075aba9238240b86109b9c1a209fe19b71234daae3ccdc60e748`.
The rejected candidate is
`runs/vertical-refinement-20260919/calibrated-s2026/best.pt`, SHA-256
`e1ff4dc450d3df2992673d4246289c163f531ed2b34c658fea149c9266b9f882`.
Its checkpoint records development acceptance; final rejection is recorded in
`logs/vertical-refinement-20260919/decision.json` and the run result's
`disposition`. Scripts and receipts are under that log directory. Frozen parent
weights and dataset contents are unchanged. No persistent feature caches were
written, and no shared runner/player behavior changed. Verification passes
391 tests with two skipped, Ruff, frozen dependency sync, and whitespace checks.

## 2026-09-19: Explicit collision timing rejected after held-out evaluation

Test a direct representation of bounce timing: jointly predict the vertical
velocities used in the first and second native movements. Their sum determines
y displacement, and the second determines next vy. Offline native replay confirms
these targets on 5,271,950 sources without mismatches. They add supervision, not
new source information; the first velocity is also recoverable from the recorded
displacement and next vy. Dataset contents remain unchanged.

Keep the existing pair frozen and train upper-field 135→128→128→64 and paddle
97→256→256→256→64 ReLU heads on its features. Use 64 joint velocity outcomes,
categorical loss, and an auxiliary timing loss. Balance unchanged, first-frame,
and second-frame events. Fit two seeds for 10,000 updates. The standalone heads
do not pass the original-validation/development gate.

Before reading the reused test, calibrate a prior from the frozen original y
probabilities. Select seed 2026 with prior weight 2 on both heads. It has zero
joint errors on all 88,590 original-validation and 742,040 development sources,
and all their complete 128-step windows are exact. However, the 256 development
episodes were part of frozen-parent training, so this is not evidence of
generalization to episodes unseen by the complete model.

| Reused-test metric | Existing pair | Timing candidate |
| --- | ---: | ---: |
| Joint errors / 45,245 transitions | 3 | 4 |
| Failed 128-step windows / 39,203 | 224 | 352 |
| Fully exact 128-step windows | 99.4286% | 99.1021% |
| Mean endpoint y error at 128 steps | 1.5996 px | 2.7291 px |
| Worst endpoint y error at 128 steps | 897.75 px | 897.75 px |

Reject the candidate and retain `vertical-pair-20260919/coupling-s2026`.
Explicit timing does not improve the reused held-out result: all three original
joint errors remain and one paddle timing error is introduced. This is the same
additional source that failed in the preceding displacement refinement. No
fitting follows the test result, and 64 untouched episodes remain reserved.
Other state fields and life boundaries are supplied; overlapping windows are
not independent failures or full-state gameplay trials.

Before another model change, establish development trajectories unused in the
training of every component. The present result does not prove missing inputs
or isolate the cause of the generalization gap. See
[the training record](training.md#explicit-vertical-collision-timing) for the
split, hyperparameters, calibration, and checkpoint identity. Plans, audit,
scripts, and rejection receipt are under `logs/vertical-timing-20260919`;
`runs/vertical-timing-20260919/result.json` records final rejection. The dataset,
existing checkpoint, and shared runner/player remain unchanged. Verification
passes 394 tests with two skipped, Ruff, and frozen dependency sync.
