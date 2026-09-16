# Diagnosing rollout collapse

Open the [Training diagnostics view](https://wandb.ai/tsilva/gymemu-Breakout-Atari2600-v0?nw=4b0b3315584).
Chart titles show their exact metric names or registered templates. Multi-metric
charts list each selected name. The Primary metrics accordion is pinned and open. Other metric families have their
own collapsed accordions. New diagnostics require a run started with this code;
the Legacy runs accordion displays the original epoch metrics of older runs.

Start with these signals:

| Chart | What to look for |
|---|---|
| `eval/mse` | A sharp rise means even predictions from recorded histories deteriorated. This is the full comparison metric, with a latest-value summary. |
| `probe/mse/h1, probe/mse/h8, probe/mse/h32` | Compare h1, h8, and h32. Later-step error growing while h1 stays good points to recursive drift. Horizons remain fixed when the training curriculum changes. |
| `probe/ball/mse` | Rising error around a detected target ball can reveal disappearance before whole-frame MSE changes much. Check detector coverage in Ball tracking. |
| `train/grad/norm/max` | A single update spike survives the reporting window. This helps locate the initiating update interval. |
| `train/grad/weights/zero/fraction` | Fraction of updates whose combined matrix/tensor weight gradient norm is exactly zero. Output biases can still learn while this reaches 1. |
| `probe/pixel/wrong_sat/fraction` | Fraction of predicted RGB values at 0 or 1 and more than 0.1 away from the target. Black background alone does not trigger this signal. |
| `train/horizon` | Align regressions with curriculum changes. |
| `train/loss/mean, train/loss/max` | Recent sample-weighted loss and maximum batch loss. Training loss changes meaning with the objective and rollout horizon. |

In Gradients, compare layer norms, nonzero gradient fractions, and missing gradients.
The peak-gradient step records the exact update that produced the window maximum.
A missing gradient differs from a computed gradient full of zeros. Frozen parameters
are omitted. Nonfinite norm rates include reduction overflow; layer nonfinite
fractions distinguish nonfinite elements. Optimizer shows sampled relative updates,
`||after - before|| / max(||before||, 1e-12)`, and parameter norms.

In Activations, pre-sigmoid magnitude and actual sigmoid saturation are measured on
the training forward pass, including its configured precision. Each point is the
worst call within that sampled update, so later recursive steps remain visible.
A high saturation fraction alone is normal for black backgrounds. Combine it with
incorrect saturation, dead weight gradients, and rising held-out error. ReLU zero
fractions can identify a different source of blocked gradients.

Rollouts compares recursive predictions and teacher forcing against the same fixed
targets. It separates starts with incomplete history from regular starts, records
motion error and predicted/target frame changes, and shows target counts at every
horizon. Frames beyond an episode end are omitted, not scored as black images.
An unavailable horizon is absent, not zero error. The rollout/clean ratio uses a
1e-12 denominator floor. Clean-history aggregates are omitted for stateful models;
their rollouts feed predicted state back through the same contract as the player.

Ball tracking uses the existing target-only rectangular-sprite detector with four
pixels of padding. `probe/ball/mse` averages region MSE over detected targets only.
Coverage uses all valid targets. Training charts also separate RGB MSE, the unweighted
ball penalty, and target detector coverage. The training penalty includes zero
contributions from undetected targets, matching the training objective; the held-out
probe ball error averages only detected targets. Occluded or ambiguous frames are excluded, not
scored as successes. Reappearance metrics cover a detected frame immediately after
an undetected frame in the same probe. This is a detector transition, not a label
proving physical occlusion or successful ball localization. Counts reveal whether
the fixed starts actually cover such events. No detections means no error metric.

Prediction examples shows up to four fixed starts. Each row pairs target frames
above predictions, with up to eight evenly spaced columns across the valid rollout.
The exact starts and target IDs live in `probe.json`; `diagnostics.jsonl` and the
PNG files preserve evidence with W&B disabled. Full held-out evaluation and best
checkpoint selection remain unchanged. Small fixed probes can miss rare events;
inspect their coverage and images rather than treating low MSE as proof of playability.

## Naming and cadence

Schema v2 follows GradLab's short semantic paths, explicit axis metrics, unit/statistic
suffixes, registered metric templates, and latest-value diagnostic summaries.
`train/step` counts optimizer updates across stages. `eval/step` identifies the
checkpoint evaluated by full validation. Probe charts use `train/step`. Legacy
aliases remain separate so existing charts and run comparisons keep working.
The managed view is declared in `configs/monitoring/workspace.json`; metric names
are registered in `gymemu/metrics.py`.

By default, global norms are measured every update and reported every 100 batches.
Detailed layer statistics are sampled every 100 batches. The first ten batches of
each epoch are measured and reported individually. A fixed probe runs before each
epoch, every 1,000 batches, and at its end, with up to eight starts and a 32-frame
horizon. Epoch-end diagnostics also flush partial windows. Probe calculations use
float32 and preserve Torch RNG, buffers, and per-module train/eval modes. They do
not compute gradients, call training losses, or alter optimizer updates.

Configure this under `trainer.diagnostics` as described in
[training options](training.md#weights--biases). Monitoring adds compute and sampled
synchronization; disable it explicitly for overhead comparisons. It does not
stop a run, clip gradients, change learning rates, or claim to identify an exact
causal update between sampled layer measurements.
