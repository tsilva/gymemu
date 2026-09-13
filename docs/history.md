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
