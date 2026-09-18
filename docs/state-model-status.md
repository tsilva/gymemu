# Learned state-model status

Updated 2026-09-18. These are separate diagnostic models, not an integrated
recursive emulator. Validation selects checkpoints; reserved-test results are
identified explicitly. Results from different targets and input contracts are
not interchangeable. Percentages denote exact predictions unless stated otherwise.
Layer lists include encoded input size and output size where the compact contract
is established. MLP means a fully connected neural network.

| Target | Model and training objective | Inputs used | Recorded result and status |
| --- | --- | --- | --- |
| Next paddle x | 25→128→128→23, SiLU; classify integer displacement with cross-entropy | Current paddle x, charge, current requested action; scalar and binary encoding | Minimal-input model: **99.9955% validation**, 4 one-pixel errors / 88,590. No reserved-test score for this reduced model. |
| Next paddle charge | 25→128→128→11, SiLU; classify charge change with cross-entropy | Current paddle x, charge, current requested action | **100% validation**, 0 / 88,590 errors in both seeds. Also zero charge errors in 30,909 recursive paddle predictions. |
| Intermediate paddle x, auxiliary to direction | 25→128→128→13, SiLU; classify displacement after one native frame | Current paddle x and charge; fixed placeholder action | **100% validation**, 0 / 7,164; **99.9870% speed-stage fresh test**, 2 / 15,346 errors. Frozen during direction and speed training. Labels derive from the audited controller rule in RAM. |
| Next horizontal ball direction, near paddle | 84→256→256→256→2, ReLU; cross-entropy, plus the frozen intermediate paddle network | Current ball x, integer RAM y, vx, vy, paddle x, paddle width, charge, prior paddle-hit count, fractional y; geometric and binary encoding | **100% validation** on 1,519 paddle hits. **99.9387% speed-stage fresh test** on 3,263 paddle hits, 2 errors; 99.9805% across 15,346 selected descending approaches, 3 errors. Predicts sign only. |
| Next horizontal speed, near paddle | 84→256→256→256→4, ReLU; cross-entropy over four magnitudes, with frozen geometry and intermediate-paddle prediction | Same nine current inputs as the direction model | **99.7548% fresh-test paddle accuracy**, 8 errors / 3,263. Correct on **384 / 388 actual speed changes**. |
| Full next horizontal velocity, near paddle | Frozen direction × learned speed; 329,747 total parameters, 154,372 trained in this stage | Same nine current inputs; no new fields or action/history inputs | **99.9342% validation paddle accuracy**; **99.7242% fresh-test paddle accuracy**, 9 errors / 3,263. Across all 15,346 fresh-test approaches: **99.9348%**, 10 errors. |
| Full-game horizontal velocity, older probe | 195→128→128→8, SiLU; cross-entropy over eight signed velocities; train-only irrelevant-brick augmentation | Current ball/paddle state excluding paddle velocity, charge, prior hit count, fractional y, brick-contact memory, 108 brick cells; relative x encoding | Older full-game one-step probe: **99.6659% validation overall**, but **85.4510% on paddle hits**. Historical baseline; superseded for full-game vx by the routed model below. |
| Upper-field horizontal acceleration | Shared 13→64→64 ReLU network per brick; occupancy-masked max pooling, then 70→128→2; 14,402 trainable parameters | Ball x, RAM y, vx, vy, fractional y, brick-contact memory, 108 brick cells, fixed layout geometry; current y ≤ 100 and incoming speed < 2 | **47 / 48 actual speed increases caught (97.9167% recall)** on the newest fresh test, with **3 false increases** across 11,197 eligible sources; 94% precision. No additional recorded inputs. |
| Full-game horizontal velocity, current | Previous routed vx predictor frozen, plus the spatial acceleration head; 386,781 total parameters | Existing 118-value current-state contract; source-only routing; no event-label inputs | **99.9966% validation**, 3 errors / 88,590. **99.9913% newest fresh test**, 16 / 184,160, compared with 57 for the previous model on the same episodes. Brick events: 2 / 4,966 errors; paddle events: 6 / 3,269, unchanged. One-step vx, not recursive simulation. |
| Next ball x | Two 128-unit SiLU hidden layers; isolated residual regression with squared error | Older context described below, plus four controller values and prior hit count | Older fit: **0.254923 px validation MAE**. Not yet rebuilt with the successful compact collision representation. |
| Next integer RAM ball y | Two 128-unit SiLU hidden layers; isolated residual regression with squared error | Older context plus prior hit count | Older fit: **0.708709 px validation MAE**. Still inaccurate around collisions. |
| Next vertical velocity, vy | Eight-class cross-entropy. Upper field: shared 13→64→64 cell encoder, max pooling, 135→128→128→8. Paddle: 97→256→256→256→8. Flight: 1→32→8. All ReLU; horizontal parent frozen; 199,064 fitted vertical parameters | Existing 118 current-state values; upper head uses ball state, fractional y, contact memory and bricks; paddle head uses the nine paddle-geometry inputs plus explicit charge encoding; flight uses current vy | **99.9989% validation**, 1 / 88,590 errors. **99.9929% fresh test**, 13 / 182,441 errors. Actual velocity changes: **10,437 / 10,441 correct**; paddle: 1 / 3,335 errors; brick: 3 / 4,895; ceiling: 0 / 2,211. One-step prediction; current hidden state still supplied. |
| Next paddle width | Two 128-unit SiLU hidden layers; binary cross-entropy for narrow/full width | One or eight full state observations, seven prior actions and current action | Aggregate validation accuracy exceeds 99.95%, but **0 / 36 width changes correct** in both isolated probes. Not solved. |
| Next brick layout | Two 128-unit SiLU hidden layers; 108 occupancy logits, change-weighted binary cross-entropy | Eight full state observations, seven prior actions and current action | Older isolated probe: **brick-removal validation F1 0.189**. Not solved; mostly unchanged cells make aggregate accuracy misleading. |
| Life-loss terminal flag | Two 128-unit SiLU hidden layers; weighted binary cross-entropy | One full state observation, seven prior actions and current action | Older isolated probe: **validation F1 0.299**. Dataset life boundaries are enforced, but accurate learned termination remains unresolved. |

The older context contains eight full state observations, seven prior requested
actions plus the current action, and 32 retained paddle observations/actions.
Those historical models still include recorded paddle velocity. Their inputs
were used experimentally; this does not establish that every input is necessary
or that this history is sufficient.

The paddle position and charge models have been rolled forward together using
their own predicted x and charge, with recorded actions. Across 30,909 scored
predictions from 256 starts, charge is always exact and position has one transient
one-pixel error. These runs last up to 128 steps and stop at reference life or
segment boundaries. They do not test learned terminal prediction or full-game
rollouts. The older paddle-position model using all four controller fields had
99.9863% reserved-test accuracy; that number must not be attributed to the reduced
charge-only model.

The new direction model's 153,858 trainable parameters plus the frozen paddle
model's 21,517 give 175,375 total. Its nine current inputs need no image history
or current action under this fixed two-native-frame transition contract. This is
not a claim for other frame skips or other state targets. Auxiliary paddle labels
use native-rule supervision, but model inference calls no native controller or
collision rules. The source dataset was not modified for these experiments.

## State still supplied rather than learned

| Value | Current handling |
| --- | --- |
| Fractional ball y | Reconstructed from known reset state and forward native replay for diagnostics. A learned next-fraction update is not established. |
| Prior paddle-hit count | Derived from preceding observed collisions and checked against native replay. It is an input, not a learned output yet. |
| Brick-contact memory | Reconstructed for full-game diagnostics; excluded from the near-paddle direction model. A learned update remains untested. |
| Paddle measurement, repeat and held-input fields | Reconstructed and present in the controller-annotated dataset. The closed x/charge paddle state does not require separate predictions of these fields under the audited reset and two-frame action contract. |
| Paddle velocity | Removed from the new dynamics input/output contract. Historical models and dataset columns retain it. |
| Lives, serve and respawn | Outside the current simulation scope. Each life ends at ball loss; later lives remain separate training segments. |

Horizontal speed and direction are now combined and integrated across the full
field in a saved model, with a learned spatial head correcting most missed
brick-triggered speed increases. Remaining work includes rare horizontal errors,
ball-position and collision-memory updates. Vertical velocity now has its own
accurate learned head with the horizontal predictor preserved unchanged.
High one-step accuracy with reconstructed current inputs does not yet establish
an autonomous compact-state simulator.

Evidence is in [the experiment history](history.md), especially the sections on
[minimal paddle inputs](history.md#2026-09-18-minimum-controller-inputs-for-paddle-position),
[charge updates](history.md#2026-09-18-learned-charge-updates-and-removal-of-paddle-velocity),
[horizontal velocity](history.md#continued-horizontal-velocity-generalization-experiments),
[the direction experiment](history.md#2026-09-18-near-perfect-direction-development-accuracy-and-reserved-test),
[the combined speed/direction experiment](history.md#2026-09-18-learned-horizontal-speed-with-frozen-direction),
[full-game vx integration](history.md#2026-09-18-full-game-vx-integration-with-a-frozen-paddle-model),
[spatial acceleration learning](history.md#2026-09-18-shared-spatial-brick-features-resolve-most-missed-accelerations),
and [vertical velocity](history.md#2026-09-18-discrete-vertical-velocity-with-a-frozen-horizontal-model).


The near-paddle test rows above use the same speed-stage set of 64 fresh held-out
episodes. The current full-game and upper-field acceleration rows use a different, subsequently reserved set
of 64 episodes; its subgroup counts must not be mixed with the speed-stage set. This differs from the earlier direction-only test of 15,069 approaches,
whose results remain in the history. On the new set, frozen direction has three errors,
learned speed has eight, and their combination has ten, including one overlap.
The same validation partition is retained. Neither fresh-test error inspection
nor the intermediate-paddle attribution audit changes the selected weights.

The vertical-velocity row uses another 64 previously unused held-out episodes,
separate from every horizontal test above. Its checkpoint was frozen before
reading those transitions. CPU and CUDA predictions agree exactly, and its
stored horizontal component matches the previous horizontal checkpoint exactly.
