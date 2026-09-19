# Learned state-model status

Updated 2026-09-19. These are separate diagnostic models, not an integrated
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
| Next ball x | 22 displacement classes, cross-entropy. Shared 13→64→64 cell encoder; upper head 135→128→128→22; paddle 97→256→256→256→22; remaining field 31→128→128→22. ReLU; 227,586 fitted parameters plus frozen velocity parent | Existing 118 current-state values; spatial upper-field and paddle geometry; remaining-field head uses current x and vx. No successor velocity inputs | **99.9966% validation**, 3 / 88,590 errors. **99.9888% fresh test**, 21 / 187,251; MAE **0.000283 px**. Paddle events: 12 / 3,251 errors; brick events: 3 / 5,122. One-step position. |
| Next RAM ball y together with fractional y | 19 displacement classes for the combined coordinate, cross-entropy. Shared 13→64→64 cell encoder; upper head 135→128→128→19; paddle 97→256→256→256→19; flight 1→32→19. ReLU; 203,673 fitted parameters plus frozen velocity parent | Same current-state contract; y includes the supplied fractional eighth-pixel component. Flight uses current vy; no successor velocity inputs | **100% validation**, 0 / 88,590 in both seeds. **99.9963% fresh-test combined-y accuracy**, 7 / 187,251; MAE **0.000156 px**. Integer y: 7 errors; fractional component: 3 errors (**99.9984%**). Fraction labels derive from audited replay. |
| Next vertical velocity, vy | Eight-class cross-entropy. Upper field: shared 13→64→64 cell encoder, max pooling, 135→128→128→8. Paddle: 97→256→256→256→8. Flight: 1→32→8. All ReLU; horizontal parent frozen; 199,064 fitted vertical parameters | Existing 118 current-state values; upper head uses ball state, fractional y, contact memory and bricks; paddle head uses the nine paddle-geometry inputs plus explicit charge encoding; flight uses current vy | **99.9989% validation**, 1 / 88,590 errors. **99.9929% fresh test**, 13 / 182,441 errors. Actual velocity changes: **10,437 / 10,441 correct**; paddle: 1 / 3,335 errors; brick: 3 / 4,895; ceiling: 0 / 2,211. One-step prediction; current hidden state still supplied. |
| Next paddle width | 33→64→64→2 ReLU classifier, cross-entropy; 6,466 fitted parameters, outputs 12 or 16 pixels | Current width, RAM ball y, fractional y, and vy. Scalar/binary encoding includes the constant-velocity proposal `y + vy`. No other state, action, or history inputs | **100% validation**, 0 / 88,590 and all 36 changes correct. **100% reused-test width accuracy**, 0 / 45,245 and all 18 changes correct. Zero false changes/widenings. Width remains exact across 52 test life segments with width/count/contact fed back together. |
| Next brick layout | 109-class cross-entropy: no change or one occupied cell removed. Shared 13→64→64 cell encoder; removal scorer 199→64→64→1; no-change head 135→128→128→1. ReLU; 56,130 fitted parameters plus frozen y/velocity parent | Current ball x, RAM y, vx, vy, fractional y, brick-contact memory, and 108 brick cells. No image/action history or successor inputs | **100% validation**, 0 / 88,590 complete-layout errors. **100% on a fresh 16-episode test**, 0 / 45,245 errors: all **1,184 removals** correct and **0 false removals** on 44,061 unchanged layouts. Removal precision/recall/F1 all 1.0. Wall clears/refills absent from audited data and unsupported by this output contract. |
| Next brick-contact memory | 86→128→128→2 ReLU classifier, cross-entropy; 27,906 fitted parameters plus frozen brick/y/velocity parent | Current ball x, RAM y, vx, vy, fractional y, contact, and bricks. Frozen parent supplies 71 ball features, 2 predicted removal probabilities, and 13 probability-weighted brick geometry features | **100% validation**, 0 / 88,590. **100% on the reused 16-episode brick test**, 0 / 45,245, including all 996 activations and 993 clearings. Contact-only feedback remains exact across 52 life segments. Labels derive from audited native replay; all other state inputs remain supplied. |
| Next capped paddle-hit count | 96→256→256→256→2 ReLU hit classifier, cross-entropy; decode `min(12, count + hit)`. 156,930 fitted parameters plus frozen vertical parent | Hit detection uses current ball x, RAM y, vx, vy, fractional y, paddle x, width and charge. Frozen intermediate-paddle prediction and geometric/binary encoding. Current count enters only the decoder | **100% validation count accuracy**, 0 / 88,590. **99.9978% reused-test count accuracy**, 1 / 45,245; all 377 increments and 801 actual hits detected, one false hit. With count and contact fed back, **99.9912%**, 4 count errors across 52 life segments. Selected detector also has one false validation hit at saturated count 12. |
| Combined y, fractional y, and vy | Original y/vy pair frozen; 10→64→64→8 ReLU correction, cross-entropy; 5,384 fitted parameters | Current vy, predicted y displacement, and eight original vy probabilities. Original predictors use their established current-state contract | **100% validation** through all 78,806 eligible 128-step windows. Reused-test one-step joint errors **7→3 / 45,245**. Fully exact 128-step windows **98.6965%→99.4286%**, 511→224 failed windows / 39,203. Only vertical fields fed back; other state supplied and reference life boundaries enforced. |
| Life-loss terminal flag | 31→64→64→2 ReLU classifier, cross-entropy; 6,338 fitted parameters | Current RAM ball y, fractional y, and vy; scalar/binary encoding includes `y + vy`. No lives, action, or history inputs | **100% validation**, 0 / 88,638 errors, all 48 deaths exact. **100% reused-test accuracy**, 0 / 45,283, all 38 deaths on the exact step and no premature stops across 52 segments. Recorded current ball state; full-state feedback remains untested. Supersedes the older history-based probe with F1 0.299. |

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

## Auxiliary state and remaining updates

| Value | Current handling |
| --- | --- |
| Fractional ball y | Current input and supervision reconstructed by native replay. Its next value is now learned jointly with y: 3 fractional errors / 187,251 fresh transitions. Now fed back jointly with y and vy in the paired experiment below; full-state feedback remains untested. |
| Prior paddle-hit count | Current state and labels reconstructed and checked against native replay. Its capped next value is now predicted by a learned hit detector. One reused-test false increment affects four feedback transitions. |
| Brick-contact memory | Current value and labels reconstructed by native replay. Its next value is now learned with zero validation and reused-test errors, including contact-only feedback. Full-state feedback remains untested. |
| Paddle measurement, repeat and held-input fields | Reconstructed and present in the controller-annotated dataset. The closed x/charge paddle state does not require separate predictions of these fields under the audited reset and two-frame action contract. |
| Paddle velocity | Removed from the new dynamics input/output contract. Historical models and dataset columns retain it. |
| Lives, serve and respawn | Outside the current simulation scope. Each life ends at ball loss; later lives remain separate training segments. |

Horizontal speed and direction are now combined and integrated across the full
field in a saved model, with a learned spatial head correcting most missed
brick-triggered speed increases. Remaining work includes rare ball-state errors,
full-state integration, and wall refills. Learned predictors now cover termination, both positions,
including fractional y, both velocities, brick layout, contact memory, capped
hit count, and paddle width. The brick model preserves the y/velocity parent,
and the separate x model remains unchanged.
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
[vertical velocity](history.md#2026-09-18-discrete-vertical-velocity-with-a-frozen-horizontal-model),
[ball positions](history.md#2026-09-18-discrete-ball-displacements-and-coherent-fractional-y),
and [brick layout](history.md#2026-09-18-coherent-brick-layout-prediction).


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

Both position rows use the same new 64-episode test set, separate from the
velocity tests above. Both positions are simultaneously exact on **99.9850%**
of its 187,251 transitions (28 joint errors). Including the frozen vx/vy
predictors, all four ball quantities are simultaneously exact on **99.9701%**
(56 errors). These are one-step predictions with the other state inputs supplied;
no complete learned-state rollout is established.

The brick-layout test uses another **16 previously unused episodes**, separate
from every earlier test, preserving **64 untouched held-out episodes** for later
full-state evaluation. That experiment supplied current contact memory. Zero errors on
that finite test do not establish perfect recursive or wall-refill behavior.

The contact-memory experiment reuses the brick-layout test's 16 episodes and
preserves the 64 untouched episodes. It does not provide a new fresh-test score.
Both geometry seeds reach zero validation errors; select seed 2026 by the
predeclared name tie-break. With contact alone fed back, every contact and brick
layout stays exact on the reused test. The frozen ball predictors retain 15
jointly incorrect transitions, giving 99.9668% joint accuracy for contact,
layout, x, combined y, vx, and vy. Of 1,181 complete seven-step windows around
brick removals, 1,178 have every output exact. The three imperfect windows contain
existing ball errors. See [the contact experiment](history.md#2026-09-19-learned-brick-contact-memory-and-isolated-feedback).

The hit-count experiment uses the same 16 reused test episodes. The selected
model detects all 801 actual hits, including 424 after count saturation, but
adds one false hit at episode 1570, step 2228. It incorrectly changes count 4
to 5. The error persists for four scored transitions before the reference
segment boundary. Its intermediate-paddle prediction is correct at 61 px;
this is a remaining hit-classification error, also present in earlier ball
velocity predictions for that source. No test-directed refitting is performed.

Feeding count and contact together preserves exact contact and layout outputs
and leaves every frozen ball prediction unchanged. There are 18 jointly wrong
state transitions, up from 15 with supplied count, giving 99.9602% joint state
accuracy. All other state fields remain supplied. See [the count experiment](history.md#2026-09-19-learned-capped-paddle-hit-count).

The width experiment again reuses those 16 test episodes. Its selected head uses
only four existing fields and catches all 18 width changes, with zero width
errors or false widenings. Adding width feedback preserves every older model's
predictions exactly, including the existing four count-feedback errors and 18
joint state errors. Ball, paddle position/charge, and brick inputs still come
from reference data. The older history-based width probes missed every change;
the new result uses reconstructed fractional y and balanced transition sampling.
See [the width experiment](history.md#2026-09-19-learned-paddle-width-and-three-field-feedback).

The stop experiment includes life-ending transitions that the other state heads
exclude. It retains later lives as separate segments and treats censored endings
separately. All 38 reused-test deaths are detected on their true transition,
while all 14 censored segments remain unstopped. This closes the isolated stop
prediction gap using current vertical state; it does not establish correct
termination once ball prediction errors accumulate. See
[the termination experiment](history.md#2026-09-19-learned-life-loss-termination).

The first paired experiment combines y, its fractional component, and vy.
A frozen position head can predict the bounce correctly while the old velocity
head misses it. A learned velocity correction uses the position prediction to
resolve this validation disagreement. On reused test it leaves three incorrect
one-step y predictions, one also paired with incorrect vy. These seed the
remaining vertical feedback failures. Windows overlap, so window counts are not
independent game outcomes. No test-directed fitting was done. See
[the paired vertical experiment](history.md#2026-09-19-paired-vertical-position-and-velocity).

A later collision-displacement refinement was **not promoted**. It repaired
22 of 23 fitting-set y errors and improved development 128-step failures from
357 to 89, while retaining exact original validation. However, reused-test joint
errors rose from 3 to 4 and failed 128-step windows rose from 224 to 352.
The earlier `vertical-pair-20260919/coupling-s2026` checkpoint remains current.
The refinement's 256 development episodes were withheld from the new heads,
but had been used to train the frozen parents; their gains were not fresh-test
evidence. See [the rejected refinement](history.md#2026-09-19-collision-displacement-refinement-rejected-after-held-out-evaluation).

An explicit two-native-frame collision-timing model was also **not promoted**.
It predicts a joint velocity pair and derives y displacement from their sum.
With a calibrated frozen-position prior, development becomes exact, but reused
test joint errors rise from 3 to 4 and failed 128-step windows from 224 to 352;
mean endpoint y error rises from 1.5996 to 2.7291 pixels. The current checkpoint
and table remain unchanged. The development split has the same frozen-parent
training overlap as the preceding refinement. See
[the rejected timing experiment](history.md#2026-09-19-explicit-collision-timing-rejected-after-held-out-evaluation).

The follow-up ancestry audit establishes 400 development episodes with zero
training overlap across the current pair and all its learned dependencies,
preserving 64 untouched final-test episodes. On 1,152,531 nonterminal development
transitions, the unchanged pair makes 78 y errors, 68 vy errors, and 99 joint
errors, or **99.9914% exact one-step joint accuracy**. These previously inspected
episodes now serve development explicitly. This is neither a fresh-test score
nor a rollout improvement; the current checkpoint remains unchanged. See
[the split and baseline](training.md#vertical-development-split-with-untrained-episodes).
