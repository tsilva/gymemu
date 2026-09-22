# Learned state-model status

Updated 2026-09-22. This includes isolated predictors and partially combined
state-transition models. Full recursive emulation remains incomplete.
Validation selects checkpoints; reserved-test results are
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
| Combined ball motion: x, y with fraction, vx, vy | One `ball_motion` checkpoint containing the horizontal independent pair and vertical specialized pair; six CE objectives; 689,916 trainable parameters | Same 118-value current state, simultaneous updates; current non-ball fields supplied | **99.9400% exact joint new-validation accuracy**, 123 / 205,075 errors. **93.3265% fully exact 128-step ball-feedback windows**, 11,668 / 174,841 failed. Mean endpoint errors x **1.6682 px**, y **6.3050 px**. Selected new merge candidate; not a full-state emulator or reserved-test result. |
| Combined ball motion and brick layout | One `ball_bricks` checkpoint with specialized ball and 109-class brick-event heads; 746,046 trainable parameters in continuation experiments | Same 118-value current state; all outputs applied together | Retained composition: **99.9337% joint accuracy**, 136 / 205,075 errors; complete layout **99.9922%**, 16 errors. **92.5407%** exact 128-step ball/layout windows. Joint continuations regress layout accuracy and remain diagnostic. Contact/paddle/count state and reference life boundaries still supplied. |
| Combined ball, bricks and contact | One `ball_bricks_contact` checkpoint; existing ball/layout branches plus 86→128→128→2 contact head; 773,952 trainable parameters | Same 118-value state; 113 outputs applied together | **99.9293%** joint accuracy, 145 / 205,075 errors. Contact **99.9888%**, 23 errors. **92.2644%** exact128 windows. Paddle/count fields and life boundaries still supplied. |
| Combined ball, bricks, contact and hit count | One `ball_bricks_contact_count` container; added 96→256→256→256→2 hit classifier, 156,930 trainable parameters this stage; existing state branch frozen | Same 118-value state; 114 outputs applied together | **99.9171%** exact joint validation, 170 / 205,075 errors. Count **99.9834%**, 34 errors. **91.5089%** entirely exact128 windows. Paddle x/width/charge and life boundaries supplied. |
| Combined ball, bricks, contact, count and paddle width | One `ball_paddle_width` container; added 33→64→64→2 ReLU width head, 6,466 trainable parameters this stage; existing state frozen | Same 118-value source; 115 simultaneous outputs | **99.9171%** joint validation accuracy, 170 / 205,075 errors. Width **100.0000%**, 0 errors, 0 / 115 change errors. **91.5089%** entirely exact128 windows. Paddle x/charge and life boundaries supplied. |
| Combined ball, bricks, contact, count, width and charge | One `ball_paddle_charge` container; added 25→128→128→11 SiLU charge classifier, 21,259 trainable parameters this stage; existing state frozen | 118 state values plus current action; 116 simultaneous outputs. Charge uses paddle x, charge and action | **99.9171%** exact joint validation, 170 / 205,075 errors. Charge **100.0000%**, 0 errors. **91.5089%** entirely exact128 windows; 0 endpoint charge errors. Paddle x and life boundaries supplied. |
| Combined compact state including paddle position | One `ball_paddle_position` container; added 25→128→128→23 SiLU position classifier, 22,807 fitted parameters this stage; previous predictors frozen | 118 state values plus action; 117 outputs. Position uses current paddle x, charge and action | **99.9005%** exact joint validation, 204 / 205,075 errors. Paddle x **99.9834%**, 34 errors. **90.0675%** entirely exact128 windows, with all compact state fields fed back. Recorded actions and life boundaries supplied. |
| Combined compact state and life-loss termination | One `ball_life_termination` container; added 31→64→64→2 ReLU stop head, 6,338 fitted parameters this stage; state branch frozen | Same 119 state/action values; 117 state outputs valid only when stop output 117 is false. Stop uses current y, fraction and vy | Stop **100%** on 205,314 validation transitions: 239/239 deaths, no false stops. Combined **99.9006%**. Full feedback: exact death timing **169/239**; exact entire segments **141/246**. Recorded actions; missed deaths censored at reference end. |
| Life-loss terminal flag | 31→64→64→2 ReLU classifier, cross-entropy; 6,338 fitted parameters | Current RAM ball y, fractional y, and vy; scalar/binary encoding includes `y + vy`. No lives, action, or history inputs | **100% validation**, 0 / 88,638 errors, all 48 deaths exact. **100% reused-test accuracy**, 0 / 45,283, all 38 deaths on the exact step and no premature stops across 52 segments. Recorded current ball state; historical isolated result. Supersedes the older history-based probe with F1 0.299. |

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

On 2026-09-21, joint vertical feedback across this development set produces
9,027 failed 128-step windows out of 1,024,358, or 99.1188% entirely exact.
The targeted factorized paddle candidate lowers failures to 7,055, or 99.3113%
exact, and mean endpoint y error from 1.5756 to 0.9690 pixels. However, it
introduces one original-validation timing error, affecting 98 validation
128-step windows. It is **not promoted**; the current-model table stays unchanged.
No identical paddle inputs with conflicting targets were found across 536,838
fitting/development examples. All 64 final-test episodes remain untouched. See
[the feedback experiment](training.md#development-vertical-feedback-and-factorized-paddle-outcomes).

Adding eight explicit paddle-edge distances to the factorized predictor gives
7,026 failed development 128-step windows and 0.9173-pixel mean endpoint y error.
This is a small further gain over its 7,055-window, 0.9690-pixel control. Both
models still make 85 development one-step errors and the same single original-
validation timing error. The edge-feature candidate is also **not promoted**;
the current reference and 64 untouched final-test episodes remain unchanged.
See [the controlled feature experiment](training.md#paddle-edge-feature-experiment).

A matched comparison with `tsilva/gradlab-breakout-6127e81d` improves this paddle
branch without changing its architecture. Across two seeds and fixed 12,000-update
budgets, new-only fitting reduces mean joint errors from 41 to 11 on 96,005 old
development paddle sources and from 73.5 to 16 on 17,074 new-validation sources.
The 50/50 mixture averages 11 and 19 errors. New-only retains one old-validation
timing error in each seed; mixed seed 91 has zero and seed 2026 has one. This tests
training data for the existing branch with identical pretrained dependencies,
not the whole emulator. No model is promoted before post-error drift checks;
the current-model table remains unchanged. Both final-test target sets remain
reserved. See [the dataset comparison](training.md#checkpoint-trajectory-dataset-comparison).

Upper-screen continuation on the new dataset shows a tradeoff. With the improved
paddle branch frozen, two new-data runs reduce 62 new-validation upper y/vy
errors to 19/13, but raise 44 old-development upper errors to 60/54. Mean actual
128-step endpoint y error improves from 8.3158 to 3.0530 pixels on new validation
and worsens from 0.8729 to 1.5529 on old development. Matched old-data controls
improve both larger sets but also introduce legacy-validation errors. These
checkpoints are **not promoted**. See the
[upper-screen comparison](training.md#upper-screen-dataset-continuation).


The subsequent [fixed-size mixture](training.md#fixed-size-mixed-upper-screen-fitting)
uses 619,839 rows per fitting condition, the same model and update budget, and two
optimization seeds. Mean upper errors are 36/46.5 on old-development/new-validation
with old-only fitting, 57/16 with new-only fitting, and 33/22 with the 50/50 mix.
Mixed 128-step y drift is 0.8977/3.8842 pixels, compared with 1.5529/3.0530 for
new-only fitting. This improves the balance across datasets. It retains 5/4
legacy-validation upper errors and slightly worsens mean old-data drift versus
the 0.8729-pixel starting model, so neither mixed checkpoint is promoted. The
paddle branch, reference, datasets and final-test target reservation remain unchanged.


## Horizontal pairing on the new dataset

New-only fitting of independent x/vx predictors gives 103/112 joint errors on
205,075 new-validation sources, mean **99.9476%** exact. A shared-hidden-layer
pair gives 111/118 errors, mean **99.9442%**, with 231,706 rather than 447,962
trainable parameters. Both start from historical x features and frozen geometry
parents; this is not from-scratch fitting. The historical x/routed-vx baseline
has 231 joint errors on the same sources.

Entirely exact 128-step horizontal windows rise from 88.5330% to 93.8424% for
independent predictors and 93.6562% for shared layers. Mean endpoint x MAE falls
from 2.6104 to 1.4660 and 1.4061 pixels respectively. Other fields and reference
life boundaries remain supplied. The independent pair is the stronger accuracy
baseline for further experiments; sharing saves parameters but slightly increases
one-step and exact-path errors. Most remaining errors occur near the paddle.
Four-variable ball feedback is not established, and no historical reference is
replaced. See [the horizontal experiment](training.md#horizontal-pair-on-new-checkpoint-trajectories).


## Four-variable ball-state merge

The next merge combines the horizontal independent pair and vertical pair into
one `ball_motion` checkpoint with simultaneous x, combined y, vx and vy outputs.
The initial composition exactly reproduces both parents on all 205,075 new-
validation transitions. Joint continuation seed 91 reduces 129 errors to 123,
or **99.9400%** exact ball-state accuracy. Four-variable 128-step feedback improves
from 92.6608% to **93.3265%** entirely exact windows. Mean endpoint errors improve
from x 1.8691 / y 7.2169 to x **1.6682** / y **6.3050** pixels. Individual head
errors are 71 x, 31 y, 50 vx and 30 vy. Some heads regress slightly, and rare
large drift remains; this is an integration result, not perfect simulation.

Use `runs/ball-motion-20260922/candidate.pt` for the next merge, copied from the
fixed final seed-91 continuation. Seed 2026 slightly worsens y drift and stays
diagnostic. Parent files and the historical reference remain unchanged. The subsequent merge adds brick
layout. Non-ball fields and reference life boundaries
are still supplied; final-test targets remain reserved. See
[the merge record](training.md#four-variable-ball-state-merge).


## Ball and brick-layout merge

The merged `ball_bricks` model now emits ball state and all 108 brick cells
from one checkpoint. The composition preserves both parents' predictions exactly.
On 205,075 new-validation sources it has 123 ball errors, 16 complete-layout
errors and 136 jointly incorrect transitions, or **99.9337%** exact combined
accuracy. Coupled feedback gives **92.5407%** entirely exact 128-step windows,
with mean endpoint errors x 1.7137 / y 6.6627 pixels. Other state and reference
life boundaries remain supplied; wall refills remain unsupported.

Two joint continuations reduce ball errors to 121/122 but increase layout errors
to 46/50 and joint errors to 164/169, with worse recursive results. Keep the
original composition as `runs/ball-bricks-20260922/candidate.pt` and continue the
merge sequence without additional tuning. The following merge adds brick-contact memory. No dataset changes or final-test reads. See
[the merge record](training.md#ball-and-brick-layout-merge).


## Ball, bricks and contact merge

The `ball_bricks_contact` model now predicts all four ball fields, 108 brick cells
and contact from one checkpoint. On the migrated dataset it reaches
**99.9293%** exact joint validation accuracy (145 / 205,075
incorrect transitions); contact alone is **99.9888%** (23 errors).
Ball and layout errors are 123 and 16.
Coupled 128-step feedback gives **92.2644%** entirely
exact windows, with x/y endpoint MAE 1.7578/7.2773 pixels.

The selected checkpoint is `runs/ball-bricks-contact-20260922/candidate.pt`.
Both joint continuations regress accuracy, so retain the original composition.
The merge is complete; further tuning stays deferred. Both seeds reproduce the
prior ball/layout continuation weights exactly: adding contact loss did not
cause the existing brick regression.

The migrated revision preserves the old train/validation values and assignments.
No dataset writes or final-test reads. Paddle state, hit count and reference life
boundaries remain supplied. The following merge adds prior paddle-hit count. See
[the experiment](training.md#ball-bricks-and-contact-merge-on-migrated-data).


## Paddle-hit count container merge

Selected **initial composition** as `runs/ball-bricks-contact-count-20260922/candidate.pt`. Neither continuation meets the predeclared gate, so retain the initial composition. Both tuned runs improve one-step and exact-window accuracy but increase x endpoint MAE from 1.813656 to 1.815813 pixels, endpoint ball errors from 12,320 to 12,381, and layout errors from 6,609 to 6,611. This is a small trade-off, not a uniform regression.
It predicts ball state, all 108 bricks, contact and capped hit count together.
On 205,075 validation transitions, combined accuracy is **99.9171%** with
170 errors; count accuracy is **99.9834%** with 34 errors.
Raw hit detection has 47 errors, including 37 false
positives and 10 misses. Count saturation can hide detector errors.

Only the new count classifier was tuned. Existing ball, layout and contact
predictions remain unchanged. Joint feedback reaches **91.5089%** entirely
exact128 windows; mean endpoint x/y error is 1.8137/7.4769 pixels.
Paddle x, width, charge and life boundaries still come from the reference data.
No dataset changes or final-test reads. The following integration adds paddle width. See
[the full record](training.md#paddle-hit-count-added-to-the-transition-container).


## Paddle width container merge

Selected **width tuning seed 91** as `runs/ball-paddle-width-20260922/candidate.pt`. Both continuations reduce endpoint width mismatches from 960 to 713 without changing the other measured errors. Both qualify; the predeclared tie-break chooses seed 91.
The container now predicts ball, bricks, contact, count and width together.
Width accuracy is **100.0000%** over 205,075 validation transitions, including
115 / 115 real shrinks. Combined accuracy is
**99.9171%**, with 170 incorrect transitions.

Joint feedback gives **91.5089%** completely exact128 windows and
713 endpoint width mismatches. The latter can follow earlier
ball divergence. Mean endpoint x/y error is 1.8137/7.4769 pixels.
Paddle x, charge and life boundaries remain supplied. Dataset unchanged, final-test
targets unread. The following integration adds paddle charge with action input. See
[the full record](training.md#paddle-width-added-to-the-transition-container).


## Paddle charge container merge

Selected **initial composition** as `runs/ball-paddle-charge-20260922/candidate.pt`. Neither continuation supplies a strict improvement while passing every no-regression check. Charge is integrated into the new checkpoint with its original weights; all previously integrated predictors remain fixed.
Charge uses current paddle x, charge and action. On 205,075 validation transitions,
charge accuracy is **100.0000%**; combined accuracy stays **99.9171%**.
With charge fed back, **91.5089%** of 128-step windows are entirely exact,
with 0 endpoint charge errors. Current actions, paddle x and
reference life boundaries are supplied. Dataset unchanged; final-test targets
unread. The following integration adds paddle position. See
[the full record](training.md#paddle-charge-added-to-the-transition-container).


## Paddle position container merge

Selected **position tuning seed 2026** as `runs/ball-paddle-position-20260922/candidate.pt`. The continuation passes every predeclared no-regression check and supplies a strict improvement.
Paddle-position accuracy is **99.9834%** on 205,075 validation transitions;
combined accuracy is **99.9005%**. Entirely exact128 windows:
**90.0675%** with every compact state field fed back. Endpoint paddle
position errors: 43; charge errors: 0.
Recorded actions and reference life boundaries remain supplied; terminal
transitions are excluded. Dataset unchanged; final-test targets unread.
The following integration adds life-loss termination. See
[the full record](training.md#paddle-position-added-to-the-transition-container).


## Life-loss termination container merge

Selected **stop tuning seed 91** as `runs/ball-life-termination-20260922/candidate.pt`. Both continuations reduce false stops in 128-step survival windows from 3,526 to 3,522, with other stop-timing and exact-window counts unchanged. Both qualify; the predeclared tie-break selects seed 91. Their longer survival exposes 150 more state-transition errors in bounded windows and four more in full-segment evaluation; those raw counts have different numbers of simulated transitions.
The container now predicts every compact state field and whether to stop. Terminal
successor state is masked and never fed back. Stop accuracy on recorded sources
is **100%**, including all 239 deaths; combined one-step accuracy is
**99.9006%**. With full state feedback, death timing is exact
in **169 / 239** life segments and entire
trajectories are exact in **141 / 246** segments.
There are 62 early deaths, 8 missed
deaths and 5 false stops on censored
segments. Missed deaths are censored at the recorded boundary.

An offline audit of 217,487 source rows evaluated during selected full-segment inference finds **zero disagreements** between the neural stop flag and the native bottom-boundary timing rule applied to the same predicted inputs. Recorded-action full-segment timing failures therefore arise from earlier state-trajectory divergence in this evaluation. Native rules only audit the predictions; they never alter model outputs.

Dataset unchanged; final-test targets unread. Next: analyze the first state
mistakes in divergent rollouts before hard-example mining or shared-MLP fitting.
See [the full record](training.md#life-loss-termination-added-to-the-transition-container).
