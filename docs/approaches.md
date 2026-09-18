# Emulator approaches

The optional [paddle controller dataset columns](training.md#paddle-controller-dataset-columns)
provide both `source_` inputs and unprefixed successor targets. An approach using
them must explicitly adapt these columns and keep their timing distinct. They are
raw int32/boolean values, with normalization divisors in the annotation schema.
The annotator replays native rules offline to produce labels. A learned approach
must still define how its own controller state advances during recursive playback.
`paddle_transition_mlp.controller_fields` selects which of charge, measure, repeat,
and held reach the scalar/binary encoder. Current x and action always remain.
The default includes all four fields and preserves existing checkpoints. For the
pinned controller's allowed resets and two-frame actions, exhaustive reachability
checks establish that x and charge determine next x and charge. Separate learned
position and charge classifiers now have a bounded recursive paddle evaluation.
Charge checkpoints store `config.target=paddle_charge` and an ordered
`model_spec.delta_values` list. Decode logits through `model.delta(...)` and add
the result to current charge. Do not treat these outputs as pixels or call native
controller rules during model inference.

New state configs disable `predict_paddle_velocity`. The registry model default
remains true so historical checkpoint specs load unchanged. New models physically
omit the velocity input and head, while the existing state-vector slot stays zero
for cache/player compatibility. State loss and score code omit that slot. Keep
the flag in saved model specs when adding approaches or changing model families.

An approach is a complete way to train and run an emulator. It can own one model or
several models, and it declares an ordered list of training stages. The player only
needs its next-frame prediction interface.

## Included approaches

| Config | Models | Training stages |
| --- | --- | --- |
| `approach=direct` | Original action-conditioned RGB CNN | Predict the next frame with uniform pixel MSE |
| `approach=direct_actions` | RGB CNN with chronological action-history planes | Predict the next frame with uniform pixel MSE |
| `approach=ball_region` | Same action-history RGB CNN | RGB MSE plus a separately normalized target-ball region loss; recorded histories |
| `approach=ball_state` | Shared CNN with RGB and normalized ball-coordinate heads | RGB MSE plus masked coordinate MSE |
| `approach=scheduled_actions` | Same action-history RGB CNN | Uniform RGB MSE with progressively sampled generated context |
| `approach=autoregressive_ball_region` | Same action-history RGB CNN | RGB and ball-region losses at every valid rollout step; optional detached feedback |
| `approach=latent` | Frame codec and separate latent CNN | Reconstruct recorded frames, freeze the codec, then predict successor latents |
| `approach=reconstruction` | Single-frame codec only | Reconstruct the input frame with uniform RGB MSE |

The direct model preserves the original architecture, RGB input, action encoding,
stride padding, and sigmoid output. `model=direct_small` inherits `direct_cnn` and
changes only its width. Width overrides also work without creating another YAML file.

`direct_actions` selects `action_history_cnn`. It keeps the direct encoder/decoder
but widens the first convolution to accept one-hot planes for each action slot.
`model.action_history` includes the current action and defaults to the RGB history
length. The [action-history recipe](recipes.md#action-history-experiment) describes
the comparison and tuning controls.

`ball_region` changes only the objective of `direct_actions`. Its target-only
detector finds a unique solid nonblack rectangle with no same-color pixel touching
its four-connected boundary. Sprite dimensions come from `game.ball_sprite`; the
Breakout config specifies height 4 and width 2. Detection uses batched tensor
operations and does not run during inference. No detection or multiple candidates
produce an empty auxiliary mask. Padding clips at image boundaries, and each
region is normalized by its actual pixel count. Missing detections contribute
zero to the region term before averaging over the full batch. The detector assumes
exact recorded RGB, not resized, noisy, or generated frames. It cannot reliably
label merged or occluded sprites. These are explicit experiment limitations.

The reconstruction approach trains the codec alone. Its `reconstruct(frames)`
method supports recorded-frame playback, and its `evaluation_metric` is
`reconstruction_rgb_mse`. It has no predictive objectives or rollout probes.
Evaluation uses the input image as the target and selects the best reconstruction
checkpoint on held-out episodes. The player uses the declared `playback_modes`
capability, not the approach name. Every evaluated epoch is retained.

The latent approach is an experimental example of multiple models and stages, not a
claim of better emulation. Its codec compresses individual RGB frames spatially by
eight using three convolutions, then reconstructs them with three transposed
convolutions. A separate CNN takes the encoded history and current action and predicts
the next latent map. The codec decodes that map back to the original RGB dimensions.

Representation fitting uses only training-window targets, including episode starts.
The second stage freezes the codec's weights and keeps it in evaluation mode. It fits
latent MSE against encoded recorded successors. Bootstrap windows still contain zero
RGB history and the reserved `START` action. Zero-padded RGB frames pass through the
codec like other inputs; zero RGB does not imply a zero latent vector.

Every approach is evaluated on the same float32 next-frame RGB MSE with recorded
histories. Reconstruction MSE and latent MSE appear as stage `loss` values and must
not be compared directly to each other. The runner selects the representation stage's
best checkpoint by its validation reconstruction loss, and the final stage's best
checkpoint by RGB MSE. The next stage starts from the previous stage's selected weights.

`recipe=breakout_detached` enables `approach.options.detach_feedback=true` on the
autoregressive approach. The history values remain generated by the model. Before each
prediction, the entire history is detached from autograd, including older predictions
still inside the window. Each prediction contributes its own loss; valid-step losses
are averaged within each example before averaging examples. One backward pass accumulates
these gradients into the shared parameters. There is no gradient path from later losses
through earlier predictions. This cuts temporal credit assignment as well as temporal
gradient amplification. The option defaults to false for existing configurations and is
stored in checkpoint metadata. It does not alter the inference interface or evaluation.

## Add a model

The separate `gymemu dynamics` experiment uses `state_mlp` and `state_gru` from
the same explicit model registry. `gymemu/state_approach.py` owns its losses and
rollout behavior. Its state adapter and runner are separate from the RGB approach
interface because it has no image prediction or comparable RGB MSE. Its checkpoint
format is `gymemu-state-dynamics-v1`; use `gymemu dynamics play`, not `gymemu play`.
The GRU warms memory from the configured history and carries it through generated
rollout steps. The MLP receives a moving state/action window. Both terminate at
ball loss and preserve the requested-action contract. Extension and experiment
instructions are in [the training guide](training.md#single-ball-state-dynamics).

1. Implement a `torch.nn.Module` in `gymemu/models/`. Match the constructor and forward
   contract expected by its slot. The direct predictor takes `history`, `actions`, and
   `shape`; its forward takes RGB histories and action indices and returns one RGB frame.
2. Add a stable name to `MODELS` in `gymemu/models/__init__.py`. Add a model YAML with
   that `kind` and its constructor options. Runtime dimensions come from the dataset.
3. Select the model through Hydra, for example `model=my_predictor`, and test training,
   checkpoint round trips, original image geometry, bootstrap inputs, and playback.

Codec slots require `encode(frames)`, `decode(latents, shape)`, and `latent_channels`.
Dynamics slots accept `history`, `actions`, and `latent_channels` at construction and
return a latent map from a latent history and action index. A model with a different
contract belongs in a matching approach, rather than adding conditionals to playback.

## Add a pipeline

Implement an `Approach` in `gymemu/approaches.py`, or import it there from its own
module, and register a stable name in `APPROACHES`.

The interface consists of:

- `forward(history, action)`: normalized RGB `[B, C, H, W]` from RGB history
  `[B, history, C, H, W]` and integer action indices `[B]` when `action_history=1`.
  An approach may declare `action_history=A`, with `1 < A <= history`, to receive
  `[B, A]` tokens ordered from oldest to current. The last token is the action toward
  the target frame; preceding tokens are the most recent actions between history
  frames. Missing context is left-padded with the reserved `START` index, which means
  absence of an action rather than a game action. The shared loaders and player use
  this declared contract, also saved in checkpoints.
- `loss(history, action, target)`: a scalar training loss for the active stage.
- `prepare_stage(objective)`: set trainable weights and modes, then return the
  parameters the runner should optimize. Override `train()` when frozen models need
  to remain in evaluation mode after the runner toggles training.
- `validate_stages(stages)`: reject incompatible objectives or ordering. The base
  implementation checks declared objectives and requires a predictive final stage.
- `begin_epoch(epoch)`: update any training curriculum outside compiled computation
  and return a dictionary for metrics/checkpoint metadata. Epochs are one-based and
  restart per stage; the default returns an empty dictionary.
- `configure_training(compile=False)`: optionally prepare compiled execution helpers.
  The runner calls this once after constructing the model. Keep helpers out of the
  registered module tree so inference checkpoint tensor names stay unchanged.
- `reset_epoch_metrics()` and `epoch_metrics()`: optionally reset and return scalar
  diagnostics for each training/validation pass. Accumulate sample-weighted totals
  on device and synchronize when reporting, rather than once per batch. Diagnostics
  must not replace the runner's loss, comparison MSE, sample counts, or timing. Use
  nonpersistent buffers so counters do not become inference checkpoint weights.

An approach can declare `training_rollout_steps=K` to request extended training
prefixes. Training then receives RGB `[B, history+K, C, H, W]` and action tokens
`[B, K+1, action_history]`. The first `history` frames precede the first prefix
prediction; each following RGB frame is its recorded alternative. Actions include
all prefix predictions and the final supervised prediction. All-`-1` action rows
mark steps before the episode begins, distinct from a valid all-START bootstrap.
The approach must mask those nonexistent steps and preserve zero padding. Held-out
evaluation and inference retain the ordinary single-target input contract. Both
loaders implement these shapes without knowing the approach name.

An approach can declare `state_fields` to request the Gradlab scalar-label adapter.
The loader then appends `state_history` and `state_target` to each RGB/action/target
batch. Their shapes are `[B, input_history, D+1]` and `[B, D+1]`, where `D` is the
number of fields and the final column is availability. These tensors remain
normalized float32; only RGB uint8 tensors are divided by 255. `loss` and `evaluate`
receive these extra arguments. `evaluate` must still return a scalar stage loss
and an RGB prediction for the shared comparison metric.

For inference, state-aware approaches implement
`predict_step(history, action, state_history) -> (rgb, next_state)`, with next state
shaped `[B, D+1]`. The player maintains and resets both histories through this
declared contract, without dispatching on approach names. State fields are checked
against the reconstructed model at checkpoint load. Scenes must preserve the field
order and one state row per frame. The built-in `ball_state` approach predicts x/y
and returns availability one for generated states. The current Gradlab adapter
requires finite scalar normalized labels and masks only the unlabeled initial frame;
other encodings or missing successor labels need an explicit adapter change.

`scheduled_actions` builds replacements in order with detached predictions and samples
whole frames independently per example and step. Its probability buffer is excluded
from inference weights. Additional constructor settings live in `approach.options`
and pass only to the registered approach constructor, never arbitrary Python targets.

Its optional selective implementation groups each example's first chosen replacement,
then its second, and so on. It gathers only the preceding history from a mutable
timeline and writes predictions back after each group. This preserves within-example
dependencies while skipping discarded predictions. It relies on the registered CNN
being independent across batch rows; batch normalization or stochastic predictor
layers would require a new equivalence audit. Before-episode steps remain zero.

Add an approach config with `kind`, named `models`, and `stages`. Every stage has a
unique `name`, an `objective`, `epochs`, and `learning_rate`. Register all models as
submodules so `state_dict()` includes everything needed by inference. The optimizer
registry supports Adam and AdamW; new optimizer families, sequence targets, or non-frame
objectives may require extending this interface. Those are not implemented through
arbitrary YAML targets.

Config construction and checkpoint reconstruction share the explicit registries.
Checkpoint files contain weights and plain metadata, loaded with `weights_only=True`.
They never import a `_target_` supplied by the checkpoint. Preserve registered names and
constructor compatibility, or introduce a new checkpoint version when those change.

## Experiment discipline

Hydra can sweep approaches, widths, history lengths, seeds, and individual stage settings.
A two-stage run with ten epochs per stage does twice as many training passes as a direct
run with ten epochs. Compare training budgets as well as MSE. `summary.json` records
optimizer steps, samples seen, parameter counts, and elapsed stage time; it does not
include dataset loading time.

Use held-out data for model selection and retain a separate test partition for final
claims. Evaluation uses recorded histories; interactive playback feeds generated frames
back into the model and can accumulate errors. Inspect rollouts before calling a model
playable. See [history.md](history.md) for the limits of finite frame stacks.

`scheduled_ball_region` combines `ScheduledSamplingApproach` and
`BallRegionApproach`. Scheduled training calls `prediction_loss` after building
context; the combined approach applies `joint_loss` there. Cooperative construction
passes ball options to the ball approach. Validation inherits the ball approach's
single recorded-history forward and joint loss. The runner and player need no
approach-specific behavior.

### Future-target sequences

An approach can declare `training_future_steps > 0` to request a clean history plus
right-padded future targets from `Windows`. Actions have shape `[B, K, A]`, targets
`[B, K, C, H, W]`; a negative final action token marks an invalid future position.
This contract is mutually exclusive with extended-prefix scheduled sampling and
auxiliary state inputs. Evaluation continues to supply ordinary one-step examples.
The approach owns curriculum, unrolling, gradient flow, and valid-step reduction.
`autoregressive_ball_region` implements this contract using the existing action-history
model and ball-region objective. No approach-specific behavior belongs in the player.

The isolated `state_probes.py` experiment constructs registered state models but
optimizes exactly one target through `SingleTargetApproach`. The optional
`state_paddle_context_probe` consumes extra recorded paddle history recovered by
`RecordedPaddleContext`. Its `gymemu-state-probe-v1` checkpoints are diagnostic and
are rejected by the joint state player. This keeps input-sufficiency experiments
separate from joint rollout training; see the isolated-target section in the
[training guide](training.md#isolated-state-targets-and-input-sufficiency).


`gymemu/state_probe_diagnostics.py` owns isolated event strata, per-event baseline
comparisons, and training-only event sampling. These labels never enter predictor
inputs. `train_target` can initialize a compatible probe checkpoint with a fresh
optimizer for matched sampling/learning-rate experiments. Event diagnostics remain
outside the shared RGB runner and the joint state player.


`state_hidden_probe` extends the context probe with five reconstructed internal
source-state values. Its explicit mode masks select controller values, paddle-hit
count, both, or a zero-input control with the same parameter count. `HiddenInputs`
loads separate arrays with checked provenance; the isolated approach owns passing
those values to the registered model. `gymemu/state_hidden_context.py` prepares
these diagnostic inputs without changing the dataset or the joint player. Keep
any future learned update for internal state in an explicit approach; do not call
the diagnostic native-code reproduction inside neural playback.


The separate `paddle_transition_mlp` registry entry consumes six current native
values: paddle x, four controller variables, and requested action. Its scalar or
hybrid binary encoding contains no game transition rules. The dedicated
`paddle_transition_training` experiment owns native-displacement MSE or categorical
cross-entropy, split-aware loading, and validation checkpoint selection. These
small sufficient-input fits do not branch the shared RGB runner or player. Their
`gymemu-paddle-transition-v1` checkpoint contract is distinct from full-state
models because controller memory and the other game variables are not outputs.

`controller_history_mlp` predicts one current internal controller variable from
observed paddle history. `controller_history_inputs` builds source-time inputs
without current/future actions or observations, preserving life boundaries. The
diagnostic experiment trains separate charge and measurement classifiers with
class vocabularies from training targets only. Their
`gymemu-controller-history-probe-v1` checkpoints are diagnostics, not playable
state models. They do not update controller state or call native game rules.

`ball_velocity_mlp` is an isolated next-horizontal-velocity probe. Its 118 source
values contain ball x, integer RAM y, both ball velocities, paddle x and width,
charge, prior paddle-hit count, fractional y in eighths, brick-contact memory,
and 108 brick cells. Paddle velocity, images, and future observations are absent.
The model owns scalar or scalar-plus-bit encoding and either eight categorical
velocity logits or one residual velocity output. `memory=False` masks hit count,
fractional y, and brick contact while retaining charge and the same architecture.
The optional `relative_offset` encoding adds ball x minus paddle x, and its bits
for hybrid inputs. This is a relation between existing source coordinates, not a
new state variable or a native collision rule.
The optional `paddle_only=True` contract accepts exactly the first nine source
values, physically excluding brick contact and layout. With hybrid encoding and
relative offset this gives 85 encoded inputs, two 128-unit SiLU layers, and eight
logits (28,552 parameters). It is a diagnostic for the source-selected descending
paddle region, not a replacement for the full-game input contract. The default
remains 118 source values; checkpoints retain this choice in the registry spec.
`objective="direction"` replaces the velocity output with two logits for left
and right. Its `predict()` returns a sign, -1 or +1, not a velocity magnitude.
The compact hybrid direction model has 27,778 parameters. Classification and
regression defaults and their checkpoint contracts remain unchanged. Diagnostic
drivers may also sum an eight-class model's probabilities by direction; this
decoding choice must be stored alongside that diagnostic checkpoint.
Native rules may reconstruct source inputs for diagnostics but must not enter
the neural forward pass or application playback. The experiment's
`gymemu-ball-velocity-probe-v1` checkpoint stores the registry spec and all model
weights; it is not a complete state-player checkpoint. Collision-memory updates
and the other state variables remain outside this model's outputs.


`ball_velocity_mlp` also accepts `spatial_features=True`. It appends eight
coordinate, relative displacement, and fractional-position features computed
from the current source and constant-velocity proposals. This gives 93 inputs
and 28,802 parameters for the compact direction model. The default is false,
so previous checkpoint shapes stay unchanged. `memory=False` also masks the
fraction used by these features.

`ball_direction_geometry` is a direction-only diagnostic with two learned
components. A frozen `paddle_transition_mlp` predicts one-native-frame paddle
displacement from current paddle position and charge. Its classification
vocabulary is determined from training targets. A separate direction MLP uses
that predicted position and the nine current source variables. Both components
are included in the checkpoint. Calling `train()` keeps the paddle component
frozen and in evaluation mode.

Geometry encoding exposes constant-velocity coordinate proposals and relative
ball/paddle positions. It contains no collision decisions or controller update
rules. `scalar`, `hybrid`, and `hybrid_absolute` produce 16, 62, and 84 inputs.
The last encoding includes binary digits of absolute ball x and its motion
proposal as well as the relative-coordinate digits. Width, depth, and ReLU or
SiLU activation are stored in the registry specification. Output remains a
left/right sign, not velocity magnitude or a complete next state.

The intermediate paddle training target is derived from recorded controller
values using the audited native rule, in RAM. This adds auxiliary supervision
to the experiment. It is distinct from the recorded next-direction label and
must be disclosed when comparing models. Native rules do not run during model
inference. These diagnostic checkpoints do not yet implement recursive playback.

`ball_horizontal_velocity` adds a learned speed head to a frozen
`ball_direction_geometry`. Its nested `direction` specification contains the
geometry model's constructor arguments, not an import target or checkpoint path.
The checkpoint stores both networks and the intermediate paddle model. Calling
`train()` keeps the complete direction component frozen and in evaluation mode.

The speed head uses the same encoded current-state geometry and predicts four
magnitudes, 0.5, 1, 1.5, and 2, with cross-entropy. `forward()` returns speed
logits, `predict_speed()` returns magnitude, and `predict()` multiplies learned
direction by learned magnitude to return signed horizontal velocity. Width,
depth, and activation apply only to the speed network. This remains a
descending near-paddle diagnostic; it does not establish full-game velocity
accuracy or recursive state updates.

`ball_horizontal_router` composes that frozen near-paddle predictor with a
`ball_velocity_mlp` full-field classifier. Its input is the existing 118-value
physical-state contract, including brick-contact memory and 108 brick cells.
The near-paddle component receives only the first nine values. Routing depends
only on current source state: integer RAM y in [160, 183] and vy > 0 selects the
near-paddle component; all other sources select the full-field component.
Neither successor state nor collision-event labels participate in routing.

Both `forward()` and `predict()` return signed native horizontal velocity.
For training, obtain logits through `global_model(source)` on sources outside
the gate. Calling `train()` keeps all near-paddle parameters frozen and in
evaluation mode. The nested `global_model` and `paddle_model` specifications
contain constructor arguments for these two fixed registry classes, and the
checkpoint stores every component's weights. This is a one-step vx diagnostic;
it supplies no position, vertical-velocity, memory, or terminal-state updates.

`ball_acceleration` wraps a frozen `ball_horizontal_router` with an isolated
upper-field speed head. The source-only domain is integer RAM y <= 100 and
incoming horizontal magnitude < 2. The observed target vocabulary there is
retain the incoming magnitude or increase it to 2. The classifier learns which
outcome occurs; its `forward()` returns two logits. `predict()` combines the
chosen magnitude with the frozen parent's predicted sign, and preserves parent
predictions outside that domain. It uses no successor/event inputs or native
collision decisions. The domain and vocabulary are specific to this Breakout
transition contract and require dataset auditing before reuse elsewhere.

The head receives ball x, RAM y, vx, vy, fractional y, brick-contact memory,
and the 108-cell brick layout. Scalar/binary encoding yields 149 features.
Optional `geometry=True` adds current and constant-velocity proposed coordinates
and their fractional/binary encodings, for 179 features. These are arithmetic
features of current inputs, not collision rules. The complete frozen parent is
stored in the checkpoint under the nested `base` constructor specification.
Calling `train()` leaves the entire parent frozen and in evaluation mode.

`ball_acceleration_spatial` uses the same frozen parent, gate, and two outcomes.
Instead of a flat layout vector, it describes each of the 108 brick cells using
13 features: ball-to-cell offsets now and after a constant-velocity proposal,
vx/vy, contact memory, grid row/column, and fractional coordinates. A shared
ReLU MLP processes every cell; absent cells are masked before max pooling. The
pooled vector and six normalized current-state scalars feed a two-class head.
`encode()` returns cell features, global scalars, and the occupancy mask;
`forward_encoded()` supports caching that fixed encoding in RAM during training.

Cell centers are fixed geometry of the 6×18 Breakout layout (x = 11.5 + 8c,
RAM-coordinate y = 50.5 + 6r). This is an explicit spatial prior, not additional
recorded state or native collision logic. The network still learns when speed
changes. The constructor stores cell width/depth and head width in its registry
specification; checkpoints contain all learned and frozen components.

`ball_vertical_velocity` adds discrete next-vy prediction while preserving the
complete frozen `ball_acceleration_spatial` horizontal model. It accepts the same
118 current-state values. `forward()` returns logits over the training-derived
signed-vy vocabulary; `predict()` decodes vy, and `predict_horizontal()` exposes
the unchanged horizontal prediction. Neither method updates positions or memory.

Routing uses only current RAM y and vy: y <= 100 selects the upper-field head;
160 <= y <= 183 with vy > 0 selects the paddle head; remaining states select a
small flight head. The audited nonterminal data has no vy changes in that last
region. The flight network is nevertheless trained from observed training pairs.
Event labels are used for evaluation and sampling, never inference routing.

The upper head has a separately trainable copy of the spatial cell encoder,
followed by masked max pooling and 71 global scalar/binary/geometric features.
The paddle head combines the frozen direction model's 84 geometry features with
charge as one scalar and 12 bits. This retains charge information even when the
learned intermediate-paddle estimate errs. The nested `horizontal` specification
contains constructor arguments for the fixed registry class. Checkpoints store
every component; calling `train()` keeps the horizontal parent in evaluation
mode with gradients disabled. This remains a one-step diagnostic model.

`ball_position` predicts one ball-coordinate displacement using the same
118-value current-state contract. Set `axis` to `x` or `y` and pass the signed
eighth-pixel displacement vocabulary observed in training. `forward()` returns
class logits, `predict_displacement()` decodes a displacement, and `predict()`
adds it to the current coordinate. For y, the coordinate is RAM y plus its
fractional eighth-pixel part; `predict_y_parts()` returns the coherent integer
RAM y and fractional remainder. Native collision rules never run in inference.

Its complete `ball_vertical_velocity` parent, including horizontal dynamics,
is stored under the nested `vertical` constructor specification and stays frozen
in evaluation mode. `predict_velocities()` exposes those unchanged predictions.
Separate trainable upper-field and paddle heads use the parent's spatial and
paddle feature encoders. The source-only region boundaries are unchanged.
The remaining region's x head receives 31 scalar/binary current/proposed-x and
vx features, allowing learned wall behavior; its y head receives current vy.
Position heads do not consume recorded successor velocities. Fractional-y
training labels derive from audited native replay, while integer positions
are checked against recorded successors. This is diagnostic supervision, not
an additional dataset column or a closed recursive simulator.

`brick_layout` predicts the next 108-cell layout using a categorical update:
class zero keeps the layout, and classes 1–108 remove the corresponding cell.
Use this model only after auditing that each transition contains at most one
removal and no additions. It cannot represent wall refills or multiple removals.
Absent cells are masked from the removal logits; an empty layout can only select
no change. These are output constraints, not native collision decisions.

The nested `position` specification constructs a frozen `ball_position` parent.
Its fixed spatial encoding uses ball x, RAM y, vx, vy, fractional y, contact
memory, and the 108 current brick cells. Paddle/controller fields are ignored by
this head. A trainable shared cell encoder supplies local features and an
occupancy-masked max pool. A shared removal scorer combines each cell's features
with that global context; a separate head scores no change. There is no event-
or successor-based routing. `forward()` returns 109 logits, `predict_event()`
selects an update, and `predict()` returns the complete next layout. Checkpoints
include all parent weights, and `train()` preserves the parent's evaluation mode
and disabled gradients. Contact memory remains a supplied input, not an output.
