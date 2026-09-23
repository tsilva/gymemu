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

`brick_contact` adds a next-contact classifier around a frozen `brick_layout`
parent. Its nested `layout` specification includes every parent dependency.
The input remains the 118-value current state. The parent supplies 71 encoded
ball-state values and two probabilities, no removal versus any removal. The
latter comes from the parent's predicted logits, never the target event.
A 73→128→128→2 ReLU network predicts inactive or active next contact.
With `collision_geometry=True`, append 13 cell-geometry features weighted by
the frozen removal probabilities. The classifier then has 86 inputs and can
distinguish predicted contacts with different bricks. This adds no dataset field.

`encode()` computes the frozen features under `no_grad`; `forward_encoded()`
allows those features to be reused in RAM during isolated training. `forward()`
accepts the full current state and returns two logits, while `predict()` returns
a binary integer. `train()` keeps the complete layout and ball parent in eval
mode with disabled gradients. Contact labels derive from offline native replay;
inference executes no native collision rules. This model needs the current
contact value. It does not infer an initial value from images.

`paddle_hit_count` predicts a paddle-hit event and decodes the next count as
`min(12, current_count + predicted_hit)`. Its nested `vertical` specification
constructs a frozen `ball_vertical_velocity` parent. The established paddle
encoder supplies geometry, a learned intermediate paddle position, and charge
features. Remove its prior-count scalar to form 96 classifier inputs, then use
96→256→256→256→2 ReLU layers. Hit detection is independent of current count,
contact memory, and brick layout. The decoder still consumes current count.

`forward()` returns two hit logits. It scores sources in the established region
of RAM y from 160 through 183 with positive vy, and emits no hit elsewhere.
Audit that every hit lies within this broad source-only region before using the
model at a new cadence. `predict_hit()` returns the binary event, including hits
after count saturation; `predict()` returns the capped count. The decoder does
not handle life resets or termination. `encode()` and `forward_encoded()` allow
RAM-only feature reuse during isolated fitting, and `train()` keeps the parent
frozen and in eval mode. No native collision rule or successor label runs during
inference.

`paddle_width` predicts 12-pixel or 16-pixel width directly. It accepts the shared
118-value current state but reads only current width, integer RAM ball y,
fractional y, and vy. Combine the y components into one eighth-pixel coordinate.
Three normalized scalars, 11 coordinate bits, six signed-velocity bits, and a
current-narrow indicator form 21 encoded features. A 21→64→64→2 ReLU network
returns width logits; `predict()` maps the argmax to 12 or 16.
The selected `proposal=True` variant appends normalized `y + vy` and its 11
coordinate bits, producing a 33→64→64→2 head. This motion proposal uses current
inputs and is not a simulated collision or a successor-state input.

This standalone model has no trainable dependency on existing models. It uses no
images, actions, horizontal state, brick cells, contact flag, hit count, or
predicted successors. It runs no native ceiling rule and imposes no hard-coded
monotonic width update. Train it on nonterminal within-life transitions and
handle life initialization separately. `encode()` and `forward_encoded()` permit
RAM-only feature reuse during isolated training.

`life_termination` is a standalone binary stop head. It accepts the shared
118-value physical source state but reads only RAM ball y, fractional y, and vy.
Two normalized scalars, 11 combined-y bits, six signed-velocity bits, and a
normalized constant-velocity proposal `y + vy` with 11 bits form 31 features.
The 31→64→64→2 ReLU network has 6,338 parameters. `forward()` returns logits;
`predict()` returns a boolean terminal decision. `encode()` and
`forward_encoded()` support RAM-only feature reuse. No native loss threshold,
collision rule, lives counter, or successor label executes during inference.

Train this head on surviving transitions **and** transitions ending a life.
The target is the existing life-loss boundary; truncation, invalid data, and
other censored endings are not deaths. At a predicted terminal transition, a
future integrated simulator must stop the life and discard the state heads'
unsupported successor outputs. Their training still excludes terminal targets.
This model is registered for isolated evaluation; it is not yet wired into the
shared runner or player. Its current evidence uses recorded ball state and
causally reconstructed fractional y at the fixed two-native-frame cadence.

`vertical_ball_pair` composes a `ball_position` specification with `axis=y` and
its existing vertical-velocity parent. `predict()` returns combined y and vy
from the same source state. Only after both outputs are computed should playback
split y into integer RAM y and fractional eighths and update all three fields.
The wrapper introduces no native transition rules or shared-runner branches.

With `coupling_width=0`, both original outputs are returned. The wrapper permits
training their y/vy heads while keeping horizontal and intermediate-paddle
dependencies frozen. The selected `coupling_width=64` variant instead freezes the
complete original pair. A 10→64→64→8 ReLU correction classifies next vy using
current vy, the frozen y predictor's displacement, and eight frozen vy
probabilities. `coupling_encode()` supports RAM-only feature reuse; neither a
recorded successor nor a true collision event enters these features. Each
original predictor runs once per forward call. The only trained component is
the 5,384-parameter correction, and y is unchanged.

`evaluate_vertical_pair()` in `gymemu/vertical_pair_eval.py` compares reference,
y-only, vy-only, and joint feedback. It starts at every eligible source and
scores horizons 1, 8, 32, and 128 only when the complete window stays in a
contiguous life segment. Supply explicit life IDs in addition to episode and
step IDs. Input rows exclude terminal transitions. Other state fields remain
recorded or reconstructed, and reference boundaries stop rollouts. Exact
reference-input predictions are reused; changed inputs run the pair again.
Tests compare this optimization with brute-force feedback across life boundaries.

`vertical_displacement_refinement` wraps a frozen `vertical_ball_pair` with a
coupling head. It adds residual logits to the original y displacement classes
in the existing upper-field and descending-paddle regions. Flight logits remain
unchanged. Both residual heads initialize their last layer to zero, so the
initial model reproduces the parent pair exactly.

For the upper field, reuse the frozen y encoder's pooled brick representation
and 71 global values. For the paddle region, reuse its 97-value paddle encoder.
Append the original 19 y probabilities and eight coupled vy probabilities.
The default heads are 162→64→64→19 and 124→64→64→19 ReLU networks, totaling
29,222 trainable parameters. No successor state or true event label enters these
features. `encode()` exposes the frozen features and original logits for
RAM-only fitting.

After correcting the y logits, use the corrected predicted displacement in the
existing frozen velocity coupling. This keeps y and vy predictions connected
without changing the original pair's weights. `forward()` returns y-displacement
and vy logits; `predict()` returns combined y and vy. `train()` keeps the complete
parent in evaluation mode. The checkpoint saves the parent and both residual
heads, and loading uses the explicit model registry.

For the calibrated refinement, multiply each residual head's final linear weight
and bias by the development-selected scale before saving. This scales the
correction to class scores, not the predicted movement or the physical y value.
The output still selects one of the original eighth-pixel displacement classes.
No extra scale must be applied after loading the checkpoint.

`vertical_collision_timing` predicts a joint pair of vertical velocities, one
for each native frame of the fixed two-frame transition. Each of eight velocities
can occur in either frame, giving 64 output classes. Decode next combined y as
current combined y plus the two predicted movements; next vy is the second
predicted velocity. Thus an early bounce and a late bounce can share an outgoing
velocity while producing different displacements. No native collision/controller
rules run during inference.

The original `vertical_ball_pair` is frozen. Reuse its y model's pooled spatial
features and paddle encoding. Upper-field and paddle heads are
135→128→128→64 and 97→256→256→256→64 ReLU networks. Initialize their hidden
layers from the loaded y model and train copied weights. Joint velocity-pair
cross-entropy can be combined with timing NLL. `timing_log_probabilities()`
marginalizes outcomes into unchanged, first-frame change, second-frame change,
or changes in both frames, relative to current vy.

`active_regions` explicitly selects upper and/or paddle regions. All remaining
regions use the original pair, including ordinary flight. This permits testing
one timing head without accepting regressions from the other. `encode()` exposes
source-only frozen features for RAM-only fitting, and `predict()` returns
combined y and vy. The full parent stays in eval mode during training.

An optional nonnegative `prior_weight` adds the original y model's log probability
for each candidate pair's summed displacement to the timing head's logits.
Velocity pairs whose sum is absent from the original displacement vocabulary
receive a finite log-prior floor of -30. This combines learned scores before
choosing one coherent pair; it does not average physical positions. The weight
and active regions are stored in the model specification. A zero weight preserves
the standalone timing model, and all lookup buffers are derived from saved
velocity/displacement vocabularies.
## Factorized paddle vertical prediction

The research-only `paddle_vertical_pair` registry model wraps a frozen
`vertical_ball_pair`. It replaces predictions only in the existing descending
paddle region. It uses the same 97 encoded source features, including charge and
fractional y, and copies the position model's hidden paddle layers for training.
The current experiment uses a shared 97→256→256→256 ReLU trunk with separate
three-class timing and four-class outgoing-velocity heads, 158,471 trainable
parameters in total.

Timing predicts no bounce, a bounce before the first native movement, or a bounce
before the second. Outgoing negative velocities come from fitting targets.
For no bounce, displacement is twice current vy and next vy stays current. For a
first-frame bounce, displacement is twice the predicted outgoing vy. For a
second-frame bounce, displacement is current vy plus predicted outgoing vy.
Both bounce cases use the predicted outgoing vy as next vy. This arithmetic
decodes learned classifications; no collision thresholds or native transition
rules execute during inference.

Training uses categorical timing supervision and speed supervision only for
bounce examples. The preparer must verify that this decomposition exactly fits
the native movements in the selected source region. The complete parent remains
frozen and in evaluation mode, including outside-region fallback predictions.
This experiment does not change the direct approach, runner, or player.

With `edge_features=true`, append eight signed raster-edge distances to the
existing 97 features. Four distances describe the current geometry; four use
the constant-velocity ball proposal and frozen learned intermediate-paddle
position. Each group contains ball-right minus paddle-left, paddle-right minus
ball-left, ball-bottom minus paddle-top, and paddle-bottom minus ball-top,
divided by 16. RAM-coordinate paddle bounds are 180..183; the ball raster extends
one pixel right and three pixels below its integer position. Paddle width comes
from the source state. These are arithmetic distances; they do not return a
collision decision or execute a game transition.

The resulting trunk is 105→256→256→256 with the same two output heads and
160,519 trainable parameters. `initialize_trunk_from_pair()` copies loaded
parent hidden weights and zeros the eight added input columns. Construction
preserves the control's random stream, including output-head initialization.
Floating-point summation order may differ after widening the input matrix.
The optional flag defaults to false so earlier factorized checkpoints load
with their original 97-feature architecture.

## Horizontal position/velocity pair

`horizontal_ball_pair` predicts next ball x and vx from the same immutable
118-value source state. Its specification contains an `axis="x"` position
specification, a finite unique `velocity_values` vocabulary and a `shared` flag.
The existing x-displacement classes and new vx classes have separate output
heads. Upper, near-paddle and flight routing uses current state only.

With `shared=true`, the two objectives share the existing position cell encoder
and region hidden layers. With `shared=false`, the velocity branch has its own
copies of those hidden layers, allowing independent objectives in one loader
pass. Source encodings are identical. The nested vertical/horizontal geometry
dependencies stay frozen and in evaluation mode in both variants. Training code
must copy loaded position hidden weights into the independent branch when using
pretrained initialization; constructor copies only its initial random weights.

`forward` returns displacement and velocity logits; `predict` returns N-by-2
next x/vx. Both outputs see current state, and neither receives the other's
recorded successor. No native rules run at inference. Complete model specs and
weights reload through the registry. This diagnostic model is not wired into
the RGB player or shared training runner.

`evaluate_horizontal_pair` supports reference, x-only, vx-only and joint feedback,
with reference boundaries and the other state fields supplied. It scores only
windows that remain within a contiguous life. Reusing reference predictions
until their inputs change is exact; tests compare against brute-force recursive
prediction after errors, across multiple life segments.

## Four-variable ball transition

`ball_motion` composes `horizontal_ball_pair` and `paddle_vertical_pair` under
one registered checkpoint. Its `horizontal` and `vertical` constructor options
are the corresponding specifications without their `kind` keys. `predict`
returns N-by-4 values ordered x, combined y, vx, vy. Split combined y into its
floor and eighth-pixel remainder when updating the 118-field state. Every branch
reads the original current state; outputs are applied simultaneously.

`forward` exposes horizontal logits, upper y/vy logits, paddle timing/speed
logits and source-only routing masks for joint training. The horizontal branches,
vertical upper cell encoders/heads and vertical paddle trunk/heads train. Vertical
flight, velocity correction, geometry and intermediate-paddle dependencies stay
frozen. Each objective uses its appropriate current-source subset; paddle speed
supervision uses true bounces only. This composition preserves specialized
networks rather than forcing additional parameter sharing.

`evaluate_ball_motion` supports reference, horizontal-only, vertical-only and
four-variable feedback. Its default is four-variable feedback. All non-ball
fields and reference life boundaries remain supplied. Tests compare the optimized
evaluator with brute-force coupled rollouts, including fractional y and cross-axis
error effects. This model does not yet predict termination, bricks or paddle
state and is not connected to the RGB player.

## Ball and brick-layout transition

`ball_bricks` combines a `ball_motion` specification and a `brick_layout`
specification under `motion` and `bricks`, without nested `kind` keys. Its single
`predict` call returns 112 values: x, combined y, vx, vy, then the 108 brick cells.
Both branches see the original source state. The ball does not receive predicted
bricks within the same step, and the brick predictor does not receive newly
predicted ball coordinates. Apply all outputs to the next state together.

`forward` preserves the ball model's named logits and adds 109 brick logits,
representing no change or one occupied-cell removal. Train the brick cell encoder,
removal scorer and no-change head alongside the ball model's trainable modules.
The brick parent and existing ball dependencies remain frozen. Audit no additions
and at most one removal before using this contract; wall refills are unsupported.

`evaluate_ball_bricks` offers reference, ball-only, bricks-only and joint feedback.
Its default feeds all 112 outputs back, with fractional-y conversion, while
supplying other state and life boundaries. It counts complete-layout errors and
individual cell errors separately. Tests compare coupled ball/layout feedback
against brute force and verify that predictions cannot add or remove absent bricks.

## Ball, bricks and contact transition

`ball_bricks_contact` composes `state` (a `ball_bricks` spec) and `contact`
(a `brick_contact` spec). Its 113 outputs are x, combined y, vx, vy, 108 brick
cells, then next contact. Every branch reads the original 118-value source.
The contact head retains its own frozen layout dependency; it does not consume
the other branch's successor output. Its classifier is 86→128→128→2 ReLU,
adding 27,906 trainable parameters to the 746,046 ball/layout parameters.

Use `evaluate_ball_bricks(..., contact=True)` with N×113 targets. Joint feedback
updates input contact column 9 from output column 112, brick input columns
10:118 from outputs 4:112, and the ball fields together. Contact errors are
reported separately. Other modes retain reference contact for ablations.
Life boundaries remain explicit. This experiment does not wire state inference
into the RGB player or predict termination, paddle state or paddle-hit count.

## Ball, bricks, contact and paddle-hit count transition

`ball_bricks_contact_count` wraps `state`, a `ball_bricks_contact` specification,
and `count`, a `paddle_hit_count` specification. Its output appends capped count
at column 113, giving 114 values. The complete state branch stays frozen/eval;
this stage trains only the added 96→256→256→256→2 ReLU hit classifier. Its
vertical geometry dependency also stays frozen. Supervise actual hit/no-hit,
including hits at count 12, and decode `min(12, current_count + predicted_hit)`.
Every branch reads the original current source. Shared representations remain
postponed until the state-transition container is complete.

Use `evaluate_ball_bricks(..., contact=True, hit_count=True)` with N×114 targets.
Joint mode feeds count output 113 into source column 7 along with ball, brick
and contact predictions. Other modes retain reference count. Contact must be
enabled when count is enabled, keeping output indices explicit. Report count
errors separately from joint, contact, layout and ball errors. Reference life
boundaries still end windows; paddle x, width and charge remain supplied.

## Paddle width added to the state container

`ball_paddle_width` wraps `state`, a `ball_bricks_contact_count` specification,
and `paddle_width`, a `paddle_width` specification. Output 114 is next width,
giving 115 outputs. All branches read the original 118-value source. The
established state branch remains frozen/eval; only the 33→64→64→2 ReLU width
classifier is trained at this stage, with 6,466 parameters. It uses combined y,
vertical velocity, current width and their scalar/binary encoding, including a
constant-velocity proposal feature. It predicts width classes 12 and 16 without
executing a native collision rule.

Use `evaluate_ball_bricks(..., contact=True, hit_count=True, paddle_width=True)`
with N×115 targets. Joint mode feeds output 114 into source column 5 along with
ball, brick, contact and count predictions. Other modes retain reference width.
Width feedback requires the contact/count output contract. Report width errors
separately, including real change events in teacher-forced evaluation. Paddle x,
charge and reference life boundaries remain supplied.

## Paddle charge and action added to the state container

`ball_paddle_charge` wraps `state`, a `ball_paddle_width` specification, and
`charge`, a `paddle_transition_mlp` specification restricted to the `charge`
controller field. The input is 119 values: the existing 118-value state followed
by the current provider action, an integer 0, 1 or 2. The state branch receives
its original 118 columns and remains frozen/eval. The charge adapter takes only
current paddle x, charge and action; unused legacy controller slots are zero.
The added 25→128→128→11 SiLU classifier has 21,259 trainable parameters. It
classifies a charge delta and adds it to current charge without clipping.

Output 115 is next charge, giving 116 simultaneous outputs. Use
`evaluate_ball_bricks(..., contact=True, hit_count=True, paddle_width=True,
paddle_charge=True)` with N×119 sources and N×116 targets. Joint mode feeds
charge output 115 into source column 6 alongside the other predictions. Action
column 118 comes from the current recorded transition on every step and is
never overwritten by feedback. Report charge error, MAE and out-of-range
predictions separately. Paddle position and reference life boundaries are still
supplied. The model executes no native controller update rule at inference.

## Paddle position added to the state container

`ball_paddle_position` wraps `state`, a `ball_paddle_charge` specification, and
`position`, a `paddle_transition_mlp` specification restricted to the `charge`
controller field. The source stays at 119 values: 118 current state values and
the current provider action. Output 116 is next paddle x, giving 117 outputs.
The position branch uses current paddle x, charge and action, with the same
adapter as the charge branch. It does not consume predicted next charge.

The established state branch stays frozen/eval. Train only the added
25→128→128→23 SiLU classifier (22,807 parameters) using cross-entropy over integer
displacements −11 through 11. Decode the winning displacement and add it to
current paddle x. Neither a native controller rule nor coordinate clipping is
used at inference. All next-state values are applied together.

Use `evaluate_ball_bricks(..., contact=True, hit_count=True, paddle_width=True,
paddle_charge=True, paddle_position=True)` with N×119 sources and N×117 targets.
Joint mode feeds paddle position output 116 into source column 4 alongside the
other predictions, including charge. The current action still comes from its
recorded transition. Report paddle-position exact errors, MAE and maximum error
separately. This enables feedback of every compact state field, but reference
life boundaries still stop windows and terminal transitions remain excluded.
Termination and RGB playback integration are separate work.

## Life-loss termination in the compact-state container

`ball_life_termination` wraps `state`, a `ball_paddle_position` specification,
and `termination`, a `life_termination` specification. It accepts the same 119
current-state/action values. Output columns 0:117 contain next state only when
output 117 (the stop flag) is zero. On a predicted stop they are zero placeholders
and must be ignored. The stop head runs first; the state predictor is called only
for continuing rows. No respawn state is generated or fed back.

The 31→64→64→2 ReLU stop head has 6,338 parameters and uses current RAM ball y,
fractional y and vy, with scalar/binary encoding and a constant-velocity proposal.
It learns the decision with two-class cross-entropy. The established
state branch stays frozen/eval. Include life-loss transitions in the RAM training
view and supervise their stop labels; mask their entire successor-state target.
Native replay may audit source phase and terminal labels offline, but production
inference uses only the neural stop head.

`predict(source, terminated=previous_stop_mask)` accepts an optional boolean mask.
Previously stopped rows remain stopped and skip both networks, even if their
unused source fields are invalid. Carry that mask forward, or remove stopped rows
from the caller's active batch. Resetting or starting another life is an explicit
caller operation. `forward` exposes named state logits and `terminated` logits;
training code must mask state losses on recorded terminal transitions.

`evaluate_life_rollouts` feeds all compact state fields back under recorded
current actions and halts each rollout on its first predicted stop. Bounded
windows start at every source. Include windows that reach a recorded death before
the requested horizon; exclude shorter windows ending in a data gap/truncation.
Full-segment mode starts once per recorded segment and includes censored segments.
Report exact death timing, premature stops, missed deaths and false stops in
survival/censored windows separately, alongside whole-trajectory state accuracy.
If a model misses a recorded death, count the miss and censor evaluation there;
its eventual late stopping time is unknown, and the next life's states/actions
must never be used. These termination-aware scores have a different denominator
from earlier fixed-length, nonterminal-only rollout scores.

The RGB player and shared experiment runner remain unchanged. This is a portable
compact-state model with a stopping contract and a dedicated rollout evaluator.

## Unified compact-state MLP

`unified_state_mlp` is a separate model-registry entry with the same N-by-119
input and N-by-118 output contract as the life-loss container. Every hidden layer
is shared across tasks. There are no regional routers, pretrained components or
learned feature dependencies. A single final linear layer emits eleven groups of
class logits: ball dx/dy/vx/vy, brick event, contact, count increment, width,
charge change, paddle displacement and termination.

The default network encodes the current inputs into 188 scalar/binary features,
projects to width 512, applies six residual blocks and a final LayerNorm, then
emits the logits. Each block is LayerNorm, Linear, SiLU, Linear plus its residual
connection. Encoding adds no history or hidden state. Actions are one-hot encoded.
The first ten numeric state values supply scaled scalars and binary features;
brick occupancy supplies 108 values. Both integer and fractional y are retained.

`fit_vocabulary` in `gymemu/unified_state_training.py` derives six displacement
and velocity class sets from training episodes only. Save those sets in
`model_spec`; reject unsupported labels rather than rounding them into another
class. Fixed event heads use 109 brick outcomes, two contact values, two count
increments, two widths and two stopping values. The one-removal brick contract
and capped count match the established container. Decode predicted events and
movement classes without calling native game rules. Absent brick cells are
masked during decoding; wall refill and respawn remain unsupported.

Use `encode_targets` and `classification_losses` for supervised training. Average
each state loss only over nonterminal rows; train stopping on every row. Terminal
successor labels may be NaN. `joint_loss` weights the eleven task means equally.
The model supports `forward_encoded` so fixed inputs can be encoded in RAM once.
Do not persist a new dataset or feature cache without user authorization.

`predict(source, terminated=mask)` skips previously stopped rows and keeps them
stopped. Active rows share one network execution for state and stopping; ignore
state outputs when stop is true. Reuse `evaluate_life_rollouts` for direct
comparison with the container. Inference requires no parent checkpoints.
This registry entry does not change the RGB approaches, runner or player.

## Recorded-state neural renderer

`state_renderer` is a registered model for an isolated same-state reconstruction
experiment. It is not wired into the shared RGB runner or player. Inputs have 112
physical-unit fields: ball x, integer RAM ball y, paddle x, paddle width, then
108 row-major brick occupancy values. Pixel coordinates supply the other input.
`from_dynamics` selects these fields from nonterminal unified outputs; callers
must stop terminal rows before rendering.

The encoder builds 20 spatial features, including clipped distances to objects
and fixed game boundaries, local brick occupancy, and coordinates within a brick.
A shared 20→64→64→64→9 SiLU MLP predicts palette classes independently at each
pixel. The palette is fitted to training images only. This explicit geometry is
an inductive bias, not a learned image encoder or a simulator renderer. The
network learns pixel colors and visibility from dataset targets. `render` returns
float32 RGB NCHW and blacks out HUD rows 0–16, whose state is unsupported.

Save `model_spec` (including palette) and `model` state dict; reload through
`build_model` with `torch.load(..., weights_only=True)`. Reconstruction training
and metrics are separate from dynamics training and next-frame RGB comparisons.
See [the experiment protocol](training.md#recorded-state-decoder).

### State dynamics with a separate renderer

`gymemu.state_playback.StatePlayback` implements the browser player protocol for
the published unified state model plus `state_renderer`. Use `gymemu play-state`
and `configs/state_playback.yaml`; this pipeline is separate from the RGB training
approaches. Model construction still goes through `build_model`. The orchestrator
maps the 118-value dynamics output into the next 119-value source and uses only
visual fields for rendering. Terminal placeholders are neither rendered nor fed
back. Reset clones the complete initial source.

Players can expose `playback_fps`, `finished`, and `history_editable` capabilities
to the shared browser transport. Defaults preserve existing RGB playback. Keep
model-specific field conversion in the orchestrator rather than adding branches
for individual approaches to the shared player or runner.

`gymemu.state_replay.StateReplay` implements the existing `ReplayPlayer` contract
for independent one-step state predictions. Its loader selects one episode from
the pinned split, checks transition and frame-ID continuity, reconstructs hidden
source fields causally, and loads referenced RGB assets by ID. No controller or
collision reconstruction runs inside the model's generated rollout. Replay keeps
termination agreement separate from state and RGB comparisons and does not stop
on a predicted terminal. A generic `context_note` lets players explain their
input semantics and per-step comparisons in Playback settings.
