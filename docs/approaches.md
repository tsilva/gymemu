# Emulator approaches

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
| `approach=latent` | Frame codec and separate latent CNN | Reconstruct recorded frames, freeze the codec, then predict successor latents |

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

## Add a model

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
