# Gymemu

Gymemu compares dataset-trained neural emulator approaches. Hydra composes game,
model, approach, trainer, and experiment configs. `gymemu/data.py` owns trajectory
alignment; approaches own model composition and training objectives; the shared
runner owns stages, evaluation, and run artifacts. Root scripts are entry points.

- Work on the current branch unless the user explicitly requests another.
- Keep `approach=direct` as the unchanged reference model: RGB history and executed
  action, convolutional encoder/decoder, uniform next-frame RGB MSE. Add alternative
  models or pipelines as explicit approaches/configs, not silent baseline changes.
- An approach may contain multiple models and ordered training stages. Keep model
  construction in the model registry and stage behavior in the approach. Do not add
  approach-specific branches to the runner or player. Frozen models must stay frozen
  and in evaluation mode during later stages.
- Never cross episode boundaries, fit any stage on held-out trajectories, or treat
  image IDs as array offsets. Compare approaches using the same held-out targets and
  float32 next-frame RGB MSE. Keep stage losses separate from comparison metrics.
- A bootstrap example has no history and no game action. It predicts an episode's
  initial frame. Training and playback share oldest-to-newest, left-zero-padding.
- Keep game IDs, revisions, keyboard bindings, and starting-scene selection in game
  configs. The built-in dataset contract is fixed-size RGB and scalar integer actions;
  other schemas/action types need an explicit adapter. Do not claim arbitrary dataset
  formats or action spaces are already supported.
- The player opens paused in teacher-forcing mode using recorded dataset history and
  actions. --autoregressive selects interactive play, which starts in single-step
  mode with inference on fresh action key presses.
  Tab toggles continuous play at 30 predictions/s for 60 Hz Atari with frameskip 2.
  Reset and focus loss pause playback. Autoregressive play loads
  start-scene.npz beside the checkpoint and restores it on reset. Breakout's config
  selects its existing full-wall scene after startup animation. --start-scene selects
  another scene; --empty-start explicitly tests learned initialization. Never silently
  fall back to empty history when a scene is missing. Predicted frames supply subsequent
  history, with no periodic correction from recorded data.
- Save resolved configs, dataset provenance, evaluation identity, and every model needed
  for inference. Keep version-1 direct-CNN checkpoint loading supported. Do not execute
  arbitrary Python targets from checkpoint metadata.
- Keep README.md and docs/training.md current. Put approach extension instructions in
  docs/approaches.md and research evidence in docs/history.md. Do not claim a finite
  frame history is fully observable or low pixel MSE guarantees playable rollouts.
- Use uv sync --frozen, maintain uv.lock, and preserve the seven-day package age gate
  and bad-package constraints. Do not add alternate indexes.
- Keep generated datasets, runs, checkpoints, and diagnostics ignored. Never commit credentials.
- Run uv run pytest, uv run ruff check ., and a bounded train/play smoke for changes
  affecting alignment, startup, checkpoint loading, or player stepping. Exercise both
  direct and multi-stage paths for changes to shared experiment infrastructure.
