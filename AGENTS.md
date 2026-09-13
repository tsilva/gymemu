# Gymemu

Gymemu trains neural emulators from existing trajectory datasets. The current baseline
is deliberately small: `train.py` owns the dataset adapter, CNN, checkpoint contract,
and training CLI; `play.py` imports those definitions and provides an interactive player.

- Work on the current branch unless the user explicitly requests another.
- Preserve the plain next-frame baseline: RGB frame stack plus the current executed
  action, convolutional encoder/decoder, and uniform pixel MSE. Architecture changes,
  auxiliary inputs, residual/warp prediction, and multi-step objectives require new intent.
- A bootstrap example has no history and no game action. It predicts an episode's
  initial frame. The player uses the same bootstrap and left-zero-padding convention.
- Never cross episode boundaries, fit on held-out trajectories, or treat image IDs as
  array offsets. Keep frame/action/target alignment shared between training and play.
- The player runs inference only on a fresh action key press. By default the first
  press after opening/reset generates the initial frame; later presses execute actions.
  An explicit recorded-scene start displays its chronological history's final frame
  without inference, executes an action on the first press, and restores that scene
  on reset. Generated frames supply all subsequent history.
- Keep README.md current. Put research evidence and the limits of a history-length
  recommendation in docs/history.md; do not claim a finite stack is fully observable.
- Use `uv sync --frozen`, maintain `uv.lock`, and preserve the seven-day package age
  gate and bad-package constraints in pyproject.toml. Do not add alternate indexes.
- Keep generated data, checkpoints, and diagnostics ignored. Never commit credentials.
- Run `uv run pytest`, `uv run ruff check .`, and a bounded train/play smoke for
  changes affecting alignment, startup, checkpoint loading, or player stepping.
