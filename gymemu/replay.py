"""Teacher-forced episode playback using the training dataset's aligned windows."""

import torch

from gymemu.data import Frames, Windows, frame_stack, read_episodes, resolve_dataset

COMPARISON_LABELS = ("Prediction", "Original", "Diff")


def load_replay(model, config, device, *, dataset=None, revision=None, split=None, episode_id=None):
    provenance = config.get("dataset", {})
    if isinstance(provenance, str):  # Older checkpoint metadata.
        provenance = {"dataset": provenance}
    source = dataset or provenance.get("dataset") or config.get("game", {}).get("dataset")
    if not source:
        raise ValueError("Checkpoint has no dataset provenance; supply --dataset")
    if dataset is None or dataset == provenance.get("dataset"):
        revision = revision or provenance.get("revision")
    split = split or config.get("game", {}).get("eval_split", config.get("eval_split", "heldout"))
    print(f"Loading replay dataset {source}, split={split}...", flush=True)
    root, resolved = resolve_dataset(source, revision)
    frames = Frames(root)
    if tuple(frames.shape) != tuple(config["shape"]):
        raise ValueError("Dataset frame dimensions differ from the checkpoint")
    episodes = read_episodes(root, split, state_fields=config.get("state_fields", ()))
    player = ReplayPlayer(model, config, device, frames, episodes, episode_id=episode_id)
    print(f"Teacher forcing: {resolved}, split={split}, episodes={len(episodes)}", flush=True)
    return player


class ReplayPlayer:
    """Predict successors from recorded RGB, actions, and optional state on every step."""

    def __init__(self, model, config, device, frames, episodes, *, episode_id=None):
        self.model, self.config, self.device = model, config, device
        self.actions = config["action_values"]
        self.windows = Windows(
            frames,
            episodes,
            config["history"],
            self.actions,
            action_history=config.get("action_history", 1),
            state_fields=config.get("state_fields", ()),
        )
        self.episode_index = 0
        if episode_id is not None:
            matches = [i for i, e in enumerate(episodes) if e.episode_id == episode_id]
            if not matches:
                raise ValueError(f"Episode ID {episode_id} is absent from the selected split")
            self.episode_index = matches[0]
        self.reset()

    @property
    def episode(self):
        return self.windows.episodes[self.episode_index]

    @property
    def start_name(self):
        return f"teacher forcing | episode {self.episode.episode_id}"

    @property
    def finished(self):
        return self.steps == len(self.episode.actions)

    def reset(self, *, cycle=False):
        if cycle:
            self.episode_index = (self.episode_index + 1) % len(self.windows.episodes)
        self.continuous = False
        self.held_keys = []
        self.steps = 0
        self.frame = None
        self.has_prediction = False
        self.mse = None
        self.recorded_action = None
        self.target = self.windows.frames.get(int(self.episode.frames[0]))
        self.input_stack = frame_stack([self.target], self.config["history"], self.config["shape"])

    @torch.inference_mode()
    def advance(self, action=None):
        # Keyboard actions never replace the dataset's executed action.
        if self.finished:
            self.continuous = False
            return
        offset = int(self.windows.ends[self.episode_index - 1]) if self.episode_index else 0
        history, token, target, *states = self.windows[offset + self.steps + 1]
        inputs = history.unsqueeze(0).to(self.device)
        tokens = torch.as_tensor(token, dtype=torch.long, device=self.device).unsqueeze(0)
        self.input_tokens = tokens
        self.input_states = states[0].unsqueeze(0).to(self.device) if states else None
        if states:
            prediction, _ = self.model.predict_step(
                inputs, tokens, self.input_states
            )
        else:
            prediction = self.model(inputs, tokens)
        self.frame = prediction[0].float().cpu()
        self.target = target
        self.input_stack = history
        self.has_prediction = True
        self.recorded_action = int(self.episode.actions[self.steps])
        self.steps += 1
        self.mse = (self.frame - self.target.float()).square().mean().item()
        if self.finished:
            self.continuous = False

    def tick(self, keymap):
        if self.continuous:
            self.advance()

    def pixels(self):
        prediction = torch.zeros_like(self.target) if self.frame is None else self.frame
        # Preserve the sign in RGB: [-1, 1] maps to [0, 1], with zero at neutral gray.
        difference = (
            torch.full_like(self.target, 0.5)
            if self.frame is None
            else 0.5 + 0.5 * (self.frame.float() - self.target.float())
        )
        return (
            torch.cat([prediction, self.target, difference], dim=2)
            .permute(1, 2, 0)
            .mul(255)
            .round()
            .clamp(0, 255)
            .byte()
            .numpy()
        )

    def status(self):
        mode = "End of episode" if self.finished else ("Playing" if self.continuous else "Paused")
        return f"{mode} | Frame {self.steps}/{len(self.episode.actions)} | Space step | Tab play"

    def comparison_label(self):
        if self.mse is None:
            return "Initial recorded frame | prediction pending"
        return f"Action {self.recorded_action} | RGB MSE {self.mse:.6f}"
