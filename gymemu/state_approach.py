"""State-only objectives and generated-state rollout, independent of the RGB runner."""

import torch
from torch import nn
from torch.nn import functional as F

from gymemu.models import build_model
from gymemu.models.state_dynamics import predicted_state
from gymemu.state_data import NATIVE_SCALES


class StateApproach(nn.Module):
    def __init__(self, model_spec, loss_config, terminal_positive_weight=1.0):
        super().__init__()
        self.predictor = build_model(model_spec)
        self.loss_config = dict(loss_config)
        self.register_buffer(
            "motion_scale",
            torch.tensor(NATIVE_SCALES / loss_config["motion_tolerances"], dtype=torch.float32),
        )
        self.register_buffer(
            "terminal_positive_weight", torch.tensor(float(terminal_positive_weight))
        )

    def rollout(self, batch):
        history, hvalid, ah = batch["history"], batch["history_valid"], batch["action_history"]
        hidden = None
        outputs, states = [], []
        for step in range(batch["actions"].shape[1]):
            if step:
                ah = torch.cat((ah[:, 1:], batch["actions"][:, step, None]), 1)
            output, hidden = self.predictor(history, hvalid, ah, hidden)
            state = predicted_state(output)
            outputs.append(output)
            states.append(state)
            history = torch.cat((history[:, 1:], state[:, None]), 1)
            hvalid = torch.cat((hvalid[:, 1:], torch.ones_like(hvalid[:, :1])), 1)
        return {k: torch.stack([o[k] for o in outputs], 1) for k in outputs[0]}, torch.stack(
            states, 1
        )

    def loss(self, batch):
        output, _ = self.rollout(batch)
        target, valid = batch["target"], batch["valid"].float()
        live = valid * (~batch["terminal"]).float()
        count = 6 if self.predictor.predict_paddle_velocity else 5
        motion = (
            ((output["motion"][..., :count] - target[..., :count]) * self.motion_scale[:count])
            .square()
            .mean(-1)
        )
        width = F.binary_cross_entropy_with_logits(
            output["width"], (target[..., 6] < 0.875).float(), reduction="none"
        )
        # Change weighting supplements, rather than replaces, all-cell supervision.
        previous = torch.cat((batch["history"][:, -1:, 7:], target[:, :-1, 7:]), 1)
        changes = (target[..., 7:] != previous).float()
        weights = 1 + changes * self.loss_config["brick_change_weight"]
        bricks = (
            F.binary_cross_entropy_with_logits(output["bricks"], target[..., 7:], reduction="none")
            * weights
        ).mean(-1)
        terminal = F.binary_cross_entropy_with_logits(
            output["terminal"],
            batch["terminal"].float(),
            reduction="none",
            pos_weight=self.terminal_positive_weight,
        )
        losses = {}
        for name, values, mask in (
            ("motion", motion, live),
            ("width", width, live),
            ("bricks", bricks, live),
            ("terminal", terminal, valid),
        ):
            losses[name] = (values * mask).sum() / mask.sum().clamp_min(1)
        total = sum(losses[name] * self.loss_config[name] for name in losses)
        return total, losses


class StatePlayer:
    """Headless state playback; generated state only, stopping on terminal output."""

    def __init__(self, approach, history, history_valid, past_actions, threshold=0.5):
        self.model = approach.eval()
        self.history, self.history_valid = history.clone(), history_valid.clone()
        self.past_actions = past_actions.clone()
        self.threshold = threshold
        self.hidden = None
        self.terminal = False
        self.steps = 0

    @torch.no_grad()
    def step(self, action):
        if self.terminal:
            raise RuntimeError("Simulation is terminal")
        if type(action) is not int or action not in (0, 1, 2):
            raise ValueError("Expected requested FIRE/RIGHT/LEFT action 0/1/2")
        current = torch.full((1, 1), action, device=self.history.device, dtype=torch.long)
        actions = torch.cat((self.past_actions, current), 1)
        output, self.hidden = self.model.predictor(
            self.history, self.history_valid, actions, self.hidden
        )
        probability = float(output["terminal"].sigmoid().item())
        self.terminal = probability >= self.threshold
        self.steps += 1
        state = predicted_state(output)
        self.past_actions = actions[:, 1:]
        self.history = torch.cat((self.history[:, 1:], state[:, None]), 1)
        self.history_valid = torch.cat(
            (self.history_valid[:, 1:], torch.ones_like(self.history_valid[:, :1])), 1
        )
        return {
            "terminal": self.terminal,
            "terminal_probability": probability,
            "state": None if self.terminal else state[0].cpu().tolist(),
        }
