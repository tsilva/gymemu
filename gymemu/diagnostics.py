"""Read-only training health measurements and fixed held-out recursive probes."""

from contextlib import contextmanager

import numpy as np
import torch
from PIL import Image

from gymemu.ball_regions import ball_region_mask


@contextmanager
def observational(model):
    """Preserve module modes, buffers and RNG; probes never touch gradients or losses."""
    modes = [(module, module.training) for module in model.modules()]
    buffers = [(buffer, buffer.clone()) for buffer in model.buffers()]
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
    with torch.random.fork_rng(devices=devices):
        try:
            model.eval()
            with torch.no_grad():
                yield
        finally:
            for buffer, saved in buffers:
                buffer.copy_(saved)
            for module, training in modes:
                module.training = training


class Health:
    """Global norms on every update; detailed layer statistics on sampled updates.

    No clipping, backward hooks, or changes to optimizer behavior. The window maximum
    retains a single-update spike that would disappear in an epoch average.
    """

    def __init__(self, model, start_step=0):
        self.model = model
        self.step = start_step
        self.reset()
        self.handles = []

    def reset(self):
        self.losses, self.norms, self.weight_norms = [], [], []
        self.raw_losses = []
        self.samples = 0
        self.details = {}
        self.activations = {}

    def state_dict(self):
        return {
            name: getattr(self, name)
            for name in (
                "step",
                "losses",
                "norms",
                "weight_norms",
                "raw_losses",
                "samples",
                "details",
            )
        }

    def load_state_dict(self, state, device):
        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, list):
                return [move(item) for item in value]
            if isinstance(value, dict):
                return {key: move(item) for key, item in value.items()}
            return value

        for name, value in state.items():
            setattr(self, name, move(value))

    def begin(self, detailed):
        self.detailed = detailed
        self.activations = {}
        if detailed:
            for name, module in self.model.named_modules():
                if isinstance(module, (torch.nn.Sigmoid, torch.nn.ReLU)):
                    self.handles.append(module.register_forward_hook(self._hook(name)))

    @contextmanager
    def capture(self, detailed):
        self.begin(detailed)
        try:
            yield
        finally:
            self.close()

    def _hook(self, name):
        def measure(module, args, output):
            with torch.no_grad():
                values = args[0].detach().float()
                if isinstance(module, torch.nn.Sigmoid):
                    # Actual rounded output, so bf16 saturation is observable too.
                    dead = (output == 0) | (output == 1)
                    root = f"train/act/{name}"
                    stats = torch.stack(
                        (
                            values.abs().max(),
                            dead.float().mean(),
                            (~torch.isfinite(values)).float().mean(),
                        )
                    )
                else:
                    root = f"train/relu/{name}"
                    stats = torch.stack(
                        (
                            output.detach().float().abs().max(),
                            (output == 0).float().mean(),
                            (~torch.isfinite(output)).float().mean(),
                        )
                    )
                self.activations.setdefault(root, []).append(stats)

        return measure

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    @torch.no_grad()
    def backward(self, loss, samples):
        self.close()
        self.step += 1
        parameters = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        parameter_norms = {
            name: p.grad.detach().float().norm() for name, p in parameters if p.grad is not None
        }
        norms = list(parameter_norms.values())
        device = loss.device
        norm = torch.stack(norms).norm() if norms else torch.zeros((), device=device)
        weights = [
            parameter_norms[name] for name, p in parameters if p.ndim > 1 and p.grad is not None
        ]
        self.norms.append(norm)
        self.weight_norms.append(torch.stack(weights).norm() if weights else norm.new_zeros(()))
        self.losses.append(loss.detach().float() * samples)
        self.raw_losses.append(loss.detach().float())
        self.samples += samples
        self.before = {}
        if not self.detailed:
            return
        for name, p in parameters:
            root = f"train/grad/{name}"
            grad = p.grad
            self.details[f"{root}/missing"] = float(grad is None)
            if grad is not None:
                grad = grad.detach().float()
                self.details[f"{root}/norm"] = parameter_norms[name]
                self.details[f"{root}/nonzero/fraction"] = grad.ne(0).float().mean()
                self.details[f"{root}/nonfinite/fraction"] = (~grad.isfinite()).float().mean()
            self.before[name] = p.detach().clone()
        for root, stats in self.activations.items():
            # Max over calls exposes the worst recursive step in this update.
            maximum, fraction, nonfinite = torch.stack(stats).amax(0).unbind()
            self.details[f"{root}/abs/max"] = maximum
            measure = "sat" if root.startswith("train/act/") else "zero"
            self.details[f"{root}/{measure}/fraction"] = fraction
            self.details[f"{root}/nonfinite/fraction"] = nonfinite

    @torch.no_grad()
    def updated(self):
        for name, p in self.model.named_parameters():
            if name in self.before:
                before = self.before[name].float()
                self.details[f"train/update/{name}/ratio"] = (
                    p.detach().float() - before
                ).norm() / before.norm().clamp_min(1e-12)
                self.details[f"train/weight/{name}/norm"] = p.detach().float().norm()
        self.before.clear()

    def metrics(self):
        norms, weights = torch.stack(self.norms), torch.stack(self.weight_norms)
        metrics = {
            "train/loss/mean": torch.stack(self.losses).sum() / self.samples,
            "train/loss/max": torch.stack(self.raw_losses).max(),
            "train/grad/norm/max": norms.max(),
            "train/grad/norm/mean": norms.mean(),
            "train/grad/norm/nonfinite/fraction": (~norms.isfinite()).float().mean(),
            "train/grad/weights/norm/min": weights.min(),
            "train/grad/weights/zero/fraction": weights.eq(0).float().mean(),
            **self.details,
        }
        tensors = {key: value for key, value in metrics.items() if isinstance(value, torch.Tensor)}
        # One host transfer for scalar measurements, instead of one per layer/statistic.
        values = torch.stack([*tensors.values(), norms.argmax().to(norms.dtype)]).tolist()
        metrics.update(zip(tensors, values[:-1], strict=True))
        metrics["train/grad/peak/step"] = self.step - len(self.norms) + int(values[-1]) + 1
        self.reset()
        return metrics


class RolloutProbe:
    """Fixed episode-local starts, recorded actions, and predicted RGB/state feedback."""

    def __init__(self, dataset, *, samples=8, horizon=32, sprite=None, starts=None):
        self.sprite, self.horizon = sprite, horizon
        self.starts = []
        # Include bootstrap/early history, then spread starts across held-out windows.
        if starts is None:
            indices = list(
                dict.fromkeys(
                    [0, min(1, len(dataset) - 1)]
                    + np.linspace(0, len(dataset) - 1, max(1, samples - 2), dtype=int).tolist()
                )
            )[:samples]
        else:
            indices = []
            for selected in starts:
                matches = [
                    (number, episode)
                    for number, episode in enumerate(dataset.episodes)
                    if int(episode.episode_id) == selected["episode"]
                ]
                if len(matches) != 1:
                    raise ValueError("Goal probe episode is absent or ambiguous")
                number, episode = matches[0]
                offset = selected["offset"]
                if not 0 <= offset < len(episode.frames):
                    raise ValueError("Goal probe offset is outside its episode")
                actual = episode.frames[offset : offset + horizon].tolist()
                if selected.get("frame_ids") and selected["frame_ids"] != actual:
                    raise ValueError("Goal probe frame IDs differ from the pinned dataset")
                indices.append((int(dataset.ends[number - 1]) if number else 0) + offset)
        for index in indices:
            number = int(np.searchsorted(dataset.ends, index, side="right"))
            position = index - (int(dataset.ends[number - 1]) if number else 0)
            episode = dataset.episodes[number]
            length = min(horizon, len(episode.frames) - position)
            initial = dataset[index]
            frames = [dataset[index + step] for step in range(length)]
            scale = 255 if dataset.frames.compact else 1
            self.starts.append(
                {
                    "episode": int(episode.episode_id),
                    "offset": int(position),
                    "frame_ids": episode.frames[position : position + length].tolist(),
                    "history": initial[0].float() / scale,
                    "state": initial[3] if dataset.state_fields else None,
                    "actions": [torch.as_tensor(frame[1]) for frame in frames],
                    "targets": torch.stack([frame[2].float() / scale for frame in frames]),
                }
            )

    def manifest(self):
        return {
            "horizon": self.horizon,
            "precision": "fp32",
            "starts": [
                {key: start[key] for key in ("episode", "offset", "frame_ids")}
                for start in self.starts
            ],
        }

    def run(self, model, device, media_path=None):
        sums = {}
        counts = {}
        pictures = []

        def add(key, value, count=1):
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + count

        with observational(model):
            for start in self.starts:
                history = start["history"].unsqueeze(0).to(device)
                clean = history.clone()
                state = start["state"]
                state = state.unsqueeze(0).to(device) if state is not None else None
                targets = start["targets"].to(device)
                row = []
                previous_detection = None
                for step, action in enumerate(start["actions"]):
                    action = action.unsqueeze(0).to(device)
                    target = targets[step : step + 1]
                    if state is None:
                        prediction = model(history, action).float()
                    else:
                        prediction, next_state = model.predict_step(history, action, state)
                        prediction = prediction.float()
                        state = torch.cat((state[:, 1:], next_state[:, None]), 1)
                    error = (prediction - target).square().mean(1, keepdim=True)
                    mse = error.mean().item()
                    add(f"probe/mse/h{step + 1}", mse)
                    add("probe/mse/mean", mse)
                    foreground = target.ne(0).any(1, keepdim=True)
                    if foreground.any():
                        add("probe/fg/mse", (error * foreground).sum() / foreground.sum())
                    add(
                        "probe/early/mse"
                        if start["offset"] < history.shape[1]
                        else "probe/regular/mse",
                        mse,
                    )
                    add("probe/pixel/nonfinite/fraction", (~prediction.isfinite()).float().mean())
                    add(
                        "probe/pixel/saturated/fraction",
                        ((prediction <= 0) | (prediction >= 1)).float().mean(),
                    )
                    add(
                        "probe/pixel/wrong_sat/fraction",
                        (
                            ((prediction <= 0) | (prediction >= 1))
                            & ((prediction - target).abs() > 0.1)
                        )
                        .float()
                        .mean(),
                    )
                    add("probe/pixel/std", prediction.flatten(2).std(2, unbiased=False).mean())
                    if step > 0:
                        add(
                            "probe/motion/mse",
                            ((prediction - history[:, -1]) - (target - targets[step - 1 : step]))
                            .square()
                            .mean(),
                        )
                        add("probe/motion/pred", (prediction - history[:, -1]).abs().mean())
                        add("probe/motion/target", (target - targets[step - 1 : step]).abs().mean())
                    if self.sprite:
                        mask, detected = ball_region_mask(
                            target,
                            sprite_height=self.sprite["height"],
                            sprite_width=self.sprite["width"],
                            padding=4,
                        )
                        add("probe/ball/coverage", detected.float().mean())
                        if detected.item():
                            region = (error * mask).sum() / mask.sum()
                            add("probe/ball/mse", region)
                            add(f"probe/ball/mse/h{step + 1}", region)
                            if previous_detection is False:
                                add("probe/ball/reappear/mse", region)
                        previous_detection = bool(detected.item())
                    # Same targets and clean histories distinguish exposure drift from bad fits.
                    if state is None:
                        teacher = model(clean, action).float()
                        add("probe/teacher/mse", (teacher - target).square().mean())
                    if media_path is not None and len(pictures) < 4:
                        pair = torch.cat((target[0], prediction[0]), dim=1)
                        row.append(pair.detach().nan_to_num().clamp(0, 1).cpu())
                    history = torch.cat((history[:, 1:], prediction[:, None]), 1)
                    clean = torch.cat((clean[:, 1:], target[:, None]), 1)
                if row:
                    # Eight evenly spaced frames keep the saved comparison readable.
                    selected = np.linspace(0, len(row) - 1, min(8, len(row)), dtype=int)
                    pictures.append(torch.cat([row[i] for i in selected], dim=2))
        metrics = {key: value / counts[key] for key, value in sums.items()}
        metrics.update(
            {
                f"probe/count/h{step + 1}": counts.get(f"probe/mse/h{step + 1}", 0)
                for step in range(self.horizon)
            }
        )
        metrics["probe/ball/count"] = counts.get("probe/ball/mse", 0)
        metrics["probe/ball/reappear/count"] = counts.get("probe/ball/reappear/mse", 0)
        if "probe/teacher/mse" in metrics:
            metrics["probe/drift/ratio"] = metrics["probe/mse/mean"] / max(
                metrics["probe/teacher/mse"], 1e-12
            )
        if pictures:
            width = max(picture.shape[2] for picture in pictures)
            pictures = [
                torch.nn.functional.pad(picture, (0, width - picture.shape[2]))
                for picture in pictures
            ]
            pixels = (torch.cat(pictures, dim=1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            media_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(media_path)
        return metrics
