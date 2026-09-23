"""Diagnostics catch collapse without changing optimization or episode alignment."""

import json

import pytest
import torch
from torch import nn

from gymemu.checkpoints import load_model
from gymemu.config import compose_config
from gymemu.data import Frames, Windows, read_episodes
from gymemu.diagnostics import Health, RolloutProbe, observational
from gymemu.engine import train
from gymemu.metrics import validate_metrics


def test_dead_weights_are_detected_even_when_output_bias_has_gradients():
    model = nn.Sequential(nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 1), nn.Sigmoid())
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.fill_(-1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    health = Health(model)
    with health.capture(True):
        loss = model(torch.ones(2, 1)).square().mean()
        loss.backward()
        health.backward(loss, 2)
        optimizer.step()
        health.updated()
    metrics = health.metrics()
    validate_metrics(metrics)
    assert metrics["train/grad/norm/max"] > 0
    assert metrics["train/grad/weights/zero/fraction"] == 1
    assert metrics["train/grad/0.weight/nonzero/fraction"] == 0
    assert metrics["train/grad/2.bias/nonzero/fraction"] == 1
    assert metrics["train/update/0.weight/ratio"] == 0
    assert metrics["train/update/2.bias/ratio"] > 0
    assert not health.handles


def test_window_preserves_single_update_spike():
    model = nn.Linear(1, 1, bias=False)
    health = Health(model)
    for magnitude in (1.0, 100.0, 1.0):
        model.weight.grad = torch.full_like(model.weight, magnitude)
        health.begin(False)
        health.backward(torch.tensor(magnitude), 1)
        health.updated()
    result = health.metrics()
    assert result["train/grad/peak/step"] == 2
    assert result["train/grad/norm/max"] == 100
    assert result["train/grad/norm/mean"] == 34
    assert result["train/loss/max"] == 100
    assert not health.norms


def test_detailed_statistics_remain_tensors_until_report():
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1), nn.Sigmoid())
    health = Health(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with health.capture(True):
        loss = model(torch.ones(3, 2)).sum()
        loss.backward()
        health.backward(loss, 3)
        optimizer.step()
        health.updated()
    # Collection must not synchronize the GPU separately for each parameter statistic.
    assert isinstance(health.details["train/grad/0.weight/norm"], torch.Tensor)
    assert isinstance(health.details["train/update/0.weight/ratio"], torch.Tensor)
    assert isinstance(health.details["train/act/3/sat/fraction"], torch.Tensor)
    result = health.metrics()
    assert isinstance(result["train/grad/0.weight/norm"], float)
    assert isinstance(result["train/update/0.weight/ratio"], float)
    validate_metrics(result)


def test_detailed_batches_use_eager_loss_outside_compiled_graph():
    from types import SimpleNamespace

    from gymemu.approaches import build_approach
    from gymemu.engine import batch_loss, run_epoch

    spec = {"kind": "direct", "models": {"predictor": {"kind": "direct_cnn", "width": 2}}}
    model = build_approach(spec, 1, 2, (3, 8, 8))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    health = Health(model)
    calls = []
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(batch_loss, backend=backend, fullgraph=True)

    def regular(*args):
        assert not health.handles
        calls.append("regular")
        return compiled(*args)

    def detailed(*args):
        assert health.handles
        calls.append("detailed")
        return batch_loss(*args)

    class Loader:
        dataset = SimpleNamespace(frames=SimpleNamespace(compact=True))

        def __len__(self):
            return 24

        def __iter__(self):
            for _ in range(24):
                yield (
                    torch.zeros(2, 1, 3, 8, 8, dtype=torch.uint8),
                    torch.zeros(2, dtype=torch.long),
                    torch.zeros(2, 3, 8, 8, dtype=torch.uint8),
                )

    run_epoch(
        model,
        Loader(),
        torch.device("cpu"),
        optimizer,
        health=health,
        loss_function=regular,
        diagnostic_loss_function=detailed,
        log_every=12,
        report=lambda *args: None,
    )
    assert calls == ["detailed"] * 10 + ["regular", "detailed"] + ["regular"] * 11 + ["detailed"]
    assert len(graphs) == 1


def test_saturation_and_hook_cleanup():
    model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
    with torch.no_grad():
        model[0].weight.fill_(1e8)
    health = Health(model)
    with health.capture(True):
        loss = model(torch.ones(1, 1)).sum()
        loss.backward()
        health.backward(loss, 1)
        health.updated()
    metrics = health.metrics()
    assert metrics["train/act/1/sat/fraction"] == 1
    assert metrics["train/act/1/abs/max"] > 1e7
    assert metrics["train/grad/weights/zero/fraction"] == 1
    with pytest.raises(RuntimeError), health.capture(True):
        raise RuntimeError("forward failed")
    assert not health.handles


def test_observation_preserves_buffers_modes_rng_and_gradients():
    model = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2))
    model[0].eval()
    model[1].weight.grad = torch.ones_like(model[1].weight)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state()
    with observational(model):
        assert not model.training
        torch.rand(9)
        model[0].running_mean.fill_(3)
    assert model.training and not model[0].training
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    assert model[1].weight.grad.eq(1).all()


class Drift(nn.Module):
    def forward(self, history, action):
        return history[:, -1] + 0.01


def test_probe_masks_episode_tails_and_uses_predicted_feedback(snapshot, tmp_path):
    dataset = Windows(Frames(snapshot, compact=True), read_episodes(snapshot, "heldout"), 2, [0, 2])
    probe = RolloutProbe(dataset, samples=3, horizon=8)
    manifest = probe.manifest()
    assert [s["frame_ids"] for s in manifest["starts"]] == [[40, 50, 60], [50, 60]]
    # Probe selection may deduplicate tiny datasets, and never borrows the next episode.
    metrics = probe.run(Drift(), torch.device("cpu"), tmp_path / "probe.png")
    validate_metrics(metrics)
    assert metrics["probe/count/h1"] == 2
    assert metrics["probe/count/h3"] == 1
    assert metrics["probe/count/h4"] == 0
    assert "probe/mse/h4" not in metrics
    assert metrics["probe/drift/ratio"] > 1
    assert metrics["probe/ball/count"] == 0
    assert "probe/ball/mse" not in metrics
    assert (tmp_path / "probe.png").exists()


def test_goal_probe_uses_exact_episode_offset_and_frame_ids(snapshot):
    dataset = Windows(Frames(snapshot, compact=True), read_episodes(snapshot, "heldout"), 2, [0, 2])
    probe = RolloutProbe(
        dataset,
        horizon=2,
        starts=[{"episode": 2, "offset": 1, "frame_ids": [50, 60]}],
    )
    assert probe.manifest()["starts"] == [{"episode": 2, "offset": 1, "frame_ids": [50, 60]}]
    with pytest.raises(ValueError, match="frame IDs"):
        RolloutProbe(
            dataset,
            horizon=2,
            starts=[{"episode": 2, "offset": 1, "frame_ids": [50, 99]}],
        )


@pytest.mark.parametrize("approach", ["direct", "latent", "autoregressive_ball_region"])
def test_diagnostics_preserve_training_weights_and_log_local_evidence(snapshot, tmp_path, approach):
    cfg = compose_config(
        [
            "game=custom",
            f"approach={approach}",
            "experiment=smoke",
            "wandb.mode=disabled",
            "r2.enabled=false",
        ]
    )
    cfg.game.dataset = str(snapshot)
    cfg.trainer.diagnostics.horizon = 4
    if approach == "autoregressive_ball_region":
        cfg.approach.options.sprite_height = 2
        cfg.approach.options.sprite_width = 1
    models = []
    for enabled in (False, True):
        cfg.trainer.diagnostics.enabled = enabled
        cfg.output = str(tmp_path / str(enabled))
        output = train(cfg)
        models.append(load_model(output / "best.pt", "cpu")[0])
    for name, value in models[0].state_dict().items():
        assert torch.equal(value, models[1].state_dict()[name]), name
    events = [json.loads(line) for line in (output / "diagnostics.jsonl").read_text().splitlines()]
    assert events[0]["train/step"] == 0
    assert any("train/grad/norm/max" in event for event in events)
    assert any("probe/mse/h1" in event for event in events)
    assert (output / "probe.json").exists()
    assert list((output / "diagnostics").glob("*.png"))
    for event in events:
        validate_metrics(event)
