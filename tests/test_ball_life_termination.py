"""Terminal-state masking, absorbing stops and independent timing accounting."""

import io

import numpy as np
import pytest
import torch
from test_ball_paddle_position import position_container_spec
from test_ball_position import inputs
from torch import nn
from torch.nn import functional as F

from gymemu.life_rollout_eval import evaluate_life_rollouts
from gymemu.models import build_model


def container():
    state = position_container_spec()
    state.pop("kind")
    spec = dict(kind="ball_life_termination", state=state, termination=dict(width=16, depth=1))
    return spec, build_model(spec)


def test_stop_training_freezes_state_and_round_trips():
    spec, model = container()
    source = torch.cat([inputs(), torch.arange(3)[:, None]], 1)
    frozen = {n: v.clone() for n, v in model.state.state_dict().items()}
    model.train()
    assert not any(m.training for m in model.state.modules())
    F.cross_entropy(model(source)["terminated"], torch.tensor([0, 1, 0])).backward()
    assert all(p.grad is None for p in model.state.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() for p in model.termination.parameters())
    torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]).step()
    assert all(torch.equal(v, frozen[n]) for n, v in model.state.state_dict().items())
    with torch.no_grad():
        model.termination.network[-1].weight.zero_()
        model.termination.network[-1].bias.copy_(torch.tensor([10.0, 0.0]))
    assert torch.equal(model.predict(source)[:, :117], model.state.predict(source))
    assert not model.predict(source)[:, 117].any()
    with torch.no_grad():
        model.termination.network[-1].bias.copy_(torch.tensor([0.0, 10.0]))
    assert model.predict(source)[:, 117].all()
    assert not model.predict(source)[:, :117].any()
    buffer = io.BytesIO()
    torch.save(dict(model_spec=spec, model=model.state_dict()), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    replica = build_model(saved["model_spec"])
    replica.load_state_dict(saved["model"])
    assert torch.equal(replica.predict(source), model.predict(source))
    assert replica.predict(source[:0]).shape == (0, 118)


def test_stopped_rows_skip_both_networks_and_cannot_restart(monkeypatch):
    _, model = container()
    source = torch.cat([inputs(), torch.arange(3)[:, None]], 1)
    source[:, 0] = torch.tensor([0.0, 1.0, 2.0])
    seen = []

    def stop(src):
        seen.append(("stop", src[:, 0].tolist()))
        return src[:, 0] > 0

    def state(src):
        seen.append(("state", src[:, 0].tolist()))
        return src.new_ones((len(src), 117))

    monkeypatch.setattr(model.termination, "predict", stop)
    monkeypatch.setattr(model.state, "predict", state)
    pred = model.predict(source, terminated=torch.tensor([False, False, True]))
    assert seen == [("stop", [0.0, 1.0]), ("state", [0.0])]
    assert torch.equal(pred[:, 117], torch.tensor([0.0, 1.0, 1.0]))
    assert not pred[1:, :117].any()
    seen.clear()
    source[1:] = float("nan")
    model.predict(source, terminated=pred[:, 117].bool())
    assert seen == [("stop", [0.0]), ("state", [0.0])]
    seen.clear()
    model.predict(source, terminated=torch.ones(3, dtype=torch.bool))
    assert not seen
    with pytest.raises(ValueError, match="boolean"):
        model.predict(source, terminated=torch.ones(3))
    with pytest.raises(ValueError, match="provider action"):
        model.predict(source[:, :118])


class ToyStop(nn.Module):
    def state_prediction(self, src):
        action = src[:, 118]
        ball = torch.stack(
            [
                src[:, 0] + src[:, 6] / 8 + action,
                src[:, 1] + src[:, 8] / 8 + src[:, 3],
                src[:, 2] + src[:, 9],
                src[:, 3],
            ],
            1,
        )
        extra = torch.stack(
            [
                1 - src[:, 9],
                src[:, 7] + 1,
                torch.where(src[:, 1] >= 4, 12, src[:, 5]),
                src[:, 6] + action - 1,
                src[:, 4] + src[:, 6] / 8,
            ],
            1,
        )
        return torch.cat([ball, 1 - src[:, 10:118], extra], 1)

    def predict(self, src):
        stop = src[:, 1] >= 6
        state = self.state_prediction(src)
        state[stop] = 0
        return torch.cat([state, stop[:, None]], 1)


def sample_data():
    lengths = [4, 5, 2, 2, 5]
    n = sum(lengths)
    source = np.zeros((n, 119), np.float32)
    source[:, 3] = 2
    source[:, 4] = 16
    source[:, 5] = 16
    source[:, 6] = 10
    source[:, 118] = np.arange(n) % 3
    terminal = np.zeros(n, bool)
    starts = np.empty(n, int)
    offset = 0
    for i, length in enumerate(lengths):
        source[offset : offset + length, 1] = np.arange(length) * 2
        starts[offset : offset + length] = offset
        if i < 3:
            terminal[offset + length - 1] = True
        offset += length
    target = ToyStop().state_prediction(torch.from_numpy(source)).numpy()
    target[terminal] = np.nan
    return dict(
        x=source,
        target=target,
        terminal=terminal,
        starts=starts,
        episodes=np.zeros(n, int),
        steps=np.arange(n),
    )


@pytest.mark.parametrize("full", [False, True])
def test_stop_rollouts_match_bruteforce_for_early_missed_and_censored_lives(full):
    data = sample_data()
    model = ToyStop()
    result = evaluate_life_rollouts(model, data, horizons=(1, 3, 8), full_segments=full)
    one = result["one_step"]
    assert one["true_positives"] == 2 and one["false_negatives"] == 1
    assert one["false_positives"] == 3 and one["joint_errors"] == 4
    for key, actual in result["rollouts"].items():
        h = len(data["x"]) if full else int(key)
        roots = np.unique(data["starts"]) if full else range(len(data["x"]))
        expected = dict(
            windows=0,
            death_windows=0,
            survival_or_censored_windows=0,
            exact_death_step=0,
            premature_death_stop=0,
            missed_death=0,
            false_stop_without_reference_death=0,
            stop_timing_errors=0,
            exact_entire_window=0,
            state_transition_errors=0,
            simulated_transitions=0,
        )
        for start in roots:
            end = start + 1
            while end < len(data["x"]) and data["starts"][end] == data["starts"][start]:
                end += 1
            if not full and end - start < h and not data["terminal"][end - 1]:
                continue
            end = min(end, start + h)
            death = bool(data["terminal"][end - 1])
            expected["windows"] += 1
            expected["death_windows" if death else "survival_or_censored_windows"] += 1
            previous = None
            exact = True
            stopped_at = None
            for row in range(start, end):
                source = data["x"][row : row + 1].copy()
                if previous is not None:
                    source[0, [0, 2, 3]] = previous[[0, 2, 3]]
                    source[0, 1] = np.floor(previous[1])
                    source[0, 8] = np.rint(previous[1] % 1 * 8)
                    source[0, 10:118] = previous[4:112]
                    source[0, 9] = previous[112]
                    source[0, 7] = previous[113]
                    source[0, 5] = previous[114]
                    source[0, 6] = previous[115]
                    source[0, 4] = previous[116]
                pred = model.predict(torch.from_numpy(source)).numpy()[0]
                expected["simulated_transitions"] += 1
                if pred[117]:
                    stopped_at = row
                    exact &= bool(data["terminal"][row])
                    break
                if data["terminal"][row]:
                    exact = False
                else:
                    wrong = not np.array_equal(pred[:117], data["target"][row])
                    expected["state_transition_errors"] += int(wrong)
                    exact &= not wrong
                previous = pred[:117]
            expected["exact_entire_window"] += int(exact)
            if death:
                outcome = (
                    "missed_death"
                    if stopped_at is None
                    else "exact_death_step"
                    if stopped_at == end - 1
                    else "premature_death_stop"
                )
                expected[outcome] += 1
                expected["stop_timing_errors"] += int(outcome != "exact_death_step")
            elif stopped_at is not None:
                expected["false_stop_without_reference_death"] += 1
                expected["stop_timing_errors"] += 1
        for field, value in expected.items():
            assert actual[field] == value, (key, field)
        assert actual["reference_end_censored_misses"] == expected["missed_death"]
        assert (
            actual["entire_window_accuracy"]
            == expected["exact_entire_window"] / expected["windows"]
        )
    if full:
        r = result["rollouts"]["segment_end"]
        assert [
            r[k]
            for k in [
                "exact_death_step",
                "premature_death_stop",
                "missed_death",
                "false_stop_without_reference_death",
            ]
        ] == [1, 1, 1, 1]


def test_terminal_inside_life_and_invalid_live_targets_are_rejected():
    data = sample_data()
    data["terminal"][0] = True
    with pytest.raises(ValueError, match="end its life"):
        evaluate_life_rollouts(ToyStop(), data)
    data = sample_data()
    data["target"][0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_life_rollouts(ToyStop(), data)
