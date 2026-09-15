import torch

from play import Player


def test_input_stack_tracks_inference_and_reset():
    calls = []

    def predict(stack, action):
        calls.append(stack.clone())
        return torch.ones(1, 3, 8, 8)

    config = {"history": 4, "shape": [3, 8, 8], "action_values": [0]}
    initial = [torch.full((3, 8, 8), 0.25), torch.full((3, 8, 8), 0.5)]
    player = Player(predict, config, torch.device("cpu"), initial)
    ready = player.input_stack.clone()
    assert not player.has_prediction
    assert not ready[:2].any()
    assert torch.equal(ready[2:], torch.stack(initial))
    player.advance(0)
    assert player.has_prediction
    assert torch.equal(player.input_stack, calls[-1][0])
    assert torch.equal(player.input_stack, ready)
    assert not torch.equal(player.input_stack[-1], player.frame)
    player.advance(0)
    assert torch.equal(player.input_stack, calls[-1][0])
    assert player.input_stack[-1].eq(1).all()
    player.reset()
    assert not player.has_prediction
    assert torch.equal(player.input_stack, ready)
