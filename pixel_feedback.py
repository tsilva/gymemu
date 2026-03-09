import torch


def feedback_from_logits(logits: torch.Tensor, mode: str):
    probs = torch.sigmoid(logits)
    binary = (probs >= 0.5).float()

    if mode == "soft":
        next_input = probs
    elif mode == "hard":
        next_input = binary
    elif mode == "ste":
        next_input = probs + (binary - probs).detach()
    else:
        raise ValueError(f"Unsupported pixel feedback mode '{mode}'")

    return probs, binary, next_input


def static_noop_mask(
    history_frames: torch.Tensor,
    action: torch.Tensor,
    history_motion_threshold: float = 0.0,
):
    if history_frames.size(1) < 2:
        history_motion = torch.zeros(history_frames.size(0), device=history_frames.device)
    else:
        history_motion = (
            history_frames[:, 1:, :, :] - history_frames[:, :-1, :, :]
        ).abs().sum(dim=(1, 2, 3))
    noop_action = action.argmax(dim=1) == 0
    return noop_action & (history_motion <= history_motion_threshold)
