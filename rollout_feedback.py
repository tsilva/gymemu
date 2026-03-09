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
        raise ValueError(f"Unsupported rollout feedback mode '{mode}'")

    return probs, binary, next_input
