import torch


def sample_history_corruption_strengths(
    batch_size: int,
    max_strength: float,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if max_strength <= 0:
        return torch.zeros(batch_size, device=device, dtype=torch.float32)
    return torch.rand(batch_size, device=device, generator=generator, dtype=torch.float32) * float(
        max_strength
    )


def apply_seed_history_corruption(
    history_frames: torch.Tensor,
    strengths: torch.Tensor,
    foreground_dropout_max: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if history_frames.ndim != 4:
        raise ValueError(
            "history_frames must have shape (batch, history, height, width) "
            f"but got {tuple(history_frames.shape)}"
        )

    strengths = strengths.to(device=history_frames.device, dtype=history_frames.dtype)
    if strengths.ndim != 1 or strengths.size(0) != history_frames.size(0):
        raise ValueError(
            "strengths must have shape (batch,) matching history_frames "
            f"but got {tuple(strengths.shape)}"
        )

    if torch.count_nonzero(strengths).item() == 0:
        return history_frames

    strength_scale = strengths[:, None, None, None]
    noise = torch.randn(
        history_frames.shape,
        device=history_frames.device,
        dtype=history_frames.dtype,
        generator=generator,
    )
    noisy = history_frames + noise * strength_scale
    corrupted = noisy.clamp(0.0, 1.0)

    if foreground_dropout_max > 0:
        dropout_probs = (strengths * float(foreground_dropout_max)).clamp(0.0, 1.0)
        foreground_mask = history_frames > 0.5
        dropout_mask = foreground_mask & (
            torch.rand(
                history_frames.shape,
                device=history_frames.device,
                dtype=history_frames.dtype,
                generator=generator,
            )
            < dropout_probs[:, None, None, None]
        )
        corrupted = corrupted.masked_fill(dropout_mask, 0.0)

    return corrupted
