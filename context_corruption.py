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


def _random_tensor(
    shape: tuple[int, ...],
    target_device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None,
    normal: bool,
) -> torch.Tensor:
    sample_device = target_device if generator is None else torch.device("cpu")
    if normal:
        tensor = torch.randn(shape, device=sample_device, dtype=dtype, generator=generator)
    else:
        tensor = torch.rand(shape, device=sample_device, dtype=dtype, generator=generator)
    if tensor.device != target_device:
        tensor = tensor.to(device=target_device, dtype=dtype)
    return tensor


def apply_seed_history_corruption(
    history_frames: torch.Tensor,
    strengths: torch.Tensor,
    max_strength: float,
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
    shared_noise = _random_tensor(
        (history_frames.size(0), 1, history_frames.size(2), history_frames.size(3)),
        target_device=history_frames.device,
        dtype=history_frames.dtype,
        generator=generator,
        normal=True,
    )
    noisy = history_frames + shared_noise * strength_scale
    corrupted = noisy.clamp(0.0, 1.0)

    if foreground_dropout_max > 0 and max_strength > 0:
        normalized_strengths = (strengths / float(max_strength)).clamp(0.0, 1.0)
        dropout_probs = normalized_strengths * float(foreground_dropout_max)
        foreground_mask = history_frames > 0.5
        shared_dropout = _random_tensor(
            (history_frames.size(0), 1, history_frames.size(2), history_frames.size(3)),
            target_device=history_frames.device,
            dtype=history_frames.dtype,
            generator=generator,
            normal=False,
        )
        dropout_mask = foreground_mask & (shared_dropout < dropout_probs[:, None, None, None])
        corrupted = corrupted.masked_fill(dropout_mask, 0.0)

    return corrupted
