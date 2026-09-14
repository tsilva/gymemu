"""Target-only masks for isolated, solid rectangular sprites in recorded RGB."""

import torch
from torch.nn import functional as F


@torch.no_grad()
def ball_region_mask(target, *, sprite_height, sprite_width, padding):
    """Return a padded mask and availability; zero or multiple candidates abstain.

    A candidate is a nonblack, constant-RGB rectangle with no same-color pixel
    touching its four-connected boundary. This matches the exploratory component
    detector for a solid sprite, without CPU scans or per-example synchronization.
    It deliberately abstains on merged/occluded sprites and ambiguous scenes.
    """
    batch, _, height, width = target.shape
    rows, cols = height - sprite_height + 1, width - sprite_width + 1
    if rows <= 0 or cols <= 0:
        return (
            torch.zeros((batch, 1, height, width), dtype=torch.bool, device=target.device),
            torch.zeros(batch, dtype=torch.bool, device=target.device),
        )
    pixels = F.pad(target.detach().float(), (1, 1, 1, 1), value=-1)
    color = pixels[:, :, 1 : rows + 1, 1 : cols + 1]
    candidates = color.ne(0).any(dim=1)

    def same(dy, dx):
        return (pixels[:, :, 1 + dy : 1 + dy + rows, 1 + dx : 1 + dx + cols] == color).all(
            dim=1
        )

    for dy in range(sprite_height):
        for dx in range(sprite_width):
            candidates = candidates & same(dy, dx)
        candidates = candidates & ~same(dy, -1) & ~same(dy, sprite_width)
    for dx in range(sprite_width):
        candidates = candidates & ~same(-1, dx) & ~same(sprite_height, dx)

    flat = candidates.flatten(1)
    available = flat.sum(dim=1) == 1
    position = flat.to(torch.int32).argmax(dim=1)
    top, left = position // cols, position % cols
    yy = torch.arange(height, device=target.device)[None, :, None]
    xx = torch.arange(width, device=target.device)[None, None, :]
    mask = (
        available[:, None, None]
        & (yy >= top[:, None, None] - padding)
        & (yy < top[:, None, None] + sprite_height + padding)
        & (xx >= left[:, None, None] - padding)
        & (xx < left[:, None, None] + sprite_width + padding)
    )
    return mask[:, None], available
