import torch
import torch.nn as nn
import torch.nn.functional as F

COMPILED_STATE_PREFIX = "_orig_mod."


def _normalized_flow_grid(flow_pixels: torch.Tensor) -> torch.Tensor:
    batch_size, _, height, width = flow_pixels.shape
    device = flow_pixels.device
    dtype = flow_pixels.dtype

    y_coords = ((torch.arange(height, device=device, dtype=dtype) + 0.5) * 2.0 / height) - 1.0
    x_coords = ((torch.arange(width, device=device, dtype=dtype) + 0.5) * 2.0 / width) - 1.0
    base_y, base_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
    base_grid = torch.stack((base_x, base_y), dim=-1).unsqueeze(0).expand(batch_size, -1, -1, -1)

    scale_x = 2.0 / max(width, 1)
    scale_y = 2.0 / max(height, 1)
    flow_x = flow_pixels[:, 0, :, :] * scale_x
    flow_y = flow_pixels[:, 1, :, :] * scale_y
    flow_grid = torch.stack((flow_x, flow_y), dim=-1)
    return base_grid + flow_grid


def normalized_spatial_model_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized_state = {}
    for key, value in state_dict.items():
        normalized_key = key
        if normalized_key.startswith(COMPILED_STATE_PREFIX):
            normalized_key = normalized_key[len(COMPILED_STATE_PREFIX) :]
        normalized_state[normalized_key] = value
    return normalized_state


def load_spatial_model_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, list[str]]:
    model_state = model.state_dict()
    remapped_state = normalized_spatial_model_state_dict(state_dict)
    notes = []

    if remapped_state.keys() != state_dict.keys():
        notes.append("stripped torch.compile _orig_mod prefixes from checkpoint")

    legacy_decoder_keys = [
        "decoder.0.weight",
        "decoder.0.bias",
        "decoder.2.weight",
        "decoder.2.bias",
    ]
    if any(key in remapped_state for key in legacy_decoder_keys):
        legacy_to_current = {
            "decoder.0.weight": "residual_head.0.weight",
            "decoder.0.bias": "residual_head.0.bias",
            "decoder.2.weight": "residual_head.2.weight",
            "decoder.2.bias": "residual_head.2.bias",
        }
        for old_key, new_key in legacy_to_current.items():
            if old_key in remapped_state:
                remapped_state[new_key] = remapped_state.pop(old_key)
        notes.append("remapped legacy decoder weights into residual_head")

    for key, value in model_state.items():
        remapped_state.setdefault(key, value)

    incompatible = model.load_state_dict(remapped_state, strict=False)
    unexpected_keys = list(incompatible.unexpected_keys)
    missing_keys = list(incompatible.missing_keys)
    if not missing_keys:
        notes.append("filled missing parameters from current model init")

    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
        "notes": notes,
    }


class SpatialActionConditionedResidualBlock(nn.Module):
    def __init__(self, channels: int, n_actions: int, dilation: int):
        super().__init__()
        self.action_affine = nn.Linear(n_actions, channels * 2)
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        scale, bias = self.action_affine(action).chunk(2, dim=1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        hidden = x * (1.0 + 0.1 * torch.tanh(scale)) + 0.1 * torch.tanh(bias)
        hidden = F.gelu(self.conv1(hidden))
        hidden = self.conv2(hidden)
        return x + hidden


class SpatialLatentWorldModel(nn.Module):
    def __init__(
        self,
        history_length: int,
        n_actions: int,
        latent_channels: int = 32,
        refine_blocks: int = 4,
    ):
        super().__init__()
        self.history_length = history_length
        self.n_actions = n_actions
        self.latent_channels = latent_channels
        self.history_input_channels = history_length + max(0, history_length - 1)
        self.max_flow_pixels = 3.0

        stem_channels = max(16, latent_channels // 2)
        context_blocks = max(1, refine_blocks // 2)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.history_input_channels, stem_channels, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(stem_channels, latent_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.local_dynamics = nn.Sequential(
            nn.Conv2d(latent_channels + n_actions, latent_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.local_blocks = nn.ModuleList(
            [
                SpatialActionConditionedResidualBlock(
                    channels=latent_channels,
                    n_actions=n_actions,
                    dilation=2 ** (block_idx % 4),
                )
                for block_idx in range(refine_blocks)
            ]
        )

        self.context_down = nn.Sequential(
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.context_blocks = nn.ModuleList(
            [
                SpatialActionConditionedResidualBlock(
                    channels=latent_channels,
                    n_actions=n_actions,
                    dilation=2 ** (block_idx % 3),
                )
                for block_idx in range(context_blocks)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(latent_channels * 2, latent_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.output_blocks = nn.ModuleList(
            [
                SpatialActionConditionedResidualBlock(
                    channels=latent_channels,
                    n_actions=n_actions,
                    dilation=2 ** (block_idx % 3),
                )
                for block_idx in range(context_blocks)
            ]
        )
        self.latent_delta = nn.Conv2d(latent_channels, latent_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.latent_delta.weight)
        nn.init.zeros_(self.latent_delta.bias)

        self.residual_head = nn.Sequential(
            nn.Conv2d(latent_channels, stem_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(stem_channels, 1, kernel_size=3, padding=1),
        )
        self.flow_head = nn.Sequential(
            nn.Conv2d(latent_channels, stem_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(stem_channels, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)

    def encode_history(self, history_frames: torch.Tensor) -> torch.Tensor:
        if history_frames.size(1) >= 2:
            history_deltas = history_frames[:, 1:, :, :] - history_frames[:, :-1, :, :]
            history_input = torch.cat([history_frames, history_deltas], dim=1)
        else:
            history_input = history_frames
        return self.encoder(history_input)

    def predict_next_latent(
        self, latent_history: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        height, width = latent_history.shape[-2:]
        action_planes = action[:, :, None, None].expand(-1, -1, height, width)
        hidden = self.local_dynamics(torch.cat([latent_history, action_planes], dim=1))
        for block in self.local_blocks:
            hidden = block(hidden, action)

        context = self.context_down(hidden)
        for block in self.context_blocks:
            context = block(context, action)
        context = F.interpolate(
            context,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        hidden = self.fuse(torch.cat([hidden, context], dim=1))
        for block in self.output_blocks:
            hidden = block(hidden, action)

        delta = self.latent_delta(hidden)
        return latent_history + delta

    def decode_latent(self, latent_state: torch.Tensor) -> torch.Tensor:
        return self.residual_head(latent_state)

    def forward(self, history_frames: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        latent_history = self.encode_history(history_frames)
        next_latent = self.predict_next_latent(latent_history, action)
        residual_logits = self.decode_latent(next_latent)
        flow_pixels = torch.tanh(self.flow_head(next_latent)) * self.max_flow_pixels
        flow_grid = _normalized_flow_grid(flow_pixels)
        last_frame = history_frames[:, -1:, :, :]
        warped = F.grid_sample(
            last_frame,
            flow_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        baseline = warped.clamp(1e-4, 1.0 - 1e-4)
        return torch.logit(baseline) + residual_logits
