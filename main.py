import argparse
import os

import numpy as np
import pygame
import torch
import torch.nn as nn
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

from dataset_utils import infer_history_length
from device_utils import configure_torch, get_device, move_batch_tensor, prepare_conv_module
from game_config import BREAKOUT_CONFIG, infer_game_config
from pixel_feedback import (
    advance_breakout_ball_state,
    clear_ball_like_components,
    erase_breakout_ball,
    feedback_from_logits,
    init_breakout_ball_state,
    overlay_breakout_ball,
    paddle_motion_mask,
    shift_paddle_frames,
    static_noop_mask,
)
from pixel_model import FrameDynamicsModel
from preprocessing import has_valid_black_background, preprocess_frame

MODEL_COMPILE_DEFAULT = True
MODEL_LATENT_DIM = 32
MODEL_LATENT_NOISE_FACTOR = 0.0
IMAGE_CHANNELS = 1
IMAGE_WIDTH = 80
IMAGE_HEIGHT = 96
HISTORY_LENGTH = 1
DISPLAY_SCALE = 4
DEFAULT_MPS_DISPLAY_SCALE = 3
PIXEL_REFINE_BLOCKS = 0


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=MODEL_LATENT_DIM):
        super().__init__()
        self.model_latent_noise_factor = MODEL_LATENT_NOISE_FACTOR
        self.use_bottleneck = latent_dim > 0

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
            dummy_output = self.encoder_conv(dummy_input)
            self._flattened_size = dummy_output.reshape(1, -1).shape[1]
            self._conv_output_shape = dummy_output.shape[1:]

        if self.use_bottleneck:
            self.fc_enc = nn.Linear(self._flattened_size, latent_dim)
            self.fc_dec = nn.Linear(latent_dim, self._flattened_size)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, IMAGE_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        if self.use_bottleneck:
            x = x.reshape(x.size(0), -1)
            x = self.fc_enc(x)
            x = torch.tanh(x)
        return x

    def decode(self, z):
        if self.use_bottleneck:
            z = torch.clamp(z, -1.0, 1.0)
            z = self.fc_dec(z)
            z = z.reshape(z.size(0), *self._conv_output_shape)
        return self.decoder_conv(z)

    def forward(self, x):
        z = self.encode(x)
        z_input = z
        if self.training and self.model_latent_noise_factor > 0:
            z_input = z_input + torch.randn_like(z_input) * self.model_latent_noise_factor
        return self.decode(z_input), z


class DynamicsModel(nn.Module):
    def __init__(
        self,
        z_dim=MODEL_LATENT_DIM,
        n_actions=BREAKOUT_CONFIG.n_actions,
        history_length=HISTORY_LENGTH,
    ):
        super().__init__()
        history_dim = z_dim * history_length
        self.n_actions = n_actions
        self.history_net = nn.Sequential(
            nn.LayerNorm(history_dim),
            nn.Linear(history_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        nn.init.orthogonal_(self.history_net[1].weight)
        self.action_heads = nn.ModuleList([nn.Linear(128, z_dim) for _ in range(n_actions)])
        for head in self.action_heads:
            nn.init.zeros_(head.bias)

    def forward(self, latent_history, action):
        hidden = self.history_net(latent_history)
        all_deltas = torch.stack([head(hidden) for head in self.action_heads], dim=1)
        action_weights = action.unsqueeze(-1)
        return (all_deltas * action_weights).sum(dim=1)


def maybe_compile(model, enabled, device):
    if enabled and device.type == "cuda":
        return torch.compile(model)
    return model


def resolve_model_path(dataset_id, artifact_name, use_local_models, models_dir):
    if use_local_models:
        dataset_name = dataset_id.replace("/", "__")
        return os.path.join(models_dir, f"{dataset_name}-{artifact_name}.pt")

    return hf_hub_download(
        repo_id=f"{dataset_id}-{artifact_name}",
        filename="model.pt",
    )


def load_trained_models(args, device, n_actions):
    if args.dynamics_mode == "pixel":
        dynamics_model = prepare_conv_module(
            FrameDynamicsModel(
                history_length=args.history_length,
                n_actions=n_actions,
                refine_blocks=args.pixel_refine_blocks,
            ),
            device,
        )
        dynamics_model = maybe_compile(dynamics_model, args.model_compile, device)
        dynamics_path = resolve_model_path(
            args.dataset,
            "pixel-dynamics",
            args.use_local_models,
            args.models_dir,
        )
        dynamics_model.load_state_dict(
            torch.load(dynamics_path, map_location=device, weights_only=True)
        )
        dynamics_model.eval()
        print(f"Pixel dynamics model: {dynamics_path}")
        return None, dynamics_model

    representation_model = prepare_conv_module(
        ConvAutoencoder(args.latent_dim), device
    )
    representation_model = maybe_compile(
        representation_model, args.model_compile, device
    )
    representation_path = resolve_model_path(
        args.dataset,
        "representation",
        args.use_local_models,
        args.models_dir,
    )
    representation_model.load_state_dict(
        torch.load(representation_path, map_location=device, weights_only=True)
    )
    representation_model.eval()

    dynamics_model = prepare_conv_module(
        DynamicsModel(
            z_dim=args.latent_dim,
            n_actions=n_actions,
            history_length=args.history_length,
        ),
        device,
    )
    dynamics_model = maybe_compile(dynamics_model, args.model_compile, device)
    dynamics_path = resolve_model_path(
        args.dataset,
        "dynamics",
        args.use_local_models,
        args.models_dir,
    )
    dynamics_model.load_state_dict(
        torch.load(dynamics_path, map_location=device, weights_only=True)
    )
    dynamics_model.eval()

    print(f"Representation model: {representation_path}")
    print(f"Dynamics model: {dynamics_path}")

    return representation_model, dynamics_model


def fetch_remote_start_frame(dataset_id, game_config):
    dataset = load_dataset(dataset_id, split="train", streaming=True)

    uses_stacked_samples = {
        "history",
        "action",
        "next_frame",
    }.issubset(set(dataset.features))
    if uses_stacked_samples:
        sample = next(iter(dataset))
        history = np.asarray(sample["history"], dtype=np.float32)
        return history

    history_frames = []
    for row in dataset:
        image = row["observations"]
        if has_valid_black_background(image, game_config):
            history_frames.append(
                preprocess_frame(
                    image,
                    game_config,
                    target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
                ).astype(np.float32, copy=False)
            )
            if len(history_frames) >= HISTORY_LENGTH:
                return np.stack(history_frames[-HISTORY_LENGTH:], axis=0)
        else:
            history_frames.clear()

    raise RuntimeError(
        f"Could not find {HISTORY_LENGTH} valid consecutive frames in dataset '{dataset_id}'"
    )


def load_initial_history(image_path, dataset_id, game_config, history_length):
    if image_path:
        image = Image.open(image_path)
        frame = preprocess_frame(
            image,
            game_config,
            target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        ).astype(np.float32, copy=False)
        history_frames = np.repeat(frame[None, ...], history_length, axis=0)
    else:
        history_frames = fetch_remote_start_frame(dataset_id, game_config)
        if history_frames.shape[0] != history_length:
            raise ValueError(
                f"Expected start history length {history_length}, got {history_frames.shape[0]}"
            )

    history_uint8 = (history_frames[-1] * 255).astype(np.uint8)
    history_tensor = torch.from_numpy(history_frames).unsqueeze(1).float()
    return history_uint8, history_tensor


def render_frame(screen, frame_uint8):
    surface = pygame.surfarray.make_surface(
        np.repeat(frame_uint8[:, :, None], 3, axis=2).swapaxes(0, 1)
    )
    if DISPLAY_SCALE != 1:
        surface = pygame.transform.scale(
            surface,
            (frame_uint8.shape[1] * DISPLAY_SCALE, frame_uint8.shape[0] * DISPLAY_SCALE),
        )
    screen.blit(surface, (0, 0))
    pygame.display.flip()


def build_action_vector(keys, game_config):
    action = np.zeros(game_config.n_actions, dtype=np.float32)
    action[0] = 1.0

    for binding in game_config.key_bindings:
        if keys[getattr(pygame, binding.pygame_key)]:
            action[:] = 0.0
            action[binding.action_id] = 1.0
            break

    return action


def parse_args():
    parser = argparse.ArgumentParser(description="Run the neural emulator")
    parser.add_argument(
        "--dataset",
        default=BREAKOUT_CONFIG.dataset_id,
        help="Dataset id used to resolve local or Hugging Face model artifacts",
    )
    parser.add_argument(
        "--game",
        default=BREAKOUT_CONFIG.name,
        help="Game profile for preprocessing and controls",
    )
    parser.add_argument(
        "--dynamics-mode",
        choices=("latent", "pixel"),
        default="latent",
        help="Run either the original latent model or a direct pixel predictor",
    )
    parser.add_argument(
        "--start-image",
        default=None,
        help="Optional local screenshot used to seed the latent state",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=MODEL_LATENT_DIM,
        help="Latent dimension size used during training",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=None,
        help=(
            "Number of recent frames fed into the dynamics model. "
            "Defaults to 1 unless inferred from the dataset name."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional square image size override applied to both width and height",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=IMAGE_WIDTH,
        help="Preprocessed frame width",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=IMAGE_HEIGHT,
        help="Preprocessed frame height",
    )
    parser.add_argument(
        "--use-local-models",
        action="store_true",
        help="Load `*.pt` files from --models-dir instead of Hugging Face",
    )
    parser.add_argument(
        "--models-dir",
        default="./models",
        help="Directory containing locally trained models",
    )
    parser.add_argument(
        "--model-compile",
        action=argparse.BooleanOptionalAction,
        default=MODEL_COMPILE_DEFAULT,
        help="Enable torch.compile when running on CUDA",
    )
    parser.add_argument(
        "--display-scale",
        type=int,
        default=None,
        help="Window scaling factor for the rendered frame",
    )
    parser.add_argument(
        "--pixel-feedback",
        choices=("soft", "hard"),
        default="soft",
        help="How pixel predictions are fed back into history at runtime",
    )
    parser.add_argument(
        "--pixel-refine-blocks",
        type=int,
        default=PIXEL_REFINE_BLOCKS,
        help="Number of extra action-conditioned residual refinement blocks in the pixel model",
    )
    parser.add_argument(
        "--pixel-static-noop-hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When history is static and the action is NOOP, keep the frame fixed exactly",
    )
    parser.add_argument(
        "--pixel-static-history-threshold",
        type=float,
        default=40.0,
        help="Maximum recent pixel motion sum treated as effectively static for NOOP hold",
    )
    parser.add_argument(
        "--pixel-static-predicted-diff-threshold",
        type=float,
        default=4.0,
        help="Maximum predicted changed-pixel count still treated as stationary during NOOP hold",
    )
    parser.add_argument(
        "--pixel-paddle-motion-hold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use a deterministic paddle shift for near-static LEFT/RIGHT motion",
    )
    parser.add_argument(
        "--pixel-paddle-motion-threshold",
        type=float,
        default=24.0,
        help="Maximum non-paddle recent motion sum treated as paddle-only movement",
    )
    parser.add_argument(
        "--pixel-paddle-shift",
        type=int,
        default=2,
        help="Horizontal paddle shift in pixels per LEFT/RIGHT step when paddle hold is active",
    )
    return parser.parse_args()


def main():
    global DISPLAY_SCALE, IMAGE_WIDTH, IMAGE_HEIGHT, HISTORY_LENGTH

    args = parse_args()
    args.history_length = infer_history_length(args.dataset, args.history_length)
    if args.image_size is not None:
        IMAGE_WIDTH = args.image_size
        IMAGE_HEIGHT = args.image_size
    else:
        IMAGE_WIDTH = args.image_width
        IMAGE_HEIGHT = args.image_height
    HISTORY_LENGTH = args.history_length

    game_config = infer_game_config(dataset_id=args.dataset, game=args.game)
    device = get_device()
    configure_torch(device)

    display_scale = args.display_scale
    if display_scale is None:
        display_scale = DEFAULT_MPS_DISPLAY_SCALE if device.type == "mps" else DISPLAY_SCALE

    print(f"Using device: {device}")
    print(f"Game profile: {game_config.name}")
    print(f"Dataset: {args.dataset}")
    print(f"Dynamics mode: {args.dynamics_mode}")
    print(f"Frame size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"History length: {args.history_length}")
    print(f"Pixel refine blocks: {args.pixel_refine_blocks}")
    print(f"Pixel static NOOP hold: {args.pixel_static_noop_hold}")
    print(f"Pixel static history threshold: {args.pixel_static_history_threshold}")
    print(
        "Pixel static predicted diff threshold: "
        f"{args.pixel_static_predicted_diff_threshold}"
    )
    print(f"Pixel paddle motion hold: {args.pixel_paddle_motion_hold}")
    print(f"Pixel paddle motion threshold: {args.pixel_paddle_motion_threshold}")
    print(f"Pixel paddle shift: {args.pixel_paddle_shift}")

    representation_model, dynamics_model = load_trained_models(
        args, device, game_config.n_actions
    )

    initial_frame_uint8, initial_history_tensor = load_initial_history(
        args.start_image,
        args.dataset,
        game_config,
        args.history_length,
    )
    initial_history_tensor = move_batch_tensor(
        initial_history_tensor, device, non_blocking=device.type == "cuda"
    )

    with torch.inference_mode():
        if args.dynamics_mode == "pixel":
            frame_history = initial_history_tensor.squeeze(1).unsqueeze(0)
            breakout_ball_enabled = game_config.name == BREAKOUT_CONFIG.name
            launch_action_id = next(
                (
                    binding.action_id
                    for binding in game_config.key_bindings
                    if binding.label == "FIRE"
                ),
                1,
            )
            ball_state = None
            if breakout_ball_enabled:
                initial_binary = (frame_history[0, -1] >= 0.5).float().detach().cpu()
                ball_state = init_breakout_ball_state(initial_binary)
                if ball_state is not None and ball_state.attached:
                    for history_index in range(frame_history.size(1)):
                        cleaned_frame = clear_ball_like_components(
                            frame_history[0, history_index].detach().cpu()
                        )
                        patched_frame = overlay_breakout_ball(
                            cleaned_frame,
                            ball_state,
                        )
                        frame_history[0, history_index] = patched_frame.to(
                            device=device,
                            dtype=frame_history.dtype,
                        )
                    initial_frame_uint8 = np.clip(
                        frame_history[0, -1].detach().cpu().numpy() * 255,
                        0,
                        255,
                    ).astype(np.uint8)
        else:
            latent_history = representation_model.encode(initial_history_tensor).unsqueeze(0)
            breakout_ball_enabled = False
            launch_action_id = 1
            ball_state = None

    pygame.init()
    window_size = (IMAGE_WIDTH * display_scale, IMAGE_HEIGHT * display_scale)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("gymemu")
    DISPLAY_SCALE = display_scale
    render_frame(screen, initial_frame_uint8)

    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        action = build_action_vector(keys, game_config)
        action_tensor = torch.from_numpy(action).unsqueeze(0).to(device)

        with torch.inference_mode():
            if args.dynamics_mode == "pixel":
                current_frame = frame_history[:, -1:, :, :]
                current_binary = (current_frame >= 0.5).float()
                deterministic_binary_frame = current_binary[0, 0]
                deterministic_input_frame = current_frame[0, 0]
                applied_prior = False
                next_binary = None
                next_input = None

                if args.pixel_static_noop_hold:
                    hold_mask = static_noop_mask(
                        frame_history,
                        action_tensor,
                        args.pixel_static_history_threshold,
                    )
                    if bool(hold_mask.any()):
                        logits = dynamics_model(frame_history, action_tensor)
                        _, model_binary, model_input = feedback_from_logits(
                            logits,
                            args.pixel_feedback,
                        )
                        predicted_diff = (model_binary - current_binary).abs().sum(dim=(1, 2, 3))
                        hold_mask = hold_mask & (
                            predicted_diff <= args.pixel_static_predicted_diff_threshold
                        )
                        if bool(hold_mask.all()):
                            next_binary = current_binary
                            next_input = current_frame
                            applied_prior = True
                        else:
                            next_binary = model_binary
                            next_input = model_input
                            if bool(hold_mask.any()):
                                hold_mask = hold_mask[:, None, None, None]
                                next_binary = torch.where(hold_mask, current_binary, next_binary)
                                next_input = torch.where(hold_mask, current_frame, next_input)
                paddle_mask = None
                if not applied_prior and args.pixel_paddle_motion_hold:
                    paddle_mask = paddle_motion_mask(
                        frame_history,
                        action_tensor,
                        args.pixel_paddle_motion_threshold,
                    )
                if paddle_mask is not None and bool(paddle_mask.any()):
                    shifted_binary, applied_mask = shift_paddle_frames(
                        current_binary,
                        action_tensor,
                        shift_pixels=args.pixel_paddle_shift,
                    )
                    paddle_mask = paddle_mask & applied_mask
                    if bool(paddle_mask.any()):
                        if next_binary is None or next_input is None:
                            logits = dynamics_model(frame_history, action_tensor)
                            _, next_binary, next_input = feedback_from_logits(
                                logits,
                                args.pixel_feedback,
                            )
                        paddle_mask = paddle_mask[:, None, None, None]
                        shifted_input = shifted_binary
                        deterministic_binary_frame = shifted_binary[0, 0]
                        deterministic_input_frame = shifted_input[0, 0]
                        next_binary = torch.where(paddle_mask, shifted_binary, next_binary)
                        next_input = torch.where(paddle_mask, shifted_input, next_input)
                        applied_prior = bool(paddle_mask.all())
                if next_binary is None or next_input is None:
                    logits = dynamics_model(frame_history, action_tensor)
                    _, next_binary, next_input = feedback_from_logits(
                        logits,
                        args.pixel_feedback,
                    )
                if breakout_ball_enabled:
                    action_id = int(action_tensor.argmax(dim=1).item())
                    attached_ball = ball_state is not None and ball_state.attached
                    # Keep the expensive connected-component ball logic on CPU. Scanning
                    # an MPS/CUDA tensor pixel-by-pixel from Python stalls the render loop.
                    deterministic_binary_frame = deterministic_binary_frame.detach().cpu()
                    deterministic_input_frame = deterministic_input_frame.detach().cpu()
                    next_binary_frame = next_binary[0, 0].detach().cpu()
                    next_input_frame = next_input[0, 0].detach().cpu()
                    deterministic_binary_frame = erase_breakout_ball(
                        deterministic_binary_frame,
                        ball_state,
                    )
                    deterministic_input_frame = erase_breakout_ball(
                        deterministic_input_frame,
                        ball_state,
                    )
                    deterministic_binary_frame = clear_ball_like_components(
                        deterministic_binary_frame
                    )
                    deterministic_input_frame = clear_ball_like_components(
                        deterministic_input_frame
                    )
                    next_binary_frame = clear_ball_like_components(next_binary_frame)
                    next_input_frame = clear_ball_like_components(next_input_frame)
                    next_binary_frame = erase_breakout_ball(next_binary_frame, ball_state)
                    next_input_frame = erase_breakout_ball(next_input_frame, ball_state)
                    if attached_ball:
                        next_binary_frame = deterministic_binary_frame
                        next_input_frame = deterministic_input_frame
                    else:
                        predicted_scene_diff = (
                            next_binary_frame - deterministic_binary_frame
                        ).abs().sum()
                        if predicted_scene_diff > args.pixel_static_predicted_diff_threshold:
                            next_binary_frame = deterministic_binary_frame
                            next_input_frame = deterministic_input_frame
                    ball_state = advance_breakout_ball_state(
                        ball_state,
                        action_id,
                        next_binary_frame,
                        launch_action_id=launch_action_id,
                        right_wall=IMAGE_WIDTH - 5,
                        bottom_wall=IMAGE_HEIGHT - 1,
                    )
                    next_binary_frame = overlay_breakout_ball(next_binary_frame, ball_state)
                    next_input_frame = overlay_breakout_ball(next_input_frame, ball_state)
                    next_binary = move_batch_tensor(
                        next_binary_frame.unsqueeze(0).unsqueeze(0),
                        device,
                    ).to(dtype=current_binary.dtype)
                    next_input = move_batch_tensor(
                        next_input_frame.unsqueeze(0).unsqueeze(0),
                        device,
                    ).to(dtype=current_frame.dtype)
                frame_history = torch.cat([frame_history[:, 1:, :, :], next_input], dim=1)
                recon_frame = next_binary.squeeze().detach().cpu().numpy()
            else:
                history_flat = latent_history.reshape(latent_history.size(0), -1)
                delta_latent = dynamics_model(history_flat, action_tensor)
                next_latent = latent_history[:, -1, :] + delta_latent
                latent_history = torch.cat(
                    [latent_history[:, 1:, :], next_latent.unsqueeze(1)],
                    dim=1,
                )
                recon = representation_model.decode(next_latent)
                recon_frame = recon.squeeze().detach().cpu().numpy()

        render_frame(screen, np.clip(recon_frame * 255, 0, 255).astype(np.uint8))
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
