import argparse
import io
import json
import os
import urllib.parse
import urllib.request

import numpy as np
import pygame
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image

from device_utils import configure_torch, get_device, move_batch_tensor, prepare_conv_module
from game_config import BREAKOUT_CONFIG, infer_game_config
from preprocessing import has_valid_black_background, preprocess_frame

MODEL_COMPILE_DEFAULT = True
MODEL_LATENT_DIM = 32
MODEL_LATENT_NOISE_FACTOR = 0.0
IMAGE_CHANNELS = 1
IMAGE_WIDTH = 80
IMAGE_HEIGHT = 96
DISPLAY_SCALE = 4
DEFAULT_MPS_DISPLAY_SCALE = 3


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
            self._flattened_size = dummy_output.view(1, -1).shape[1]
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
            x = x.view(x.size(0), -1)
            x = self.fc_enc(x)
        return x

    def decode(self, z):
        if self.use_bottleneck:
            z = self.fc_dec(z)
            z = z.view(z.size(0), *self._conv_output_shape)
        return self.decoder_conv(z)

    def forward(self, x):
        z = self.encode(x)
        z_input = z
        if self.training and self.model_latent_noise_factor > 0:
            z_input = z_input + torch.randn_like(z_input) * self.model_latent_noise_factor
        return self.decode(z_input), z


class DynamicsModel(nn.Module):
    def __init__(self, z_dim=MODEL_LATENT_DIM, n_actions=BREAKOUT_CONFIG.n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(z_dim + n_actions),
            nn.Linear(z_dim + n_actions, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, z_dim),
        )
        nn.init.orthogonal_(self.net[1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z_and_a):
        return self.net(z_and_a)


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
        DynamicsModel(z_dim=args.latent_dim, n_actions=n_actions),
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
    params = urllib.parse.urlencode(
        {
            "dataset": dataset_id,
            "config": "default",
            "split": "train",
            "offset": 0,
            "length": 32,
        }
    )
    rows_url = f"https://datasets-server.huggingface.co/rows?{params}"
    with urllib.request.urlopen(rows_url) as response:
        payload = json.load(response)

    for row in payload["rows"]:
        image_url = row["row"]["observations"]["src"]
        with urllib.request.urlopen(image_url) as image_response:
            image = Image.open(io.BytesIO(image_response.read())).convert("RGB")
        if has_valid_black_background(image, game_config):
            return image

    raise RuntimeError(f"Could not find a valid start frame in dataset '{dataset_id}'")


def load_initial_frame(image_path, dataset_id, game_config):
    if image_path:
        image = Image.open(image_path)
    else:
        image = fetch_remote_start_frame(dataset_id, game_config)
    frame = preprocess_frame(image, game_config, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    frame_uint8 = (frame * 255).astype(np.uint8)
    frame_tensor = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0).float()
    return frame_uint8, frame_tensor


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
    return parser.parse_args()


def main():
    global DISPLAY_SCALE, IMAGE_WIDTH, IMAGE_HEIGHT

    args = parse_args()
    if args.image_size is not None:
        IMAGE_WIDTH = args.image_size
        IMAGE_HEIGHT = args.image_size
    else:
        IMAGE_WIDTH = args.image_width
        IMAGE_HEIGHT = args.image_height

    game_config = infer_game_config(dataset_id=args.dataset, game=args.game)
    device = get_device()
    configure_torch(device)

    display_scale = args.display_scale
    if display_scale is None:
        display_scale = DEFAULT_MPS_DISPLAY_SCALE if device.type == "mps" else DISPLAY_SCALE

    print(f"Using device: {device}")
    print(f"Game profile: {game_config.name}")
    print(f"Dataset: {args.dataset}")
    print(f"Frame size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")

    representation_model, dynamics_model = load_trained_models(
        args, device, game_config.n_actions
    )

    initial_frame_uint8, initial_frame_tensor = load_initial_frame(
        args.start_image,
        args.dataset,
        game_config,
    )
    initial_frame_tensor = move_batch_tensor(
        initial_frame_tensor, device, non_blocking=device.type == "cuda"
    )

    with torch.inference_mode():
        latent = representation_model.encode(initial_frame_tensor)

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
            z_and_a = torch.cat([latent, action_tensor], dim=1)
            delta_latent = dynamics_model(z_and_a)
            latent = latent + delta_latent
            recon = representation_model.decode(latent)
            recon_frame = recon.squeeze().detach().cpu().numpy()

        render_frame(screen, np.clip(recon_frame * 255, 0, 255).astype(np.uint8))
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
