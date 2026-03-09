import argparse
import os

import numpy as np
import pygame
import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image

from dataset_utils import infer_history_length
from device_utils import configure_torch, get_device, move_batch_tensor, prepare_conv_module
from game_config import BREAKOUT_CONFIG, infer_game_config
from preprocessing import has_valid_black_background, preprocess_frame
from rollout_feedback import feedback_from_logits
from spatial_model import SpatialLatentWorldModel, load_spatial_model_state_dict

MODEL_COMPILE_DEFAULT = True
IMAGE_WIDTH = 80
IMAGE_HEIGHT = 96
HISTORY_LENGTH = 1
DISPLAY_SCALE = 4
DEFAULT_MPS_DISPLAY_SCALE = 3
SPATIAL_LATENT_CHANNELS = 32
SPATIAL_REFINE_BLOCKS = 4


def maybe_compile(model, enabled, device):
    if enabled and device.type == "cuda":
        return torch.compile(model)
    return model


def resolve_model_path(dataset_id, use_local_models, models_dir):
    if use_local_models:
        dataset_name = dataset_id.replace("/", "__")
        return os.path.join(models_dir, f"{dataset_name}-spatial-dynamics.pt")

    return hf_hub_download(
        repo_id=f"{dataset_id}-spatial-dynamics",
        filename="model.pt",
    )


def load_spatial_model(args, device, n_actions):
    model = prepare_conv_module(
        SpatialLatentWorldModel(
            history_length=args.history_length,
            n_actions=n_actions,
            latent_channels=args.spatial_latent_channels,
            refine_blocks=args.spatial_refine_blocks,
        ),
        device,
    )
    model = maybe_compile(model, args.model_compile, device)
    model_path = resolve_model_path(args.dataset, args.use_local_models, args.models_dir)
    load_result = load_spatial_model_state_dict(
        model,
        torch.load(model_path, map_location=device, weights_only=True),
    )
    model.eval()

    print(f"Spatial dynamics model: {model_path}")
    for note in load_result["notes"]:
        print(f"  -> {note}")
    if load_result["missing_keys"]:
        print(f"  -> Missing keys after load: {len(load_result['missing_keys'])}")
    if load_result["unexpected_keys"]:
        print(f"  -> Unexpected keys after load: {len(load_result['unexpected_keys'])}")

    return model


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


def build_action_vector(keys, game_config, allow_fire=True):
    action = np.zeros(game_config.n_actions, dtype=np.float32)
    action[0] = 1.0

    fire_action_id = None
    horizontal_action_id = None
    for binding in game_config.key_bindings:
        if not keys[getattr(pygame, binding.pygame_key)]:
            continue
        if binding.label == "FIRE":
            fire_action_id = binding.action_id
            continue
        if binding.label in {"RIGHT", "LEFT"} and horizontal_action_id is None:
            horizontal_action_id = binding.action_id

    chosen_action_id = None
    if allow_fire and fire_action_id is not None:
        chosen_action_id = fire_action_id
    elif horizontal_action_id is not None:
        chosen_action_id = horizontal_action_id

    if chosen_action_id is not None:
        action[:] = 0.0
        action[chosen_action_id] = 1.0

    return action


def parse_args():
    parser = argparse.ArgumentParser(description="Run the spatial latent emulator")
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
        help="Optional local screenshot used to seed the frame history",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=None,
        help=(
            "Number of recent frames fed into the spatial world model. "
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
        help="Directory containing locally trained spatial checkpoints",
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
        "--feedback",
        choices=("soft", "hard"),
        default="soft",
        help="How model predictions are fed back into history at runtime",
    )
    parser.add_argument(
        "--spatial-latent-channels",
        type=int,
        default=SPATIAL_LATENT_CHANNELS,
        help="Channel count of the spatial latent map used by the world model",
    )
    parser.add_argument(
        "--spatial-refine-blocks",
        type=int,
        default=SPATIAL_REFINE_BLOCKS,
        help="Number of action-conditioned residual blocks in the spatial latent model",
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
    print(f"Frame size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"History length: {args.history_length}")
    print(f"Feedback mode: {args.feedback}")
    print(f"Spatial latent channels: {args.spatial_latent_channels}")
    print(f"Spatial refine blocks: {args.spatial_refine_blocks}")

    model = load_spatial_model(args, device, game_config.n_actions)

    initial_frame_uint8, initial_history_tensor = load_initial_history(
        args.start_image,
        args.dataset,
        game_config,
        args.history_length,
    )
    initial_history_tensor = move_batch_tensor(
        initial_history_tensor, device, non_blocking=device.type == "cuda"
    )
    frame_history = initial_history_tensor.squeeze(1).unsqueeze(0)

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
        if keys[pygame.K_ESCAPE]:
            running = False
            continue

        action = build_action_vector(keys, game_config)
        action_tensor = torch.from_numpy(action).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(frame_history, action_tensor)
            _, next_binary, next_input = feedback_from_logits(logits, args.feedback)
            frame_history = torch.cat([frame_history[:, 1:, :, :], next_input], dim=1)
            next_frame = next_binary[0, 0].detach().cpu().numpy()

        render_frame(screen, np.clip(next_frame * 255, 0, 255).astype(np.uint8))
        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
