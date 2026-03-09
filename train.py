"""
Training script for neural emulator models.
Trains ConvAutoencoder and DynamicsModel on Hugging Face datasets.
"""

import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset_utils import infer_history_length
from device_utils import configure_torch, get_device, move_batch_tensor, prepare_conv_module
from game_config import BREAKOUT_CONFIG, infer_game_config
from pixel_feedback import feedback_from_logits
from pixel_model import FrameDynamicsModel
from preprocessing import encode_action, has_valid_black_background, preprocess_frame
from spatial_model import SpatialLatentWorldModel
from wandb_utils import TrainingTracker, make_image_grid

# =============================================================================
# Configuration (must match main.py for model compatibility)
# =============================================================================

SEED = 42
MODEL_LATENT_DIM = 32
MODEL_LATENT_NOISE_FACTOR = 0.0

# Image dimensions after cropping the score bar.
IMAGE_CHANNELS = 1
IMAGE_WIDTH = 80
IMAGE_HEIGHT = 96
HISTORY_LENGTH = 1

# Training hyperparameters
TRAIN_N_EPOCHS = 50
TRAIN_BATCH_SIZE = 64
TRAIN_LEARNING_RATE = 0.001
TRAIN_MAX_GRAD_NORM = 0
TRAIN_WEIGHT_DECAY = 0
ENCODE_BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.0
MAX_FOREGROUND_LOSS_WEIGHT = 64.0
DICE_LOSS_WEIGHT = 1.0
PIXEL_CHANGE_LOSS_WEIGHT = 24.0
PIXEL_UNROLL_STEPS = 4
PIXEL_FEEDBACK_MODE = "soft"
PIXEL_REFINE_BLOCKS = 0
SPATIAL_LATENT_CHANNELS = 32
SPATIAL_REFINE_BLOCKS = 4

GAME_CONFIG = BREAKOUT_CONFIG
N_ACTIONS = GAME_CONFIG.n_actions

USE_BOTTLENECK = True
VAL_SPLIT_RATIO = 0.2

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device setup
device = get_device()
configure_torch(device)
print(f"Using device: {device}")


def empty_device_cache():
    if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def default_batch_size_for_device(current_device):
    if current_device.type == "mps":
        return 64
    if current_device.type == "cuda":
        return 128
    return 32


def dataloader_kwargs(current_device):
    return {
        "num_workers": 0,
        "pin_memory": current_device.type == "cuda",
    }


def is_val_improved(current_loss, best_loss, min_delta):
    return current_loss < (best_loss - min_delta)


def use_single_episode_val_slot(sequence_index, val_split):
    if val_split <= 0:
        return False
    return int((sequence_index + 1) * val_split) > int(sequence_index * val_split)


def metric_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    return float(value)


def mean_absolute_ratio(prediction, target):
    return (prediction - target).abs().mean()


def cosine_similarity_mean(prediction, target):
    return F.cosine_similarity(prediction, target, dim=1).mean()


def compute_grad_norm(parameters):
    grad_norm_sq = 0.0
    has_grad = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        has_grad = True
        grad_norm_sq += float(parameter.grad.detach().pow(2).sum().cpu().item())
    if not has_grad:
        return 0.0
    return grad_norm_sq ** 0.5


def action_histogram_dict(n_actions):
    return {action_id: 0 for action_id in range(n_actions)}


def log_action_histogram(target_hist, action_array):
    action_id = int(np.argmax(action_array))
    target_hist[action_id] += 1


def action_fraction_metrics(prefix, histogram):
    total = sum(histogram.values())
    metrics = {}
    for action_id, count in histogram.items():
        denominator = total if total > 0 else 1
        metrics[f"{prefix}/action_{action_id}"] = count / denominator
    return metrics


def sequence_action_fraction_metrics(train_sequences, val_sequences):
    train_hist = action_histogram_dict(N_ACTIONS)
    val_hist = action_histogram_dict(N_ACTIONS)
    for _, action_array, _ in train_sequences:
        log_action_histogram(train_hist, action_array)
    for _, action_array, _ in val_sequences:
        log_action_histogram(val_hist, action_array)
    metrics = {}
    metrics.update(action_fraction_metrics("data/train_action_fraction", train_hist))
    metrics.update(action_fraction_metrics("data/val_action_fraction", val_hist))
    return metrics


def fixed_sample_indices(n_items, n_samples):
    if n_items <= 0:
        return np.array([], dtype=np.int64)
    count = min(n_items, n_samples)
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(n_items, size=count, replace=False))


def numpy_frame_batch_to_tensor(frames):
    return torch.stack([torch.from_numpy(frame).unsqueeze(0).float() for frame in frames])


def sequence_frame_to_rgb(frame):
    frame_array = np.asarray(frame, dtype=np.float32)
    return np.clip(frame_array, 0.0, 1.0)


def reconstruction_components(recon, target):
    positive_ratio = target.mean(dim=(1, 2, 3), keepdim=True)
    foreground_weight = ((1.0 - positive_ratio) / positive_ratio.clamp_min(1e-6)).clamp(
        1.0,
        MAX_FOREGROUND_LOSS_WEIGHT,
    )
    weights = 1.0 + (foreground_weight - 1.0) * target
    bce = F.binary_cross_entropy(recon, target, weight=weights)
    intersection = (recon * target).sum(dim=(1, 2, 3))
    union = recon.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2.0 * intersection + 1e-6) / (union + 1e-6))
    loss_dice = dice.mean()
    foreground_pred_ratio = recon.mean()
    foreground_target_ratio = target.mean()
    return {
        "loss_total": bce + DICE_LOSS_WEIGHT * loss_dice,
        "loss_bce": bce,
        "loss_dice": loss_dice,
        "foreground_pred_ratio": foreground_pred_ratio,
        "foreground_target_ratio": foreground_target_ratio,
        "foreground_ratio_error": mean_absolute_ratio(
            foreground_pred_ratio, foreground_target_ratio
        ),
    }


def reconstruction_loss(recon, target):
    return reconstruction_components(recon, target)["loss_total"]


# =============================================================================
# Model Definitions (must match main.py)
# =============================================================================


class ConvAutoencoder(nn.Module):
    """ConvAutoencoder: Encodes frames into latent space and decodes back."""

    def __init__(self, latent_dim=MODEL_LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.model_latent_noise_factor = MODEL_LATENT_NOISE_FACTOR
        self.use_bottleneck = latent_dim > 0

        # Encoder convolutional layers
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        # Calculate flattened size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH)
            dummy_output = self.encoder_conv(dummy_input)
            self._flattened_size = dummy_output.reshape(1, -1).shape[1]
            self._conv_output_shape = dummy_output.shape[1:]

        # Bottleneck fully-connected layers
        if self.use_bottleneck:
            self.fc_enc = nn.Linear(self._flattened_size, latent_dim)
            self.fc_dec = nn.Linear(latent_dim, self._flattened_size)

        # Decoder transposed convolutional layers
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, IMAGE_CHANNELS, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        """Encode image to latent vector."""
        x = self.encoder_conv(x)
        if self.use_bottleneck:
            x = x.reshape(x.size(0), -1)
            x = self.fc_enc(x)
            x = torch.tanh(x)
        return x

    def decode(self, z):
        """Decode latent vector to image."""
        if self.use_bottleneck:
            z = torch.clamp(z, -1.0, 1.0)
            z = self.fc_dec(z)
            z = z.reshape(z.size(0), *self._conv_output_shape)
        z = self.decoder_conv(z)
        return z

    def forward(self, x):
        """Full forward pass: encode then decode."""
        z = self.encode(x)
        z_input = z
        if self.training and self.model_latent_noise_factor > 0:
            noise = torch.randn_like(z_input) * self.model_latent_noise_factor
            z_input += noise
        out = self.decode(z_input)
        return out, z


class DynamicsModel(nn.Module):
    """DynamicsModel: Predicts latent delta given (latent, action)."""

    def __init__(self, z_dim=MODEL_LATENT_DIM, n_actions=N_ACTIONS, history_length=HISTORY_LENGTH):
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
        """Predict delta in latent space."""
        hidden = self.history_net(latent_history)
        all_deltas = torch.stack([head(hidden) for head in self.action_heads], dim=1)
        action_weights = action.unsqueeze(-1)
        return (all_deltas * action_weights).sum(dim=1)


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================


def normalize_episode_id(raw_episode_id):
    if isinstance(raw_episode_id, bytes):
        return raw_episode_id.hex()
    return str(raw_episode_id)


def load_and_preprocess_dataset(dataset_id, val_split=VAL_SPLIT_RATIO):
    """
    Load dataset from Hugging Face and preprocess.

    Returns:
        train_sequences: list of (history_frames, action_t, next_frame) for training
        val_sequences: list of (history_frames, action_t, next_frame) for validation
    """
    print(f"\nLoading dataset: {dataset_id}")

    try:
        dataset = load_dataset(dataset_id, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Attempting to load with trust_remote_code=True...")
        dataset = load_dataset(dataset_id, split="train", trust_remote_code=True)

    print(f"Dataset loaded: {len(dataset)} samples")
    uses_stacked_samples = {
        "history",
        "action",
        "next_frame",
    }.issubset(set(dataset.column_names))
    if uses_stacked_samples:
        print("Dataset format: preprocessed stacked histories")
    else:
        print("Dataset format: raw observation transitions")

    # Group samples by episode
    episodes = {}
    for i, sample in enumerate(dataset):
        episode_id = normalize_episode_id(sample.get("episode_id", 0))
        if episode_id not in episodes:
            episodes[episode_id] = []
        episodes[episode_id].append((i, sample))

    print(f"Found {len(episodes)} episodes")

    # Split episodes into train/val
    episode_ids = list(episodes.keys())
    np.random.shuffle(episode_ids)

    if len(episode_ids) > 1:
        n_val_episodes = max(1, int(len(episode_ids) * val_split))
        val_episode_ids = set(episode_ids[:n_val_episodes])
        train_episode_ids = set(episode_ids[n_val_episodes:])
    else:
        val_episode_ids = set()
        train_episode_ids = set(episode_ids)

    print(
        f"Train episodes: {len(train_episode_ids)}, Val episodes: {len(val_episode_ids)}"
    )

    # Create sequences from episodes
    train_sequences = []
    val_sequences = []
    skipped_invalid_frames = 0
    skipped_terminal_pairs = 0
    train_action_hist = action_histogram_dict(N_ACTIONS)
    val_action_hist = action_histogram_dict(N_ACTIONS)

    for episode_id, samples in episodes.items():
        sequence_index = 0
        if uses_stacked_samples:
            samples.sort(key=lambda item: item[1].get("source_index", item[0]))

            for i, (_, sample) in enumerate(samples):
                history_frames = np.asarray(sample["history"], dtype=np.uint8)
                next_frame = np.asarray(sample["next_frame"], dtype=np.uint8)

                if history_frames.shape[0] != HISTORY_LENGTH:
                    raise ValueError(
                        f"Expected history length {HISTORY_LENGTH}, got {history_frames.shape[0]}"
                    )
                if history_frames.shape[1:] != (IMAGE_HEIGHT, IMAGE_WIDTH):
                    raise ValueError(
                        "Stacked history shape "
                        f"{history_frames.shape[1:]} does not match {(IMAGE_HEIGHT, IMAGE_WIDTH)}"
                    )
                if next_frame.shape != (IMAGE_HEIGHT, IMAGE_WIDTH):
                    raise ValueError(
                        "Next frame shape "
                        f"{next_frame.shape} does not match {(IMAGE_HEIGHT, IMAGE_WIDTH)}"
                    )

                try:
                    action_array = encode_action(sample["action"], GAME_CONFIG)
                except ValueError as exc:
                    print(f"Warning: Skipping sample {i} due to invalid action: {exc}")
                    continue

                if len(episode_ids) > 1:
                    target_list = (
                        val_sequences if episode_id in val_episode_ids else train_sequences
                    )
                else:
                    target_list = (
                        val_sequences
                        if use_single_episode_val_slot(sequence_index, val_split)
                        else train_sequences
                    )

                target_list.append((history_frames, action_array, next_frame))
                target_hist = (
                    val_action_hist if target_list is val_sequences else train_action_hist
                )
                log_action_histogram(target_hist, action_array)
                sequence_index += 1
            continue

        frame_history = deque(maxlen=HISTORY_LENGTH)
        prev_valid = False
        prev_action_array = None
        prev_cut = False

        samples.sort(key=lambda x: x[0])
        for i, (_, sample) in enumerate(samples):
            frame = sample["observations"]
            current_valid = has_valid_black_background(frame, GAME_CONFIG)
            current_processed = None
            current_action_array = None

            if current_valid:
                current_processed = preprocess_frame(
                    frame, GAME_CONFIG, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)
                )
                try:
                    current_action_array = encode_action(sample["actions"], GAME_CONFIG)
                except ValueError as exc:
                    print(f"Warning: Skipping sample {i} due to invalid action: {exc}")
                    current_valid = False
                    current_processed = None
                    skipped_invalid_frames += 1
            else:
                skipped_invalid_frames += 1

            if (
                prev_valid
                and current_valid
                and not prev_cut
                and len(frame_history) == HISTORY_LENGTH
            ):
                if len(episode_ids) > 1:
                    target_list = (
                        val_sequences if episode_id in val_episode_ids else train_sequences
                    )
                else:
                    target_list = (
                        val_sequences
                        if use_single_episode_val_slot(sequence_index, val_split)
                        else train_sequences
                    )

                target_list.append(
                    (np.stack(frame_history, axis=0), prev_action_array, current_processed)
                )
                target_hist = (
                    val_action_hist if target_list is val_sequences else train_action_hist
                )
                log_action_histogram(target_hist, prev_action_array)
                sequence_index += 1

            if sample.get("terminations") or sample.get("truncations"):
                skipped_terminal_pairs += 1
                prev_cut = True
            else:
                prev_cut = False

            if current_valid:
                frame_history.append(current_processed)
            else:
                frame_history.clear()

            prev_valid = current_valid
            prev_action_array = current_action_array

    print(f"Train sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    print(f"Skipped invalid-background pairs: {skipped_invalid_frames}")
    print(f"Skipped terminated/truncated pairs: {skipped_terminal_pairs}")

    dataset_stats = {
        "data/raw_samples": len(dataset),
        "data/episode_count": len(episode_ids),
        "data/train_episodes": len(train_episode_ids),
        "data/val_episodes": len(val_episode_ids),
        "data/train_sequences": len(train_sequences),
        "data/val_sequences": len(val_sequences),
        "data/skipped_invalid_background": skipped_invalid_frames,
        "data/skipped_terminal_pairs": skipped_terminal_pairs,
    }
    dataset_stats.update(action_fraction_metrics("data/train_action_fraction", train_action_hist))
    dataset_stats.update(action_fraction_metrics("data/val_action_fraction", val_action_hist))

    return train_sequences, val_sequences, dataset_stats


def load_pixel_rollout_dataset(
    dataset_id,
    unroll_steps=PIXEL_UNROLL_STEPS,
    val_split=VAL_SPLIT_RATIO,
    sequence_stride=1,
):
    """Load raw data and build rollout windows for pixel-space autoregressive training."""

    print(f"\nLoading dataset: {dataset_id}")

    try:
        dataset = load_dataset(dataset_id, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Attempting to load with trust_remote_code=True...")
        dataset = load_dataset(dataset_id, split="train", trust_remote_code=True)

    print(f"Dataset loaded: {len(dataset)} samples")
    uses_stacked_samples = {
        "history",
        "action",
        "next_frame",
    }.issubset(set(dataset.column_names))
    if uses_stacked_samples:
        raise ValueError(
            "Pixel unroll training currently supports raw observation datasets only"
        )

    episodes = {}
    for i, sample in enumerate(dataset):
        episode_id = normalize_episode_id(sample.get("episode_id", 0))
        if episode_id not in episodes:
            episodes[episode_id] = []
        episodes[episode_id].append((i, sample))

    print(f"Found {len(episodes)} episodes")

    episode_ids = list(episodes.keys())
    np.random.shuffle(episode_ids)

    if len(episode_ids) > 1:
        n_val_episodes = max(1, int(len(episode_ids) * val_split))
        val_episode_ids = set(episode_ids[:n_val_episodes])
    else:
        val_episode_ids = set()

    train_rollouts = []
    val_rollouts = []
    skipped_invalid_frames = 0
    skipped_terminal_pairs = 0
    train_action_hist = action_histogram_dict(N_ACTIONS)
    val_action_hist = action_histogram_dict(N_ACTIONS)

    for episode_id, samples in episodes.items():
        samples.sort(key=lambda item: item[0])
        segment_frames = []
        segment_actions = []
        kept_window_index = 0

        def flush_segment():
            nonlocal kept_window_index
            if len(segment_frames) < HISTORY_LENGTH + unroll_steps:
                segment_frames.clear()
                segment_actions.clear()
                return

            n_windows = len(segment_frames) - HISTORY_LENGTH - unroll_steps + 1
            for window_index in range(n_windows):
                if window_index % sequence_stride != 0:
                    continue

                history_frames = np.stack(
                    segment_frames[window_index : window_index + HISTORY_LENGTH],
                    axis=0,
                )
                action_seq = np.stack(
                    segment_actions[
                        window_index
                        + HISTORY_LENGTH
                        - 1 : window_index
                        + HISTORY_LENGTH
                        - 1
                        + unroll_steps
                    ],
                    axis=0,
                )
                target_frames = np.stack(
                    segment_frames[
                        window_index + HISTORY_LENGTH : window_index + HISTORY_LENGTH + unroll_steps
                    ],
                    axis=0,
                )

                if len(episode_ids) > 1:
                    target_list = (
                        val_rollouts if episode_id in val_episode_ids else train_rollouts
                    )
                else:
                    target_list = (
                        val_rollouts
                        if use_single_episode_val_slot(kept_window_index, val_split)
                        else train_rollouts
                    )
                    kept_window_index += 1

                target_list.append((history_frames, action_seq, target_frames))
                target_hist = val_action_hist if target_list is val_rollouts else train_action_hist
                for action_array in action_seq:
                    log_action_histogram(target_hist, action_array)

            segment_frames.clear()
            segment_actions.clear()

        for _, sample in samples:
            frame = sample["observations"]
            current_valid = has_valid_black_background(frame, GAME_CONFIG)
            if not current_valid:
                skipped_invalid_frames += 1
                flush_segment()
                continue

            processed = preprocess_frame(
                frame,
                GAME_CONFIG,
                target_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
            )
            try:
                action_array = encode_action(sample["actions"], GAME_CONFIG)
            except ValueError:
                skipped_invalid_frames += 1
                flush_segment()
                continue

            segment_frames.append(processed)
            segment_actions.append(action_array)

            if sample.get("terminations") or sample.get("truncations"):
                skipped_terminal_pairs += 1
                flush_segment()

        flush_segment()

    print(f"Train rollout sequences: {len(train_rollouts)}")
    print(f"Validation rollout sequences: {len(val_rollouts)}")
    print(f"Skipped invalid-background pairs: {skipped_invalid_frames}")
    print(f"Skipped terminated/truncated pairs: {skipped_terminal_pairs}")

    dataset_stats = {
        "data/raw_samples": len(dataset),
        "data/episode_count": len(episode_ids),
        "data/train_episodes": len(episode_ids) - len(val_episode_ids),
        "data/val_episodes": len(val_episode_ids),
        "data/train_sequences": len(train_rollouts),
        "data/val_sequences": len(val_rollouts),
        "data/skipped_invalid_background": skipped_invalid_frames,
        "data/skipped_terminal_pairs": skipped_terminal_pairs,
    }
    dataset_stats.update(action_fraction_metrics("data/train_action_fraction", train_action_hist))
    dataset_stats.update(action_fraction_metrics("data/val_action_fraction", val_action_hist))

    return train_rollouts, val_rollouts, dataset_stats


def extract_unique_frames(sequences):
    """Return a stable list of unique frames found in histories and targets."""

    unique_frames = []
    seen_frames = {}

    for history_frames, _, next_frame in sequences:
        for frame in history_frames:
            frame_key = tuple(frame.flatten())
            if frame_key not in seen_frames:
                seen_frames[frame_key] = len(unique_frames)
                unique_frames.append(frame)

        next_frame_key = tuple(next_frame.flatten())
        if next_frame_key not in seen_frames:
            seen_frames[next_frame_key] = len(unique_frames)
            unique_frames.append(next_frame)

    return unique_frames


class FrameDataset(Dataset):
    """PyTorch Dataset for preprocessed single frames."""

    def __init__(self, frames):
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        return torch.from_numpy(frame).unsqueeze(0).float()


# =============================================================================
# Training Functions
# =============================================================================


def train_autoencoder_phase(
    train_loader,
    val_loader,
    n_epochs,
    output_dir,
    dataset_name,
    early_stopping_patience,
    early_stopping_min_delta,
    tracker=None,
):
    """Phase 1: Train the ConvAutoencoder on frame reconstruction."""

    print("\n" + "=" * 60)
    print("PHASE 1: Training Autoencoder")
    print("=" * 60)

    model = prepare_conv_module(ConvAutoencoder(latent_dim=MODEL_LATENT_DIM), device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=TRAIN_LEARNING_RATE, weight_decay=TRAIN_WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    best_model_path = None
    best_epoch = None
    epochs_without_improvement = 0
    optimizer_step = 0
    eval_frames = build_autoencoder_eval_examples(getattr(val_loader.dataset, "frames", []))

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_metric_sums = {}
        n_batches = 0
        train_sample_count = 0
        train_epoch_start = time.perf_counter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Train]")
        for batch_idx, frames in enumerate(pbar, start=1):
            frames = move_batch_tensor(
                frames, device, non_blocking=device.type == "cuda"
            )
            batch_size = frames.size(0)

            optimizer.zero_grad(set_to_none=True)

            recon, z = model(frames)
            batch_metrics = reconstruction_components(recon, frames)
            batch_metrics["latent_std"] = z.std(unbiased=False)
            batch_metrics["latent_sat_frac"] = (z.abs() > 0.95).float().mean()
            loss = batch_metrics["loss_total"]

            loss.backward()

            grad_norm = compute_grad_norm(model.parameters())

            if TRAIN_MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_MAX_GRAD_NORM)

            optimizer.step()

            optimizer_step += 1
            n_batches += 1
            train_sample_count += batch_size
            add_weighted_metric_sums(train_metric_sums, batch_metrics, batch_size)
            if batch_idx == 1 or batch_idx % 20 == 0:
                pbar.set_postfix({"loss": f"{metric_value(loss):.6f}"})
                if tracker is not None:
                    tracker.log_batch(
                        "autoencoder",
                        {
                            "train/loss_total": metric_value(batch_metrics["loss_total"]),
                            "train/loss_bce": metric_value(batch_metrics["loss_bce"]),
                            "train/loss_dice": metric_value(batch_metrics["loss_dice"]),
                            "train/foreground_pred_ratio": metric_value(
                                batch_metrics["foreground_pred_ratio"]
                            ),
                            "train/foreground_target_ratio": metric_value(
                                batch_metrics["foreground_target_ratio"]
                            ),
                            "train/foreground_ratio_error": metric_value(
                                batch_metrics["foreground_ratio_error"]
                            ),
                            "train/latent_std": metric_value(batch_metrics["latent_std"]),
                            "train/latent_sat_frac": metric_value(
                                batch_metrics["latent_sat_frac"]
                            ),
                            "train/grad_norm": grad_norm,
                            "train/lr": optimizer.param_groups[0]["lr"],
                        },
                        optimizer_step,
                    )

        train_metrics = {
            f"train/{key}": value
            for key, value in normalize_metric_sums(train_metric_sums, train_sample_count).items()
        }
        train_epoch_time = time.perf_counter() - train_epoch_start
        train_metrics["train/epoch_time_s"] = train_epoch_time
        train_metrics["train/samples_per_s"] = (
            train_sample_count / train_epoch_time if train_epoch_time > 0 else 0.0
        )
        avg_train_loss = train_metrics["train/loss_total"]

        # Validation
        model.eval()
        val_metric_sums = {}
        n_val_batches = 0
        val_sample_count = 0
        val_epoch_start = time.perf_counter()

        with torch.inference_mode():
            for frames in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Val]", leave=False
            ):
                frames = move_batch_tensor(
                    frames, device, non_blocking=device.type == "cuda"
                )
                batch_size = frames.size(0)

                recon, z = model(frames)
                batch_metrics = reconstruction_components(recon, frames)
                batch_metrics["latent_std"] = z.std(unbiased=False)
                batch_metrics["latent_sat_frac"] = (z.abs() > 0.95).float().mean()
                add_weighted_metric_sums(val_metric_sums, batch_metrics, batch_size)
                val_sample_count += batch_size
                n_val_batches += 1

        val_metrics = {
            f"val/{key}": value
            for key, value in normalize_metric_sums(val_metric_sums, val_sample_count).items()
        }
        val_epoch_time = time.perf_counter() - val_epoch_start
        val_metrics["val/epoch_time_s"] = val_epoch_time
        val_metrics["val/samples_per_s"] = (
            val_sample_count / val_epoch_time if val_epoch_time > 0 else 0.0
        )
        avg_val_loss = val_metrics["val/loss_total"]

        print(
            "Epoch "
            f"{epoch + 1}/{n_epochs} - Train Loss: {avg_train_loss:.6f}, "
            f"Val Loss: {avg_val_loss:.6f}"
        )

        # Save best model
        best_val_improved = False
        if is_val_improved(avg_val_loss, best_val_loss, early_stopping_min_delta):
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_model_path = os.path.join(
                output_dir, f"{dataset_name}-representation.pt"
            )
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved best model (val_loss: {best_val_loss:.6f})")
            best_val_improved = True
        elif early_stopping_patience > 0:
            epochs_without_improvement += 1
            print(
                "  -> No validation improvement "
                f"({epochs_without_improvement}/{early_stopping_patience})"
            )
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    "  -> Early stopping triggered for autoencoder "
                    f"after epoch {epoch + 1}"
                )
                empty_device_cache()
                break

        epoch_metrics = {}
        epoch_metrics.update(train_metrics)
        epoch_metrics.update(val_metrics)
        epoch_metrics["health/generalization_gap_ratio"] = (
            avg_val_loss / avg_train_loss if avg_train_loss > 0 else 0.0
        )
        epoch_metrics["health/best_val_improved"] = float(best_val_improved)
        epoch_metrics["health/epochs_without_improvement"] = epochs_without_improvement
        if tracker is not None:
            tracker.log_epoch("autoencoder", epoch_metrics, epoch)

        if tracker is not None and eval_frames and (best_val_improved or (epoch + 1) % 5 == 0):
            with torch.inference_mode():
                eval_tensor = numpy_frame_batch_to_tensor(eval_frames)
                eval_tensor = move_batch_tensor(
                    eval_tensor, device, non_blocking=device.type == "cuda"
                )
                recon_eval, _ = model(eval_tensor)
            originals = [frame for frame in eval_frames]
            reconstructions = [frame[0] for frame in recon_eval.detach().cpu().numpy()]
            grid = make_autoencoder_media_grid(originals, reconstructions)
            tracker.log_media(
                "autoencoder",
                {
                    "media/recon_grid": tracker.image(
                        grid, caption=f"Epoch {epoch + 1} autoencoder reconstructions"
                    )
                },
                epoch,
            )

        empty_device_cache()

    print(f"\nPhase 1 complete. Best model saved to: {best_model_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return {
        "best_model_path": best_model_path,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


def encode_frames(model, sequences, batch_size=ENCODE_BATCH_SIZE):
    """
    Encode history frames and targets to latent vectors using the trained autoencoder.

    Returns:
        List of (latent_history, action_t, latent_next) tuples
    """
    model.eval()

    all_frames = []
    frame_to_idx = {}

    for history_frames, _, next_frame in sequences:
        for frame in history_frames:
            frame_tuple = tuple(frame.flatten())
            if frame_tuple not in frame_to_idx:
                frame_to_idx[frame_tuple] = len(all_frames)
                all_frames.append(frame)

        next_frame_tuple = tuple(next_frame.flatten())
        if next_frame_tuple not in frame_to_idx:
            frame_to_idx[next_frame_tuple] = len(all_frames)
            all_frames.append(next_frame)

    print(f"Encoding {len(all_frames)} unique frames...")

    # Batch encode frames
    all_latents = []

    with torch.inference_mode():
        for i in range(0, len(all_frames), batch_size):
            batch_frames = all_frames[i : i + batch_size]
            batch_tensor = torch.stack([
                torch.from_numpy(f).unsqueeze(0).float() for f in batch_frames
            ])
            batch_tensor = move_batch_tensor(
                batch_tensor, device, non_blocking=device.type == "cuda"
            )

            batch_latents = model.encode(batch_tensor)
            all_latents.extend(batch_latents.float().cpu().numpy())

    # Create latent sequences
    latent_sequences = []

    for history_frames, action_t, next_frame in sequences:
        latent_history = np.stack(
            [all_latents[frame_to_idx[tuple(frame.flatten())]] for frame in history_frames],
            axis=0,
        )
        latent_next = all_latents[frame_to_idx[tuple(next_frame.flatten())]]
        latent_sequences.append((latent_history, action_t, latent_next))

    return latent_sequences


class LatentDataset(Dataset):
    """PyTorch Dataset for (history_latents, action, next_latent) sequences."""

    def __init__(self, latent_sequences):
        self.latent_sequences = latent_sequences

    def __len__(self):
        return len(self.latent_sequences)

    def __getitem__(self, idx):
        latent_history, action_t, latent_next = self.latent_sequences[idx]

        latent_history_tensor = torch.from_numpy(latent_history).float()
        action_t_tensor = torch.from_numpy(action_t).float()
        latent_next_tensor = torch.from_numpy(latent_next).float()

        return latent_history_tensor, action_t_tensor, latent_next_tensor


class PixelRolloutDataset(Dataset):
    """PyTorch Dataset for autoregressive pixel-space rollout training."""

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        history_frames, action_seq, target_frames = self.sequences[idx]
        history_tensor = torch.from_numpy(history_frames).float()
        action_tensor = torch.from_numpy(action_seq).float()
        target_tensor = torch.from_numpy(target_frames).unsqueeze(1).float()
        return history_tensor, action_tensor, target_tensor


def pixel_dynamics_components(logits, target, current_frame):
    current_frame = current_frame.detach()
    positive_ratio = target.mean(dim=(1, 2, 3), keepdim=True)
    foreground_weight = ((1.0 - positive_ratio) / positive_ratio.clamp_min(1e-6)).clamp(
        1.0,
        MAX_FOREGROUND_LOSS_WEIGHT,
    )
    change_mask = (target - current_frame).abs()
    weights = 1.0 + (foreground_weight - 1.0) * target + PIXEL_CHANGE_LOSS_WEIGHT * change_mask
    probs = torch.sigmoid(logits)
    binary = (probs >= 0.5).float()
    intersection = (binary * target).sum(dim=(1, 2, 3))
    union = binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = ((2.0 * intersection + 1e-6) / (union + 1e-6)).mean()
    foreground_pred_ratio = probs.mean()
    foreground_target_ratio = target.mean()
    return {
        "loss_total": F.binary_cross_entropy_with_logits(logits, target, weight=weights),
        "foreground_weight_mean": foreground_weight.mean(),
        "change_weight_mean": change_mask.mean(),
        "weight_mean": weights.mean(),
        "dice": dice,
        "foreground_pred_ratio": foreground_pred_ratio,
        "foreground_target_ratio": foreground_target_ratio,
        "foreground_ratio_error": mean_absolute_ratio(
            foreground_pred_ratio, foreground_target_ratio
        ),
    }


def pixel_dynamics_loss(logits, target, current_frame):
    return pixel_dynamics_components(logits, target, current_frame)["loss_total"]


def add_weighted_metric_sums(metric_sums, metrics, weight):
    for key, value in metrics.items():
        metric_sums[key] = metric_sums.get(key, 0.0) + metric_value(value) * weight


def normalize_metric_sums(metric_sums, total_weight):
    if total_weight <= 0:
        return {key: 0.0 for key in metric_sums}
    return {key: value / total_weight for key, value in metric_sums.items()}


def build_autoencoder_eval_examples(frames, n_samples=8):
    indices = fixed_sample_indices(len(frames), n_samples)
    return [frames[index] for index in indices]


def build_latent_dynamics_eval_examples(val_sequences, val_latent_sequences, n_samples=6):
    indices = fixed_sample_indices(len(val_sequences), n_samples)
    return [
        {
            "history_frames": val_sequences[index][0],
            "action": val_sequences[index][1],
            "target_frame": val_sequences[index][2],
            "latent_history": val_latent_sequences[index][0],
            "latent_next": val_latent_sequences[index][2],
        }
        for index in indices
    ]


def build_pixel_rollout_eval_examples(val_sequences, n_samples=4):
    indices = fixed_sample_indices(len(val_sequences), n_samples)
    return [val_sequences[index] for index in indices]


def make_autoencoder_media_grid(originals, reconstructions):
    rows = []
    for original, reconstruction in zip(originals, reconstructions):
        diff = np.abs(reconstruction - original)
        rows.append([original, reconstruction, diff])
    return make_image_grid(rows)


def make_latent_dynamics_media_grid(history_frames, predictions, targets):
    rows = []
    for history_frame, prediction, target in zip(history_frames, predictions, targets):
        diff = np.abs(prediction - target)
        rows.append([history_frame, prediction, target, diff])
    return make_image_grid(rows)


def make_pixel_rollout_media_grid(history_frames, predicted_rollouts, target_rollouts):
    rows = []
    for history_frame, predicted_frames, target_frames in zip(
        history_frames, predicted_rollouts, target_rollouts
    ):
        rows.append([history_frame, *predicted_frames])
        rows.append([history_frame, *target_frames])
    return make_image_grid(rows)


def train_dynamics_phase(
    autoencoder_path,
    train_sequences,
    val_sequences,
    n_epochs,
    output_dir,
    dataset_name,
    early_stopping_patience,
    early_stopping_min_delta,
    tracker=None,
):
    """Phase 2: Train the DynamicsModel on latent delta prediction."""

    print("\n" + "=" * 60)
    print("PHASE 2: Training Dynamics Model")
    print("=" * 60)

    # Load trained autoencoder
    autoencoder = prepare_conv_module(
        ConvAutoencoder(latent_dim=MODEL_LATENT_DIM), device
    )
    autoencoder.load_state_dict(
        torch.load(autoencoder_path, map_location=device, weights_only=True)
    )
    autoencoder.eval()

    print(f"Loaded autoencoder from: {autoencoder_path}")

    # Encode all frames to latents
    print("\nEncoding training frames...")
    train_latent_sequences = encode_frames(autoencoder, train_sequences)

    print("Encoding validation frames...")
    val_latent_sequences = encode_frames(autoencoder, val_sequences)

    print(f"Train latent sequences: {len(train_latent_sequences)}")
    print(f"Val latent sequences: {len(val_latent_sequences)}")

    # Create datasets and loaders
    train_dataset = LatentDataset(train_latent_sequences)
    val_dataset = LatentDataset(val_latent_sequences)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        **dataloader_kwargs(device),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        **dataloader_kwargs(device),
    )

    # Initialize dynamics model
    dynamics_model = prepare_conv_module(
        DynamicsModel(
            z_dim=MODEL_LATENT_DIM,
            n_actions=N_ACTIONS,
            history_length=HISTORY_LENGTH,
        ),
        device,
    )

    optimizer = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=TRAIN_LEARNING_RATE,
        weight_decay=TRAIN_WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_model_path = None
    best_epoch = None
    epochs_without_improvement = 0
    optimizer_step = 0
    eval_examples = build_latent_dynamics_eval_examples(
        val_sequences, val_latent_sequences, n_samples=6
    )

    for epoch in range(n_epochs):
        # Training
        dynamics_model.train()
        train_metric_sums = {}
        n_batches = 0
        train_sample_count = 0
        train_epoch_start = time.perf_counter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Train]")
        for batch_idx, (latent_history, action_t, latent_next) in enumerate(pbar, start=1):
            latent_history = latent_history.to(device, non_blocking=device.type == "cuda")
            action_t = action_t.to(device, non_blocking=device.type == "cuda")
            latent_next = latent_next.to(device, non_blocking=device.type == "cuda")
            batch_size = latent_history.size(0)

            optimizer.zero_grad(set_to_none=True)

            history_flat = latent_history.reshape(latent_history.size(0), -1)
            delta_pred = dynamics_model(history_flat, action_t)
            delta_target = latent_next - latent_history[:, -1, :]
            loss = F.mse_loss(delta_pred, delta_target)

            loss.backward()

            grad_norm = compute_grad_norm(dynamics_model.parameters())

            if TRAIN_MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    dynamics_model.parameters(), TRAIN_MAX_GRAD_NORM
                )

            optimizer.step()

            optimizer_step += 1
            n_batches += 1
            train_sample_count += batch_size
            target_norm = delta_target.norm(dim=1)
            pred_norm = delta_pred.norm(dim=1)
            delta_norm_ratio = pred_norm.mean() / target_norm.mean().clamp_min(1e-8)
            batch_metrics = {
                "loss_total": loss,
                "loss_delta_mse": loss,
                "delta_cosine": cosine_similarity_mean(delta_pred, delta_target),
                "pred_delta_l2": pred_norm.mean(),
                "target_delta_l2": target_norm.mean(),
                "delta_norm_ratio": delta_norm_ratio,
            }
            add_weighted_metric_sums(train_metric_sums, batch_metrics, batch_size)
            if batch_idx == 1 or batch_idx % 20 == 0:
                pbar.set_postfix({"loss": f"{metric_value(loss):.6f}"})
                if tracker is not None:
                    tracker.log_batch(
                        "latent_dynamics",
                        {
                            "train/loss_total": metric_value(loss),
                            "train/loss_delta_mse": metric_value(loss),
                            "train/delta_cosine": metric_value(batch_metrics["delta_cosine"]),
                            "train/pred_delta_l2": metric_value(batch_metrics["pred_delta_l2"]),
                            "train/target_delta_l2": metric_value(
                                batch_metrics["target_delta_l2"]
                            ),
                            "train/delta_norm_ratio": metric_value(
                                batch_metrics["delta_norm_ratio"]
                            ),
                            "train/grad_norm": grad_norm,
                            "train/lr": optimizer.param_groups[0]["lr"],
                        },
                        optimizer_step,
                    )

        train_metrics = {
            f"train/{key}": value
            for key, value in normalize_metric_sums(train_metric_sums, train_sample_count).items()
        }
        train_epoch_time = time.perf_counter() - train_epoch_start
        train_metrics["train/epoch_time_s"] = train_epoch_time
        train_metrics["train/samples_per_s"] = (
            train_sample_count / train_epoch_time if train_epoch_time > 0 else 0.0
        )
        avg_train_loss = train_metrics["train/loss_total"]

        # Validation uses the ground-truth history window for each batch.
        dynamics_model.eval()
        val_metric_sums = {}
        n_val_batches = 0
        val_sample_count = 0
        val_epoch_start = time.perf_counter()

        with torch.inference_mode():
            for latent_history, action_t, latent_next in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Val]", leave=False
            ):
                latent_history = latent_history.to(
                    device, non_blocking=device.type == "cuda"
                )
                action_t = action_t.to(device, non_blocking=device.type == "cuda")
                latent_next = latent_next.to(device, non_blocking=device.type == "cuda")
                batch_size = latent_history.size(0)

                history_flat = latent_history.reshape(latent_history.size(0), -1)
                delta_pred = dynamics_model(history_flat, action_t)

                delta_target = latent_next - latent_history[:, -1, :]
                loss = F.mse_loss(delta_pred, delta_target)

                target_norm = delta_target.norm(dim=1)
                pred_norm = delta_pred.norm(dim=1)
                delta_norm_ratio = pred_norm.mean() / target_norm.mean().clamp_min(1e-8)
                batch_metrics = {
                    "loss_total": loss,
                    "loss_delta_mse": loss,
                    "delta_cosine": cosine_similarity_mean(delta_pred, delta_target),
                    "pred_delta_l2": pred_norm.mean(),
                    "target_delta_l2": target_norm.mean(),
                    "delta_norm_ratio": delta_norm_ratio,
                }
                add_weighted_metric_sums(val_metric_sums, batch_metrics, batch_size)
                val_sample_count += batch_size
                n_val_batches += 1

        val_metrics = {
            f"val/{key}": value
            for key, value in normalize_metric_sums(val_metric_sums, val_sample_count).items()
        }
        val_epoch_time = time.perf_counter() - val_epoch_start
        val_metrics["val/epoch_time_s"] = val_epoch_time
        val_metrics["val/samples_per_s"] = (
            val_sample_count / val_epoch_time if val_epoch_time > 0 else 0.0
        )

        if eval_examples:
            action_frames = np.stack([example["action"] for example in eval_examples], axis=0)
            target_frames = np.stack([example["target_frame"] for example in eval_examples], axis=0)
            latent_history = np.stack(
                [example["latent_history"] for example in eval_examples], axis=0
            )
            with torch.inference_mode():
                latent_history_tensor = torch.from_numpy(latent_history).float().to(device)
                action_tensor = torch.from_numpy(action_frames).float().to(device)
                target_tensor = torch.from_numpy(target_frames).unsqueeze(1).float().to(device)
                predicted_delta = dynamics_model(
                    latent_history_tensor.reshape(latent_history_tensor.size(0), -1),
                    action_tensor,
                )
                predicted_next = latent_history_tensor[:, -1, :] + predicted_delta
                decoded_next = autoencoder.decode(predicted_next)
                decoded_metrics = reconstruction_components(decoded_next, target_tensor)
            val_metrics["val/decoded_next_loss_bce"] = metric_value(
                decoded_metrics["loss_bce"]
            )
            val_metrics["val/decoded_next_loss_dice"] = metric_value(
                decoded_metrics["loss_dice"]
            )
            val_metrics["val/decoded_foreground_ratio_error"] = metric_value(
                decoded_metrics["foreground_ratio_error"]
            )

        avg_val_loss = val_metrics["val/loss_total"]

        print(
            "Epoch "
            f"{epoch + 1}/{n_epochs} - Train Loss: {avg_train_loss:.6f}, "
            f"Val Loss: {avg_val_loss:.6f}"
        )

        # Save best model
        best_val_improved = False
        if is_val_improved(avg_val_loss, best_val_loss, early_stopping_min_delta):
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_model_path = os.path.join(output_dir, f"{dataset_name}-dynamics.pt")
            torch.save(dynamics_model.state_dict(), best_model_path)
            print(f"  -> Saved best model (val_loss: {best_val_loss:.6f})")
            best_val_improved = True
        elif early_stopping_patience > 0:
            epochs_without_improvement += 1
            print(
                "  -> No validation improvement "
                f"({epochs_without_improvement}/{early_stopping_patience})"
            )
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    "  -> Early stopping triggered for dynamics model "
                    f"after epoch {epoch + 1}"
                )
                empty_device_cache()
                break

        epoch_metrics = {}
        epoch_metrics.update(train_metrics)
        epoch_metrics.update(val_metrics)
        epoch_metrics["health/generalization_gap_ratio"] = (
            avg_val_loss / avg_train_loss if avg_train_loss > 0 else 0.0
        )
        epoch_metrics["health/best_val_improved"] = float(best_val_improved)
        epoch_metrics["health/epochs_without_improvement"] = epochs_without_improvement
        if tracker is not None:
            tracker.log_epoch("latent_dynamics", epoch_metrics, epoch)

        if tracker is not None and eval_examples and (best_val_improved or (epoch + 1) % 5 == 0):
            with torch.inference_mode():
                latent_history_tensor = torch.from_numpy(
                    np.stack([example["latent_history"] for example in eval_examples], axis=0)
                ).float().to(device)
                action_tensor = torch.from_numpy(
                    np.stack([example["action"] for example in eval_examples], axis=0)
                ).float().to(device)
                predicted_delta = dynamics_model(
                    latent_history_tensor.reshape(latent_history_tensor.size(0), -1),
                    action_tensor,
                )
                predicted_next = latent_history_tensor[:, -1, :] + predicted_delta
                decoded_next = autoencoder.decode(predicted_next).detach().cpu().numpy()
            history_last_frames = [example["history_frames"][-1] for example in eval_examples]
            prediction_frames = [frame[0] for frame in decoded_next]
            target_frames = [example["target_frame"] for example in eval_examples]
            grid = make_latent_dynamics_media_grid(
                history_last_frames,
                prediction_frames,
                target_frames,
            )
            tracker.log_media(
                "latent_dynamics",
                {
                    "media/next_frame_grid": tracker.image(
                        grid, caption=f"Epoch {epoch + 1} latent dynamics predictions"
                    )
                },
                epoch,
            )

        empty_device_cache()

    print(f"\nPhase 2 complete. Best model saved to: {best_model_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return {
        "best_model_path": best_model_path,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


def train_frame_rollout_phase(
    model,
    train_sequences,
    val_sequences,
    n_epochs,
    output_dir,
    dataset_name,
    early_stopping_patience,
    early_stopping_min_delta,
    feedback_mode,
    tracker_prefix,
    artifact_suffix,
    tracker=None,
):
    train_dataset = PixelRolloutDataset(train_sequences)
    val_dataset = PixelRolloutDataset(val_sequences)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        **dataloader_kwargs(device),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=False,
        **dataloader_kwargs(device),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_LEARNING_RATE,
        weight_decay=TRAIN_WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_model_path = None
    best_epoch = None
    epochs_without_improvement = 0
    optimizer_step = 0
    eval_examples = build_pixel_rollout_eval_examples(val_sequences, n_samples=4)

    for epoch in range(n_epochs):
        model.train()
        train_metric_sums = {}
        train_sample_count = 0
        train_epoch_start = time.perf_counter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Train]")
        for batch_idx, (history_frames, action_seq, target_frames) in enumerate(pbar, start=1):
            history_frames = move_batch_tensor(
                history_frames, device, non_blocking=device.type == "cuda"
            )
            action_seq = action_seq.to(device, non_blocking=device.type == "cuda")
            target_frames = move_batch_tensor(
                target_frames, device, non_blocking=device.type == "cuda"
            )
            batch_size = history_frames.size(0)

            optimizer.zero_grad(set_to_none=True)

            rollout_history = history_frames
            loss = torch.zeros((), device=device)
            n_steps = action_seq.size(1)
            batch_metric_sums = {}
            first_step_metrics = None
            last_step_metrics = None
            for step_idx in range(n_steps):
                logits = model(rollout_history, action_seq[:, step_idx, :])
                target_frame = target_frames[:, step_idx, :, :, :]
                step_metrics = pixel_dynamics_components(
                    logits,
                    target_frame,
                    rollout_history[:, -1:, :, :],
                )
                loss = loss + step_metrics["loss_total"]
                add_weighted_metric_sums(batch_metric_sums, step_metrics, 1.0)
                if step_idx == 0:
                    first_step_metrics = step_metrics
                if step_idx == n_steps - 1:
                    last_step_metrics = step_metrics
                _, _, next_input = feedback_from_logits(logits, feedback_mode)
                rollout_history = torch.cat(
                    [rollout_history[:, 1:, :, :], next_input],
                    dim=1,
                )
            loss = loss / n_steps
            loss.backward()

            grad_norm = compute_grad_norm(model.parameters())
            if TRAIN_MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_MAX_GRAD_NORM)
            optimizer.step()

            optimizer_step += 1
            train_sample_count += batch_size
            batch_metrics = normalize_metric_sums(batch_metric_sums, n_steps)
            batch_metrics["loss_total"] = metric_value(loss)
            batch_metrics["loss_step_1"] = metric_value(first_step_metrics["loss_total"])
            batch_metrics["loss_step_last"] = metric_value(last_step_metrics["loss_total"])
            batch_metrics["loss_last_over_first"] = batch_metrics["loss_step_last"] / max(
                batch_metrics["loss_step_1"], 1e-8
            )
            batch_metrics["dice_step_1"] = metric_value(first_step_metrics["dice"])
            batch_metrics["dice_step_last"] = metric_value(last_step_metrics["dice"])
            add_weighted_metric_sums(train_metric_sums, batch_metrics, batch_size)
            if batch_idx == 1 or batch_idx % 20 == 0:
                pbar.set_postfix({"loss": f"{metric_value(loss):.6f}"})
                if tracker is not None:
                    tracker.log_batch(
                        tracker_prefix,
                        {
                            "train/loss_total": batch_metrics["loss_total"],
                            "train/loss_step_1": batch_metrics["loss_step_1"],
                            "train/loss_step_last": batch_metrics["loss_step_last"],
                            "train/loss_last_over_first": batch_metrics["loss_last_over_first"],
                            "train/dice_step_1": batch_metrics["dice_step_1"],
                            "train/dice_step_last": batch_metrics["dice_step_last"],
                            "train/foreground_ratio_error": batch_metrics[
                                "foreground_ratio_error"
                            ],
                            "train/foreground_weight_mean": batch_metrics[
                                "foreground_weight_mean"
                            ],
                            "train/change_weight_mean": batch_metrics["change_weight_mean"],
                            "train/weight_mean": batch_metrics["weight_mean"],
                            "train/grad_norm": grad_norm,
                            "train/lr": optimizer.param_groups[0]["lr"],
                        },
                        optimizer_step,
                    )

        train_metrics = {
            f"train/{key}": value
            for key, value in normalize_metric_sums(train_metric_sums, train_sample_count).items()
        }
        train_epoch_time = time.perf_counter() - train_epoch_start
        train_metrics["train/epoch_time_s"] = train_epoch_time
        train_metrics["train/samples_per_s"] = (
            train_sample_count / train_epoch_time if train_epoch_time > 0 else 0.0
        )
        avg_train_loss = train_metrics["train/loss_total"]

        model.eval()
        val_metric_sums = {}
        val_sample_count = 0
        val_epoch_start = time.perf_counter()

        with torch.inference_mode():
            for history_frames, action_seq, target_frames in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Val]", leave=False
            ):
                history_frames = move_batch_tensor(
                    history_frames, device, non_blocking=device.type == "cuda"
                )
                action_seq = action_seq.to(device, non_blocking=device.type == "cuda")
                target_frames = move_batch_tensor(
                    target_frames, device, non_blocking=device.type == "cuda"
                )
                batch_size = history_frames.size(0)

                rollout_history = history_frames
                loss = torch.zeros((), device=device)
                n_steps = action_seq.size(1)
                batch_metric_sums = {}
                first_step_metrics = None
                last_step_metrics = None
                for step_idx in range(n_steps):
                    logits = model(rollout_history, action_seq[:, step_idx, :])
                    target_frame = target_frames[:, step_idx, :, :, :]
                    step_metrics = pixel_dynamics_components(
                        logits,
                        target_frame,
                        rollout_history[:, -1:, :, :],
                    )
                    loss = loss + step_metrics["loss_total"]
                    add_weighted_metric_sums(batch_metric_sums, step_metrics, 1.0)
                    if step_idx == 0:
                        first_step_metrics = step_metrics
                    if step_idx == n_steps - 1:
                        last_step_metrics = step_metrics
                    _, _, next_input = feedback_from_logits(logits, feedback_mode)
                    rollout_history = torch.cat(
                        [rollout_history[:, 1:, :, :], next_input],
                        dim=1,
                    )
                loss = loss / n_steps

                batch_metrics = normalize_metric_sums(batch_metric_sums, n_steps)
                batch_metrics["loss_total"] = metric_value(loss)
                batch_metrics["loss_step_1"] = metric_value(first_step_metrics["loss_total"])
                batch_metrics["loss_step_last"] = metric_value(last_step_metrics["loss_total"])
                batch_metrics["loss_last_over_first"] = batch_metrics["loss_step_last"] / max(
                    batch_metrics["loss_step_1"], 1e-8
                )
                batch_metrics["dice_step_1"] = metric_value(first_step_metrics["dice"])
                batch_metrics["dice_step_last"] = metric_value(last_step_metrics["dice"])
                add_weighted_metric_sums(val_metric_sums, batch_metrics, batch_size)
                val_sample_count += batch_size

        val_metrics = {
            f"val/{key}": value
            for key, value in normalize_metric_sums(val_metric_sums, val_sample_count).items()
        }
        val_epoch_time = time.perf_counter() - val_epoch_start
        val_metrics["val/epoch_time_s"] = val_epoch_time
        val_metrics["val/samples_per_s"] = (
            val_sample_count / val_epoch_time if val_epoch_time > 0 else 0.0
        )
        avg_val_loss = val_metrics["val/loss_total"]

        print(
            "Epoch "
            f"{epoch + 1}/{n_epochs} - Train Loss: {avg_train_loss:.6f}, "
            f"Val Loss: {avg_val_loss:.6f}"
        )

        best_val_improved = False
        if is_val_improved(avg_val_loss, best_val_loss, early_stopping_min_delta):
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_model_path = os.path.join(output_dir, f"{dataset_name}-{artifact_suffix}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved best model (val_loss: {best_val_loss:.6f})")
            best_val_improved = True
        elif early_stopping_patience > 0:
            epochs_without_improvement += 1
            print(
                "  -> No validation improvement "
                f"({epochs_without_improvement}/{early_stopping_patience})"
            )
            if epochs_without_improvement >= early_stopping_patience:
                print(
                    "  -> Early stopping triggered for "
                    f"{tracker_prefix.replace('_', ' ')} after epoch {epoch + 1}"
                )
                empty_device_cache()
                break

        epoch_metrics = {}
        epoch_metrics.update(train_metrics)
        epoch_metrics.update(val_metrics)
        epoch_metrics["health/generalization_gap_ratio"] = (
            avg_val_loss / avg_train_loss if avg_train_loss > 0 else 0.0
        )
        epoch_metrics["health/best_val_improved"] = float(best_val_improved)
        epoch_metrics["health/epochs_without_improvement"] = epochs_without_improvement
        if tracker is not None:
            tracker.log_epoch(tracker_prefix, epoch_metrics, epoch)

        if tracker is not None and eval_examples and (best_val_improved or (epoch + 1) % 5 == 0):
            predicted_rollouts = []
            target_rollouts = []
            history_last_frames = []
            with torch.inference_mode():
                for history_frames, action_seq, target_frames in eval_examples:
                    history_tensor = (
                        torch.from_numpy(history_frames).float().unsqueeze(0).to(device)
                    )
                    action_tensor = torch.from_numpy(action_seq).float().unsqueeze(0).to(device)
                    rollout_history = history_tensor
                    rollout_predictions = []
                    for step_idx in range(action_tensor.size(1)):
                        logits = model(rollout_history, action_tensor[:, step_idx, :])
                        probs, _, next_input = feedback_from_logits(logits, feedback_mode)
                        rollout_predictions.append(probs[0, 0].detach().cpu().numpy())
                        rollout_history = torch.cat(
                            [rollout_history[:, 1:, :, :], next_input],
                            dim=1,
                        )
                    predicted_rollouts.append(rollout_predictions)
                    target_rollouts.append([frame for frame in target_frames])
                    history_last_frames.append(history_frames[-1])

            grid = make_pixel_rollout_media_grid(
                history_last_frames,
                predicted_rollouts,
                target_rollouts,
            )
            tracker.log_media(
                tracker_prefix,
                {
                    "media/rollout_strip": tracker.image(
                        grid,
                        caption=(
                            f"Epoch {epoch + 1} "
                            f"{tracker_prefix.replace('_', ' ')} rollouts"
                        ),
                    )
                },
                epoch,
            )

        empty_device_cache()

    return {
        "best_model_path": best_model_path,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


def train_pixel_dynamics_phase(
    train_sequences,
    val_sequences,
    n_epochs,
    output_dir,
    dataset_name,
    early_stopping_patience,
    early_stopping_min_delta,
    pixel_dynamics_path=None,
    pixel_feedback_mode=PIXEL_FEEDBACK_MODE,
    pixel_refine_blocks=PIXEL_REFINE_BLOCKS,
    tracker=None,
):
    """Train a direct frame predictor on history windows and actions."""

    print("\n" + "=" * 60)
    print("PHASE 1: Training Pixel Dynamics Model")
    print("=" * 60)

    model = prepare_conv_module(
        FrameDynamicsModel(
            history_length=HISTORY_LENGTH,
            n_actions=N_ACTIONS,
            refine_blocks=pixel_refine_blocks,
        ),
        device,
    )
    if pixel_dynamics_path:
        load_result = model.load_state_dict(
            torch.load(pixel_dynamics_path, map_location=device, weights_only=True),
            strict=pixel_refine_blocks == 0,
        )
        print(f"Loaded pixel dynamics model from: {pixel_dynamics_path}")
        if pixel_refine_blocks > 0:
            print(f"Missing keys after partial load: {len(load_result.missing_keys)}")
            print(f"Unexpected keys after partial load: {len(load_result.unexpected_keys)}")

    result = train_frame_rollout_phase(
        model,
        train_sequences,
        val_sequences,
        n_epochs,
        output_dir,
        dataset_name,
        early_stopping_patience,
        early_stopping_min_delta,
        pixel_feedback_mode,
        tracker_prefix="pixel_dynamics",
        artifact_suffix="pixel-dynamics",
        tracker=tracker,
    )
    print(f"\nPhase 1 complete. Best model saved to: {result['best_model_path']}")
    print(f"Best validation loss: {result['best_val_loss']:.6f}")
    return result


def train_spatial_dynamics_phase(
    train_sequences,
    val_sequences,
    n_epochs,
    output_dir,
    dataset_name,
    early_stopping_patience,
    early_stopping_min_delta,
    spatial_dynamics_path=None,
    pixel_feedback_mode=PIXEL_FEEDBACK_MODE,
    spatial_latent_channels=SPATIAL_LATENT_CHANNELS,
    spatial_refine_blocks=SPATIAL_REFINE_BLOCKS,
    tracker=None,
):
    """Train an end-to-end spatial-latent world model."""

    print("\n" + "=" * 60)
    print("PHASE 1: Training Spatial Latent Dynamics Model")
    print("=" * 60)

    model = prepare_conv_module(
        SpatialLatentWorldModel(
            history_length=HISTORY_LENGTH,
            n_actions=N_ACTIONS,
            latent_channels=spatial_latent_channels,
            refine_blocks=spatial_refine_blocks,
        ),
        device,
    )
    if spatial_dynamics_path:
        load_result = model.load_state_dict(
            torch.load(spatial_dynamics_path, map_location=device, weights_only=True),
            strict=spatial_refine_blocks == SPATIAL_REFINE_BLOCKS,
        )
        print(f"Loaded spatial dynamics model from: {spatial_dynamics_path}")
        if spatial_refine_blocks != SPATIAL_REFINE_BLOCKS:
            print(f"Missing keys after partial load: {len(load_result.missing_keys)}")
            print(f"Unexpected keys after partial load: {len(load_result.unexpected_keys)}")

    result = train_frame_rollout_phase(
        model,
        train_sequences,
        val_sequences,
        n_epochs,
        output_dir,
        dataset_name,
        early_stopping_patience,
        early_stopping_min_delta,
        pixel_feedback_mode,
        tracker_prefix="spatial_dynamics",
        artifact_suffix="spatial-dynamics",
        tracker=tracker,
    )
    print(f"\nPhase 1 complete. Best model saved to: {result['best_model_path']}")
    print(f"Best validation loss: {result['best_val_loss']:.6f}")
    return result


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    # Declare globals at the very beginning of the function
    global GAME_CONFIG, MODEL_LATENT_DIM, IMAGE_HEIGHT, IMAGE_WIDTH, TRAIN_BATCH_SIZE
    global TRAIN_N_EPOCHS, N_ACTIONS, ENCODE_BATCH_SIZE, HISTORY_LENGTH
    global EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA, TRAIN_LEARNING_RATE

    parser = argparse.ArgumentParser(description="Train neural emulator models")
    parser.add_argument(
        "--dataset",
        type=str,
        default=BREAKOUT_CONFIG.dataset_id,
        help="Hugging Face dataset ID",
    )
    parser.add_argument(
        "--game",
        type=str,
        default=BREAKOUT_CONFIG.name,
        help="Game profile to use for preprocessing and controls",
    )
    parser.add_argument(
        "--dynamics-mode",
        choices=("latent", "pixel", "spatial"),
        default="latent",
        help="Train a latent model, a direct pixel predictor, or a spatial-latent world model",
    )
    parser.add_argument(
        "--epochs", type=int, default=TRAIN_N_EPOCHS, help="Number of epochs per phase"
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Stop after this many epochs without validation improvement. Use 0 to disable.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=EARLY_STOPPING_MIN_DELTA,
        help="Minimum validation-loss improvement required to reset early stopping.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training; defaults to an MPS-safe value on Apple Silicon",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=TRAIN_LEARNING_RATE,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=MODEL_LATENT_DIM, help="Latent dimension size"
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=None,
        help=(
            "Number of recent frames to feed into the dynamics model. "
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
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--sequence-stride",
        type=int,
        default=1,
        help="Keep every Nth training sequence after preprocessing to reduce redundancy",
    )
    parser.add_argument(
        "--pixel-unroll-steps",
        type=int,
        default=PIXEL_UNROLL_STEPS,
        help="Number of autoregressive steps to unroll during pixel-space training",
    )
    parser.add_argument(
        "--pixel-feedback-mode",
        choices=("soft", "ste"),
        default=PIXEL_FEEDBACK_MODE,
        help="How pixel predictions are fed back into history during training rollouts",
    )
    parser.add_argument(
        "--pixel-refine-blocks",
        type=int,
        default=PIXEL_REFINE_BLOCKS,
        help="Number of extra action-conditioned residual refinement blocks in the pixel model",
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
        help="Number of action-conditioned residual blocks in the spatial latent dynamics model",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=None,
        help="Batch size used when encoding frames into latent space",
    )
    parser.add_argument(
        "--skip-autoencoder",
        action="store_true",
        help="Skip autoencoder training (if already trained)",
    )
    parser.add_argument(
        "--skip-dynamics", action="store_true", help="Skip dynamics model training"
    )
    parser.add_argument(
        "--autoencoder-path",
        type=str,
        default=None,
        help="Path to pre-trained autoencoder for dynamics training",
    )
    parser.add_argument(
        "--pixel-dynamics-path",
        type=str,
        default=None,
        help="Optional path to a pre-trained pixel dynamics checkpoint to resume from",
    )
    parser.add_argument(
        "--spatial-dynamics-path",
        type=str,
        default=None,
        help="Optional path to a pre-trained spatial dynamics checkpoint to resume from",
    )

    load_dotenv()
    args = parser.parse_args()

    GAME_CONFIG = infer_game_config(dataset_id=args.dataset, game=args.game)
    N_ACTIONS = GAME_CONFIG.n_actions
    HISTORY_LENGTH = infer_history_length(args.dataset, args.history_length)

    # Update global config from args
    MODEL_LATENT_DIM = args.latent_dim
    if args.image_size is not None:
        IMAGE_WIDTH = args.image_size
        IMAGE_HEIGHT = args.image_size
    else:
        IMAGE_WIDTH = args.image_width
        IMAGE_HEIGHT = args.image_height
    TRAIN_BATCH_SIZE = args.batch_size or default_batch_size_for_device(device)
    TRAIN_LEARNING_RATE = args.learning_rate
    ENCODE_BATCH_SIZE = args.encode_batch_size or TRAIN_BATCH_SIZE
    TRAIN_N_EPOCHS = args.epochs
    EARLY_STOPPING_PATIENCE = args.early_stopping_patience
    EARLY_STOPPING_MIN_DELTA = args.early_stopping_min_delta

    # Sanitize dataset name for filename
    dataset_name = args.dataset.replace("/", "__")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Neural Emulator Training")
    print("=" * 60)
    print(f"Game profile: {GAME_CONFIG.name}")
    print(f"Dataset: {args.dataset}")
    print(f"Dynamics mode: {args.dynamics_mode}")
    print(f"Device: {device}")
    print(f"Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Latent dim: {MODEL_LATENT_DIM}")
    print(f"Actions: {N_ACTIONS}")
    print(f"History length: {HISTORY_LENGTH}")
    print(f"Epochs per phase: {TRAIN_N_EPOCHS}")
    print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"Early stopping min delta: {EARLY_STOPPING_MIN_DELTA}")
    print(f"Batch size: {TRAIN_BATCH_SIZE}")
    print(f"Learning rate: {TRAIN_LEARNING_RATE}")
    print(f"Encode batch size: {ENCODE_BATCH_SIZE}")
    print(f"Sequence stride: {args.sequence_stride}")
    print(f"Pixel unroll steps: {args.pixel_unroll_steps}")
    print(f"Pixel feedback mode: {args.pixel_feedback_mode}")
    print(f"Pixel refine blocks: {args.pixel_refine_blocks}")
    print(f"Pixel dynamics resume path: {args.pixel_dynamics_path}")
    print(f"Spatial latent channels: {args.spatial_latent_channels}")
    print(f"Spatial refine blocks: {args.spatial_refine_blocks}")
    print(f"Spatial dynamics resume path: {args.spatial_dynamics_path}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    tracker = TrainingTracker()

    # Load and preprocess data
    if args.dynamics_mode in {"pixel", "spatial"}:
        train_sequences, val_sequences, dataset_stats = load_pixel_rollout_dataset(
            args.dataset,
            unroll_steps=args.pixel_unroll_steps,
            sequence_stride=args.sequence_stride,
        )
    else:
        train_sequences, val_sequences, dataset_stats = load_and_preprocess_dataset(args.dataset)

        if args.sequence_stride > 1:
            train_sequences = train_sequences[:: args.sequence_stride]
            val_sequences = val_sequences[:: args.sequence_stride]
            dataset_stats["data/train_sequences"] = len(train_sequences)
            dataset_stats["data/val_sequences"] = len(val_sequences)
            dataset_stats.update(sequence_action_fraction_metrics(train_sequences, val_sequences))
            print(
                "After sequence stride "
                f"{args.sequence_stride}: train={len(train_sequences)}, val={len(val_sequences)}"
            )

    if len(train_sequences) == 0:
        print("Error: No training sequences found!")
        sys.exit(1)

    if args.dynamics_mode in {"pixel", "spatial"}:
        tracker.init_run(
            args,
            {
                **dataset_stats,
                "run/device_type": device.type,
                "run/image_width": IMAGE_WIDTH,
                "run/image_height": IMAGE_HEIGHT,
                "run/history_length": HISTORY_LENGTH,
            },
        )
        tracker.log_run_metrics(dataset_stats)

        if args.skip_dynamics:
            print(f"\nSkipping {args.dynamics_mode} dynamics model training")
            frame_result = None
        elif args.dynamics_mode == "pixel":
            frame_result = train_pixel_dynamics_phase(
                train_sequences,
                val_sequences,
                TRAIN_N_EPOCHS,
                args.output_dir,
                dataset_name,
                EARLY_STOPPING_PATIENCE,
                EARLY_STOPPING_MIN_DELTA,
                pixel_dynamics_path=args.pixel_dynamics_path,
                pixel_feedback_mode=args.pixel_feedback_mode,
                pixel_refine_blocks=args.pixel_refine_blocks,
                tracker=tracker,
            )
        else:
            frame_result = train_spatial_dynamics_phase(
                train_sequences,
                val_sequences,
                TRAIN_N_EPOCHS,
                args.output_dir,
                dataset_name,
                EARLY_STOPPING_PATIENCE,
                EARLY_STOPPING_MIN_DELTA,
                spatial_dynamics_path=args.spatial_dynamics_path,
                pixel_feedback_mode=args.pixel_feedback_mode,
                spatial_latent_channels=args.spatial_latent_channels,
                spatial_refine_blocks=args.spatial_refine_blocks,
                tracker=tracker,
            )
    else:
        train_frames = extract_unique_frames(train_sequences)
        val_frames = extract_unique_frames(val_sequences)

        print(f"Unique autoencoder train frames: {len(train_frames)}")
        print(f"Unique autoencoder val frames: {len(val_frames)}")

        run_stats = {
            **dataset_stats,
            "run/device_type": device.type,
            "run/image_width": IMAGE_WIDTH,
            "run/image_height": IMAGE_HEIGHT,
            "run/history_length": HISTORY_LENGTH,
            "run/unique_autoencoder_train_frames": len(train_frames),
            "run/unique_autoencoder_val_frames": len(val_frames),
        }
        tracker.init_run(args, run_stats)
        tracker.log_run_metrics(
            {
                **dataset_stats,
                "run/unique_autoencoder_train_frames": len(train_frames),
                "run/unique_autoencoder_val_frames": len(val_frames),
            }
        )

        train_dataset = FrameDataset(train_frames)
        val_dataset = FrameDataset(val_frames)

        train_loader = DataLoader(
            train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=0
        )

        autoencoder_path = None
        autoencoder_result = None
        dynamics_result = None
        if not args.skip_autoencoder:
            autoencoder_result = train_autoencoder_phase(
                train_loader,
                val_loader,
                TRAIN_N_EPOCHS,
                args.output_dir,
                dataset_name,
                EARLY_STOPPING_PATIENCE,
                EARLY_STOPPING_MIN_DELTA,
                tracker=tracker,
            )
            autoencoder_path = autoencoder_result["best_model_path"]
        else:
            if args.autoencoder_path:
                autoencoder_path = args.autoencoder_path
            else:
                autoencoder_path = os.path.join(
                    args.output_dir, f"{dataset_name}-representation.pt"
                )
                if not os.path.exists(autoencoder_path):
                    print(f"Error: Autoencoder not found at {autoencoder_path}")
                    print("Either train the autoencoder or provide --autoencoder-path")
                    sys.exit(1)
            print(f"\nSkipping autoencoder training. Using: {autoencoder_path}")

        if not args.skip_dynamics:
            dynamics_result = train_dynamics_phase(
                autoencoder_path,
                train_sequences,
                val_sequences,
                TRAIN_N_EPOCHS,
                args.output_dir,
                dataset_name,
                EARLY_STOPPING_PATIENCE,
                EARLY_STOPPING_MIN_DELTA,
                tracker=tracker,
            )
        else:
            print("\nSkipping dynamics model training")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Models saved in: {args.output_dir}")
    print("Run the emulator with:")
    print(
        "  python main.py "
        f"--dataset {args.dataset} --game {GAME_CONFIG.name} "
        f"--use-local-models --models-dir {args.output_dir} "
        f"--dynamics-mode {args.dynamics_mode} "
        f"--history-length {HISTORY_LENGTH} "
        f"--image-width {IMAGE_WIDTH} --image-height {IMAGE_HEIGHT}"
    )
    print("=" * 60)

    summary = {
        "run/dataset": args.dataset,
        "run/dynamics_mode": args.dynamics_mode,
        "run/output_dir": args.output_dir,
    }
    if args.dynamics_mode == "pixel":
        if not args.skip_dynamics and frame_result is not None:
            summary["pixel_dynamics/best_val_loss"] = frame_result["best_val_loss"]
            summary["pixel_dynamics/best_epoch"] = frame_result["best_epoch"]
            summary["pixel_dynamics/best_model_path"] = frame_result["best_model_path"]
    elif args.dynamics_mode == "spatial":
        if not args.skip_dynamics and frame_result is not None:
            summary["spatial_dynamics/best_val_loss"] = frame_result["best_val_loss"]
            summary["spatial_dynamics/best_epoch"] = frame_result["best_epoch"]
            summary["spatial_dynamics/best_model_path"] = frame_result["best_model_path"]
    else:
        if autoencoder_result is not None:
            summary["autoencoder/best_val_loss"] = autoencoder_result["best_val_loss"]
            summary["autoencoder/best_epoch"] = autoencoder_result["best_epoch"]
            summary["autoencoder/best_model_path"] = autoencoder_result["best_model_path"]
        if dynamics_result is not None:
            summary["latent_dynamics/best_val_loss"] = dynamics_result["best_val_loss"]
            summary["latent_dynamics/best_epoch"] = dynamics_result["best_epoch"]
            summary["latent_dynamics/best_model_path"] = dynamics_result["best_model_path"]
    tracker.finish(summary)


if __name__ == "__main__":
    main()
