"""
Training script for neural emulator models.
Trains ConvAutoencoder and DynamicsModel on Hugging Face datasets.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm import tqdm

from device_utils import configure_torch, get_device, move_batch_tensor, prepare_conv_module
from game_config import BREAKOUT_CONFIG, infer_game_config
from preprocessing import encode_action, has_valid_black_background, preprocess_frame

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

# Training hyperparameters
TRAIN_N_EPOCHS = 50
TRAIN_BATCH_SIZE = 64
TRAIN_LEARNING_RATE = 0.001
TRAIN_MAX_GRAD_NORM = 0
TRAIN_WEIGHT_DECAY = 0
ENCODE_BATCH_SIZE = 64

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
            self._flattened_size = dummy_output.view(1, -1).shape[1]
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
            x = x.view(x.size(0), -1)
            x = self.fc_enc(x)
        return x

    def decode(self, z):
        """Decode latent vector to image."""
        if self.use_bottleneck:
            z = self.fc_dec(z)
            z = z.view(z.size(0), *self._conv_output_shape)
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

    def __init__(self, z_dim=MODEL_LATENT_DIM, n_actions=N_ACTIONS):
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
        """Predict delta in latent space."""
        return self.net(z_and_a)


# =============================================================================
# Data Loading and Preprocessing
# =============================================================================


def load_and_preprocess_dataset(dataset_id, val_split=VAL_SPLIT_RATIO):
    """
    Load dataset from Hugging Face and preprocess.

    Returns:
        train_sequences: list of (frame_t, action_t, frame_t+1) for training
        val_sequences: list of (frame_t, action_t, frame_t+1) for validation
    """
    print(f"\nLoading dataset: {dataset_id}")

    try:
        dataset = load_dataset(dataset_id, split="train")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Attempting to load with trust_remote_code=True...")
        dataset = load_dataset(dataset_id, split="train", trust_remote_code=True)

    print(f"Dataset loaded: {len(dataset)} samples")

    # Group samples by episode
    episodes = {}
    for i, sample in enumerate(dataset):
        episode_id = sample.get("episode_id", 0)
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

    for episode_id, samples in episodes.items():
        # Sort by original index
        samples.sort(key=lambda x: x[0])

        # Create consecutive pairs
        for i in range(len(samples) - 1):
            _, sample_t = samples[i]
            _, sample_t1 = samples[i + 1]

            if sample_t.get("terminations") or sample_t.get("truncations"):
                skipped_terminal_pairs += 1
                continue

            # Extract data
            frame_t = sample_t["observations"]
            action_t = sample_t["actions"]
            frame_t1 = sample_t1["observations"]

            if not has_valid_black_background(frame_t, GAME_CONFIG):
                skipped_invalid_frames += 1
                continue
            if not has_valid_black_background(frame_t1, GAME_CONFIG):
                skipped_invalid_frames += 1
                continue

            frame_t_processed = preprocess_frame(
                frame_t, GAME_CONFIG, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)
            )
            frame_t1_processed = preprocess_frame(
                frame_t1, GAME_CONFIG, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)
            )

            try:
                action_array = encode_action(action_t, GAME_CONFIG)
            except ValueError as exc:
                print(f"Warning: Skipping sample {i} due to invalid action: {exc}")
                continue

            if len(episode_ids) > 1:
                target_list = (
                    val_sequences if episode_id in val_episode_ids else train_sequences
                )
            else:
                if i < int((len(samples) - 1) * (1.0 - val_split)):
                    target_list = train_sequences
                else:
                    target_list = val_sequences

            target_list.append((frame_t_processed, action_array, frame_t1_processed))

    print(f"Train sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    print(f"Skipped invalid-background pairs: {skipped_invalid_frames}")
    print(f"Skipped terminated/truncated pairs: {skipped_terminal_pairs}")

    return train_sequences, val_sequences


class SequenceDataset(Dataset):
    """PyTorch Dataset for (frame, action, next_frame) sequences."""

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        frame_t, action_t, frame_t1 = self.sequences[idx]

        # Convert to tensors
        # Frame: (H, W) -> (1, H, W) -> tensor
        frame_t_tensor = torch.from_numpy(frame_t).unsqueeze(0).float()
        frame_t1_tensor = torch.from_numpy(frame_t1).unsqueeze(0).float()
        action_t_tensor = torch.from_numpy(action_t).float()

        return frame_t_tensor, action_t_tensor, frame_t1_tensor


# =============================================================================
# Training Functions
# =============================================================================


def train_autoencoder_phase(
    train_loader, val_loader, n_epochs, output_dir, dataset_name
):
    """Phase 1: Train the ConvAutoencoder on frame reconstruction."""

    print("\n" + "=" * 60)
    print("PHASE 1: Training Autoencoder")
    print("=" * 60)

    model = prepare_conv_module(ConvAutoencoder(latent_dim=MODEL_LATENT_DIM), device)

    # Use L1 loss as specified in config
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=TRAIN_LEARNING_RATE, weight_decay=TRAIN_WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    best_model_path = None

    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = torch.zeros((), device=device)
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Train]")
        for frame_t, _, _ in pbar:
            frame_t = move_batch_tensor(
                frame_t, device, non_blocking=device.type == "cuda"
            )

            optimizer.zero_grad(set_to_none=True)

            # Reconstruct frame_t
            recon, z = model(frame_t)
            loss = criterion(recon, frame_t)

            loss.backward()

            if TRAIN_MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_MAX_GRAD_NORM)

            optimizer.step()

            train_loss += loss.detach()
            n_batches += 1
            if n_batches == 1 or n_batches % 20 == 0:
                pbar.set_postfix({"loss": f"{loss.detach().cpu().item():.6f}"})

        avg_train_loss = (train_loss / n_batches).detach().cpu().item()

        # Validation
        model.eval()
        val_loss = torch.zeros((), device=device)
        n_val_batches = 0

        with torch.inference_mode():
            for frame_t, _, _ in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Val]", leave=False
            ):
                frame_t = move_batch_tensor(
                    frame_t, device, non_blocking=device.type == "cuda"
                )

                recon, z = model(frame_t)
                loss = criterion(recon, frame_t)

                val_loss += loss
                n_val_batches += 1

        avg_val_loss = (val_loss / n_val_batches).detach().cpu().item()

        print(
            f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(
                output_dir, f"{dataset_name}-representation.pt"
            )
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> Saved best model (val_loss: {best_val_loss:.6f})")

        empty_device_cache()

    print(f"\nPhase 1 complete. Best model saved to: {best_model_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return best_model_path


def encode_frames(model, sequences, batch_size=ENCODE_BATCH_SIZE):
    """
    Encode all frames to latent vectors using the trained autoencoder.

    Returns:
        List of (latent_t, action_t, latent_t1) tuples
    """
    model.eval()

    # Extract all unique frames
    all_frames = []
    frame_to_idx = {}

    for frame_t, action_t, frame_t1 in sequences:
        # Add frame_t
        frame_t_tuple = tuple(frame_t.flatten())
        if frame_t_tuple not in frame_to_idx:
            frame_to_idx[frame_t_tuple] = len(all_frames)
            all_frames.append(frame_t)

        # Add frame_t1
        frame_t1_tuple = tuple(frame_t1.flatten())
        if frame_t1_tuple not in frame_to_idx:
            frame_to_idx[frame_t1_tuple] = len(all_frames)
            all_frames.append(frame_t1)

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

    for frame_t, action_t, frame_t1 in sequences:
        latent_t_idx = frame_to_idx[tuple(frame_t.flatten())]
        latent_t1_idx = frame_to_idx[tuple(frame_t1.flatten())]

        latent_t = all_latents[latent_t_idx]
        latent_t1 = all_latents[latent_t1_idx]

        latent_sequences.append((latent_t, action_t, latent_t1))

    return latent_sequences


class LatentDataset(Dataset):
    """PyTorch Dataset for (latent, action, next_latent) sequences."""

    def __init__(self, latent_sequences):
        self.latent_sequences = latent_sequences

    def __len__(self):
        return len(self.latent_sequences)

    def __getitem__(self, idx):
        latent_t, action_t, latent_t1 = self.latent_sequences[idx]

        latent_t_tensor = torch.from_numpy(latent_t).float()
        action_t_tensor = torch.from_numpy(action_t).float()
        latent_t1_tensor = torch.from_numpy(latent_t1).float()

        return latent_t_tensor, action_t_tensor, latent_t1_tensor


def train_dynamics_phase(
    autoencoder_path, train_sequences, val_sequences, n_epochs, output_dir, dataset_name
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
        DynamicsModel(z_dim=MODEL_LATENT_DIM, n_actions=N_ACTIONS),
        device,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        dynamics_model.parameters(),
        lr=TRAIN_LEARNING_RATE,
        weight_decay=TRAIN_WEIGHT_DECAY,
    )

    best_val_loss = float("inf")
    best_model_path = None

    for epoch in range(n_epochs):
        # Training
        dynamics_model.train()
        train_loss = torch.zeros((), device=device)
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Train]")
        for latent_t, action_t, latent_t1 in pbar:
            latent_t = latent_t.to(device, non_blocking=device.type == "cuda")
            action_t = action_t.to(device, non_blocking=device.type == "cuda")
            latent_t1 = latent_t1.to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad(set_to_none=True)

            # Concatenate latent and action
            z_and_a = torch.cat([latent_t, action_t], dim=1)

            # Predict delta
            delta_pred = dynamics_model(z_and_a)

            # Compute target delta
            delta_target = latent_t1 - latent_t

            # Loss on delta prediction
            loss = criterion(delta_pred, delta_target)

            loss.backward()

            if TRAIN_MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    dynamics_model.parameters(), TRAIN_MAX_GRAD_NORM
                )

            optimizer.step()

            train_loss += loss.detach()
            n_batches += 1
            if n_batches == 1 or n_batches % 20 == 0:
                pbar.set_postfix({"loss": f"{loss.detach().cpu().item():.6f}"})

        avg_train_loss = (train_loss / n_batches).detach().cpu().item()

        # Validation (open-loop: use ground truth latent_t each time)
        dynamics_model.eval()
        val_loss = torch.zeros((), device=device)
        n_val_batches = 0

        with torch.inference_mode():
            for latent_t, action_t, latent_t1 in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Val]", leave=False
            ):
                latent_t = latent_t.to(device, non_blocking=device.type == "cuda")
                action_t = action_t.to(device, non_blocking=device.type == "cuda")
                latent_t1 = latent_t1.to(device, non_blocking=device.type == "cuda")

                z_and_a = torch.cat([latent_t, action_t], dim=1)
                delta_pred = dynamics_model(z_and_a)

                delta_target = latent_t1 - latent_t
                loss = criterion(delta_pred, delta_target)

                val_loss += loss
                n_val_batches += 1

        avg_val_loss = (val_loss / n_val_batches).detach().cpu().item()

        print(
            f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(output_dir, f"{dataset_name}-dynamics.pt")
            torch.save(dynamics_model.state_dict(), best_model_path)
            print(f"  -> Saved best model (val_loss: {best_val_loss:.6f})")

        empty_device_cache()

    print(f"\nPhase 2 complete. Best model saved to: {best_model_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return best_model_path


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    # Declare globals at the very beginning of the function
    global GAME_CONFIG, MODEL_LATENT_DIM, IMAGE_HEIGHT, IMAGE_WIDTH, TRAIN_BATCH_SIZE
    global TRAIN_N_EPOCHS, N_ACTIONS, ENCODE_BATCH_SIZE

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
        "--epochs", type=int, default=TRAIN_N_EPOCHS, help="Number of epochs per phase"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training; defaults to an MPS-safe value on Apple Silicon",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=MODEL_LATENT_DIM, help="Latent dimension size"
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

    args = parser.parse_args()

    GAME_CONFIG = infer_game_config(dataset_id=args.dataset, game=args.game)
    N_ACTIONS = GAME_CONFIG.n_actions

    # Update global config from args
    MODEL_LATENT_DIM = args.latent_dim
    if args.image_size is not None:
        IMAGE_WIDTH = args.image_size
        IMAGE_HEIGHT = args.image_size
    else:
        IMAGE_WIDTH = args.image_width
        IMAGE_HEIGHT = args.image_height
    TRAIN_BATCH_SIZE = args.batch_size or default_batch_size_for_device(device)
    ENCODE_BATCH_SIZE = args.encode_batch_size or TRAIN_BATCH_SIZE
    TRAIN_N_EPOCHS = args.epochs

    # Sanitize dataset name for filename
    dataset_name = args.dataset.replace("/", "__")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Neural Emulator Training")
    print("=" * 60)
    print(f"Game profile: {GAME_CONFIG.name}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Latent dim: {MODEL_LATENT_DIM}")
    print(f"Actions: {N_ACTIONS}")
    print(f"Epochs per phase: {TRAIN_N_EPOCHS}")
    print(f"Batch size: {TRAIN_BATCH_SIZE}")
    print(f"Encode batch size: {ENCODE_BATCH_SIZE}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Load and preprocess data
    train_sequences, val_sequences = load_and_preprocess_dataset(args.dataset)

    if len(train_sequences) == 0:
        print("Error: No training sequences found!")
        sys.exit(1)

    # Create data loaders
    train_dataset = SequenceDataset(train_sequences)
    val_dataset = SequenceDataset(val_sequences)

    train_loader = DataLoader(
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Phase 1: Train Autoencoder
    autoencoder_path = None
    if not args.skip_autoencoder:
        autoencoder_path = train_autoencoder_phase(
            train_loader, val_loader, TRAIN_N_EPOCHS, args.output_dir, dataset_name
        )
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

    # Phase 2: Train Dynamics Model
    if not args.skip_dynamics:
        dynamics_path = train_dynamics_phase(
            autoencoder_path,
            train_sequences,
            val_sequences,
            TRAIN_N_EPOCHS,
            args.output_dir,
            dataset_name,
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
        f"--image-width {IMAGE_WIDTH} --image-height {IMAGE_HEIGHT}"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
