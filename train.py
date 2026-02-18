"""
Training script for neural emulator models.
Trains ConvAutoencoder and DynamicsModel on Hugging Face datasets.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# =============================================================================
# Configuration (must match main.py for model compatibility)
# =============================================================================

SEED = 42
MODEL_LATENT_DIM = 32
MODEL_LATENT_NOISE_FACTOR = 0.0

# Image dimensions (80x80 square)
IMAGE_CHANNELS = 1
IMAGE_HEIGHT = 80
IMAGE_WIDTH = 80

# Training hyperparameters
TRAIN_N_EPOCHS = 50
TRAIN_BATCH_SIZE = 128
TRAIN_LEARNING_RATE = 0.001
TRAIN_MAX_GRAD_NORM = 0
TRAIN_WEIGHT_DECAY = 0

N_ACTIONS = 9

USE_BOTTLENECK = True
VAL_SPLIT_RATIO = 0.2

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


def preprocess_image(image, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT)):
    """
    Preprocess image for training.

    Args:
        image: PIL Image or numpy array
        target_size: (width, height) tuple

    Returns:
        Preprocessed image as numpy array (H, W), normalized to [0, 1]
    """
    if isinstance(image, np.ndarray):
        # Convert numpy to PIL
        if image.ndim == 3 and image.shape[2] in [3, 4]:
            # RGB or RGBA
            pil_img = Image.fromarray(image.astype(np.uint8))
        else:
            # Already grayscale or single channel
            pil_img = Image.fromarray(image.squeeze().astype(np.uint8), mode="L")
    else:
        pil_img = image

    # Convert to grayscale if needed
    if pil_img.mode != "L":
        pil_img = pil_img.convert("L")

    # Resize to target size
    pil_img = pil_img.resize(target_size, Image.BILINEAR)

    # Convert to numpy and normalize
    img_array = np.array(pil_img, dtype=np.float32) / 255.0

    return img_array


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

    n_val_episodes = max(1, int(len(episode_ids) * val_split))
    val_episode_ids = set(episode_ids[:n_val_episodes])
    train_episode_ids = set(episode_ids[n_val_episodes:])

    print(
        f"Train episodes: {len(train_episode_ids)}, Val episodes: {len(val_episode_ids)}"
    )

    # Create sequences from episodes
    train_sequences = []
    val_sequences = []

    for episode_id, samples in episodes.items():
        # Sort by original index
        samples.sort(key=lambda x: x[0])

        target_list = (
            val_sequences if episode_id in val_episode_ids else train_sequences
        )

        # Create consecutive pairs
        for i in range(len(samples) - 1):
            _, sample_t = samples[i]
            _, sample_t1 = samples[i + 1]

            # Extract data
            frame_t = sample_t["observations"]
            action_t = sample_t["actions"]  # Multibinary: list of 9 values
            frame_t1 = sample_t1["observations"]

            # Preprocess frames
            frame_t_processed = preprocess_image(frame_t)
            frame_t1_processed = preprocess_image(frame_t1)

            # Convert action to numpy array (multibinary)
            action_array = np.array(action_t, dtype=np.float32)
            if len(action_array) != N_ACTIONS:
                print(
                    f"Warning: Action dimension mismatch. Expected {N_ACTIONS}, got {len(action_array)}"
                )
                # Pad or truncate to match
                if len(action_array) < N_ACTIONS:
                    action_array = np.pad(
                        action_array, (0, N_ACTIONS - len(action_array))
                    )
                else:
                    action_array = action_array[:N_ACTIONS]

            target_list.append((frame_t_processed, action_array, frame_t1_processed))

    print(f"Train sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")

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

    model = ConvAutoencoder(latent_dim=MODEL_LATENT_DIM).to(device)

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
        train_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Train]")
        for frame_t, _, frame_t1 in pbar:
            frame_t = frame_t.to(device)

            optimizer.zero_grad()

            # Reconstruct frame_t
            recon, z = model(frame_t)
            loss = criterion(recon, frame_t)

            loss.backward()

            if TRAIN_MAX_GRAD_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_MAX_GRAD_NORM)

            optimizer.step()

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / n_batches

        # Validation
        model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for frame_t, _, frame_t1 in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Val]", leave=False
            ):
                frame_t = frame_t.to(device)

                recon, z = model(frame_t)
                loss = criterion(recon, frame_t)

                val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / n_val_batches

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

    print(f"\nPhase 1 complete. Best model saved to: {best_model_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return best_model_path


def encode_frames(model, sequences, batch_size=TRAIN_BATCH_SIZE):
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

    with torch.no_grad():
        for i in range(0, len(all_frames), batch_size):
            batch_frames = all_frames[i : i + batch_size]
            batch_tensor = torch.stack(
                [torch.from_numpy(f).unsqueeze(0) for f in batch_frames]
            ).to(device)

            batch_latents = model.encode(batch_tensor)
            all_latents.extend(batch_latents.cpu().numpy())

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
    autoencoder = ConvAutoencoder(latent_dim=MODEL_LATENT_DIM).to(device)
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
        train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Initialize dynamics model
    dynamics_model = DynamicsModel(z_dim=MODEL_LATENT_DIM, n_actions=N_ACTIONS).to(
        device
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
        train_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Train]")
        for latent_t, action_t, latent_t1 in pbar:
            latent_t = latent_t.to(device)
            action_t = action_t.to(device)
            latent_t1 = latent_t1.to(device)

            optimizer.zero_grad()

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

            train_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / n_batches

        # Validation (open-loop: use ground truth latent_t each time)
        dynamics_model.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for latent_t, action_t, latent_t1 in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{n_epochs} [Val]", leave=False
            ):
                latent_t = latent_t.to(device)
                action_t = action_t.to(device)
                latent_t1 = latent_t1.to(device)

                z_and_a = torch.cat([latent_t, action_t], dim=1)
                delta_pred = dynamics_model(z_and_a)

                delta_target = latent_t1 - latent_t
                loss = criterion(delta_pred, delta_target)

                val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss / n_val_batches

        print(
            f"Epoch {epoch + 1}/{n_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(output_dir, f"{dataset_name}-dynamics.pt")
            torch.save(dynamics_model.state_dict(), best_model_path)
            print(f"  -> Saved best model (val_loss: {best_val_loss:.6f})")

    print(f"\nPhase 2 complete. Best model saved to: {best_model_path}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    return best_model_path


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    # Declare globals at the very beginning of the function
    global MODEL_LATENT_DIM, IMAGE_HEIGHT, IMAGE_WIDTH, TRAIN_BATCH_SIZE, TRAIN_N_EPOCHS

    parser = argparse.ArgumentParser(description="Train neural emulator models")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Hugging Face dataset ID (e.g., 'tsilva/gymnasium-recorder__SuperMarioBros_Nes_v0')",
    )
    parser.add_argument(
        "--epochs", type=int, default=TRAIN_N_EPOCHS, help="Number of epochs per phase"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TRAIN_BATCH_SIZE,
        help="Batch size for training",
    )
    parser.add_argument(
        "--latent-dim", type=int, default=MODEL_LATENT_DIM, help="Latent dimension size"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_HEIGHT,
        help="Image size (assumes square, e.g., 80 for 80x80)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save trained models",
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

    # Update global config from args
    MODEL_LATENT_DIM = args.latent_dim
    IMAGE_HEIGHT = args.image_size
    IMAGE_WIDTH = args.image_size
    TRAIN_BATCH_SIZE = args.batch_size
    TRAIN_N_EPOCHS = args.epochs

    # Sanitize dataset name for filename
    dataset_name = args.dataset.replace("/", "__")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Neural Emulator Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Image size: {IMAGE_WIDTH}x{IMAGE_HEIGHT}")
    print(f"Latent dim: {MODEL_LATENT_DIM}")
    print(f"Epochs per phase: {TRAIN_N_EPOCHS}")
    print(f"Batch size: {TRAIN_BATCH_SIZE}")
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
    print(f"To use in main.py:")
    print(f"  1. Update image_height = {IMAGE_HEIGHT}, image_width = {IMAGE_WIDTH}")
    print(f"  2. Set ds_id = '{dataset_name}' (local models use sanitized name)")
    print(f"  3. Update main.py to load local models from ./models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
