# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Gymnasium Emulator is a neural emulator that visualizes and interacts with latent dynamics of retro games using pre-trained deep learning models. It allows exploration of game state space through learned representations rather than traditional emulation.

The system uses:
- **Autoencoder (ConvAutoencoder)**: Encodes game frames into a 32-dimensional latent space and decodes back to images
- **Dynamics Model**: Predicts how the latent state changes given an action (9 discrete actions: A, B, SELECT, START, UP, DOWN, LEFT, RIGHT, B-alt)
- **Pygame Interface**: Real-time visualization and keyboard control

Models are downloaded from Hugging Face repositories at runtime.

## Environment Setup

**Create venv and install dependencies:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu118 -e .
```

**Environment configuration:**
- Python 3.11+
- PyTorch with CUDA 11.8 support
- Dependencies defined in `pyproject.toml`

**Required setup:**
```bash
cp .env.example .env
# Edit .env and add your HF_TOKEN
```

## Running the Emulator

```bash
python main.py
```

**Controls:**
- Z: A button
- X: B button
- Q: SELECT
- R: START
- Arrow keys: directional input

## Code Architecture

### Model Architecture

**ConvAutoencoder** (main.py:29-82)
- Encoder: 3-layer conv network (1→32→64→128 channels)
- Bottleneck: Optional fully-connected layer to compress to latent_dim
- Decoder: 3-layer transposed conv (128→64→32→1 channels)
- Key methods: `encode()`, `decode()`, `forward()`
- Training mode supports latent noise injection

**DynamicsModel** (main.py:84-99)
- Input: concatenated latent vector + one-hot action (z_dim + 9)
- Architecture: LayerNorm → 128 → 128 → z_dim (GELU activations)
- Output: **delta in latent space** (not next state directly)
- Orthogonal weight initialization for stability

### Inference Loop

The main loop (main.py:155-200) implements latent space stepping:

1. Capture keyboard input → action vector (9-dim one-hot)
2. Concatenate current latent with action
3. Predict delta: `delta_latent = dynamics_model(z_and_a)`
4. Update latent: `next_latent = latent + delta_latent` (residual connection)
5. Decode to image: `recon = representation_model.decode(next_latent)`
6. Display frame at 30 FPS

**Critical implementation detail:** The dynamics model predicts the **change** in latent space, not the absolute next state. Always use `latent + delta_latent`.

### Model Configuration

Hyperparameters are defined at the top of main.py (lines 8-21):
- `model_latent_dim = 32`: Latent space dimensionality
- `ds_id`: Hugging Face dataset/model identifier
- `model_compile = True`: Uses torch.compile() for optimization
- Image dimensions: 80×144 grayscale

Models are automatically downloaded from:
- `{ds_id}-representation`: Autoencoder
- `{ds_id}-dynamics`: Dynamics predictor

### Initial State

The emulator loads `start.png` as the initial frame, encodes it to get the starting latent vector, then allows interactive stepping through latent space.

## Important Instructions

- **README.md must be kept up to date** with any significant project changes
- When modifying model architecture, ensure input/output dimensions remain compatible
- GPU (CUDA) is required for inference
- Model downloads require valid HF_TOKEN in .env
