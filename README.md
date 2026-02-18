<div align="center">
  <img src="logo.png" alt="gymemu" width="512"/>

  # gymemu

  [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
  [![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg)](https://python.org)
  [![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_11.8-EE4C2C.svg)](https://pytorch.org)
  [![Hugging Face](https://img.shields.io/badge/Models-Hugging_Face-FFD21E.svg)](https://huggingface.co/tsilva)

  **🎮 Play retro games through learned latent dynamics—no ROM required 🧠**

  [How It Works](#how-it-works) · [Quick Start](#quick-start) · [Controls](#controls)
</div>

---

## Overview

Gymnasium Emulator visualizes and interacts with the latent dynamics of retro games using pre-trained deep learning models. Instead of traditional emulation, it uses a convolutional autoencoder to encode game frames into a 32-dimensional latent space and a dynamics model to predict how that state changes with each action.

The result: real-time gameplay powered entirely by neural networks.

## How It Works

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Keyboard   │ ──▶ │   Dynamics   │ ──▶ │   Decoder   │ ──▶ Display
│   Input     │     │    Model     │     │  (latent→   │
│ (9 actions) │     │ (Δ latent)   │     │   frame)    │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                    latent + Δlatent
                           │
                    ┌──────▼──────┐
                    │   Current   │
                    │   Latent    │
                    │   State     │
                    └─────────────┘
```

- **Autoencoder**: 3-layer convolutional network compresses 80×144 grayscale frames to 32 dimensions
- **Dynamics Model**: Predicts the *change* in latent space given an action (residual connection)
- **30 FPS**: Real-time visualization through Pygame

Models are downloaded automatically from Hugging Face at runtime.

## Quick Start

**Prerequisites**: Python 3.11+, NVIDIA GPU with CUDA support

```bash
# Clone and setup
git clone https://github.com/tsilva/gymemu.git
cd gymemu

# Create environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cu118 -e .

# Configure Hugging Face token
cp .env.example .env
# Edit .env and add: HF_TOKEN=your-token

# Run
python main.py
```

## Controls

| Key | Action |
|-----|--------|
| `Z` | A button |
| `X` | B button |
| `Q` | SELECT |
| `R` | START |
| `↑` `↓` `←` `→` | D-pad |

## Requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.11 |
| GPU | NVIDIA with CUDA 11.8+ |
| RAM | 8GB+ recommended |
| Dependencies | PyTorch, Pygame, PIL, NumPy |

## Project Structure

```
gymemu/
├── main.py           # Neural emulator with model definitions and game loop
├── start.png         # Initial game frame (Tetris title screen)
├── pyproject.toml    # Project metadata and dependencies
└── .env.example      # Template for Hugging Face credentials
```

## License

[MIT](LICENSE) © 2025 Tiago Silva