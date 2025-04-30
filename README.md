# 🕹️ Gymnasium Emulator

🔹 **A neural emulator for visualizing and interacting with latent dynamics of retro games using pre-trained models.**

## 📖 Overview

Gymnasium Emulator is a tool for exploring the latent state space of classic video games, such as Tetris for Game Boy, using deep learning models. It leverages pre-trained autoencoder and dynamics models to reconstruct and predict game frames in real time based on user keyboard input. The emulator provides a visual interface powered by Pygame, allowing you to interact with the learned game dynamics without a traditional emulator or ROM.

The project downloads pre-trained models from Hugging Face, encodes an initial game frame, and lets you step through the game's latent space by pressing keys corresponding to game controls. Each action updates the latent state and displays the predicted next frame.

## 🚀 Installation

1. Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.
2. Clone this repository and navigate to its directory.
3. Run:

   ```bash
   source activate-env.sh
   ```

   This will create and activate the `gymnasium-emulator` Conda environment with all dependencies.

4. Copy `.env.example` to `.env` and add your Hugging Face API token:

   ```bash
   cp .env.example .env
   # Edit .env and set HF_TOKEN=your-api-token
   ```

## 🛠️ Usage

1. Ensure the environment is activated:

   ```bash
   conda activate gymnasium-emulator
   ```

2. Run the emulator:

   ```bash
   python main.py
   ```

3. Use the following keys to interact:

   - **Z**: A button
   - **X**: B button
   - **Q**: Select
   - **R**: Start
   - **Arrow keys**: Up, Down, Left, Right

   The emulator will display the reconstructed game frame and respond to your inputs in real time.

## 📄 License

This project is licensed under the [MIT License](LICENSE).