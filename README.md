# 🕹️ gymnasium-emulator

<p align="center">
  <img src="logo.png" alt="Logo" width="400"/>
</p>

🔹 **A neural emulator for visualizing and interacting with latent dynamics of retro games using pre-trained models.**

## 📖 Overview

Gymnasium Emulator lets you explore the latent state space of classic video games—like Tetris for Game Boy—using deep learning models. It leverages pre-trained autoencoder and dynamics models to reconstruct and predict game frames in real time based on your keyboard input. With a simple Pygame interface, you can interact with learned game dynamics without needing a traditional emulator or ROM.

The emulator downloads models from Hugging Face, encodes an initial game frame, and allows you to step through the game's latent space by pressing keys mapped to game controls. Each action updates the latent state and displays the predicted next frame.

## 🚀 Installation

1. Ensure [Miniconda](https://docs.conda.io/en/latest/miniconda.html) is installed.
2. Clone this repository and navigate to its directory.
3. Run:

   ```bash
   source activate-env.sh
   ```

   This will create and activate the `gymnasium-emulator` Conda environment with all dependencies.

4. Copy the example environment file and add your Hugging Face API token:

   ```bash
   cp .env.example .env
   # Edit .env and set HF_TOKEN=your-api-token
   ```

## 🛠️ Usage

1. Make sure the environment is activated:

   ```bash
   conda activate gymnasium-emulator
   ```

2. Start the emulator:

   ```bash
   python main.py
   ```

3. Control the emulator using these keys:

   - **Z**: A button
   - **X**: B button
   - **Q**: Select
   - **R**: Start
   - **Arrow keys**: Up, Down, Left, Right

   The emulator displays the reconstructed game frame and responds to your inputs in real time.

## 📄 License

This project is licensed under the [MIT License](LICENSE).