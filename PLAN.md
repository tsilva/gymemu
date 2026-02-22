*updated*

# Roadmap: Retro Neural Emulator

Build an interactive neural emulator that replaces game engines with learned models. Progress from a single deterministic game to a multi-game, stochastic, high-fidelity system.

---

## Phase 1: Deterministic Baseline (NES Tetris)

Start with the simplest possible game to validate the full pipeline end-to-end.

### 1.1 Dataset Pipeline for NES Tetris

Ingest raw episode recordings into training-ready transition pairs.

**Tasks:**
- [ ] Collect NES Tetris gameplay recordings (frame + 8-button NES action per step)
- [ ] Upload dataset to Hugging Face with schema: `episode_id`, `observations` (image), `actions` (list of 8 binary values)
- [ ] Adapt `train.py` dataset loading to handle NES action space (8 buttons: A, B, SELECT, START, UP, DOWN, LEFT, RIGHT)
- [ ] Verify pair construction: `(frame_t, action_t, frame_t+1)` with episode-level train/val split (80/20)
- [ ] Preprocess: grayscale, resize to 80x80, normalize to [0,1]

**Success criteria:**
- `python train.py --dataset "your/nes-tetris" --epochs 1` runs without errors
- DataLoader yields batches of shape `(128, 1, 80, 80)` for frames and `(128, 8)` for actions
- Visual spot-check: 10 random pairs show correct temporal ordering (frame_t+1 is plausibly the next state after action_t)

### 1.2 Train Autoencoder on Tetris Frames

Learn a latent representation that can faithfully reconstruct game frames.

**Tasks:**
- [ ] Train ConvAutoencoder with L1 loss, latent_dim=32, ~50 epochs
- [ ] Generate reconstruction grid: 16 original frames vs their reconstructions side by side
- [ ] Plot training/validation loss curves, confirm convergence (val loss plateaus)
- [ ] Test edge cases: empty board, full board, piece in various positions, game over screen

**Success criteria:**
- Val L1 loss < 0.02
- Reconstructions are pixel-sharp -- individual blocks and score digits are clearly readable
- No visible artifacts on game-critical elements (piece shape, board grid lines, score)

### 1.3 Train Deterministic Dynamics Model

Predict latent-space deltas given current state + action.

**Tasks:**
- [ ] Encode full dataset to latent space using frozen autoencoder
- [ ] Train DynamicsModel (LayerNorm -> 128 -> 128 -> 32, GELU, orthogonal init)
- [ ] Loss: MSE on `delta_target = z_{t+1} - z_t`
- [ ] ~50 epochs, monitor val loss

**Success criteria:**
- Val MSE loss converges and stabilizes
- Single-step prediction test: encode 100 random frames, predict next frame, decode, compare to ground truth -- average L1 pixel error < 0.05
- "No action" test: feeding zero-action vector produces near-zero delta (piece falls by gravity only)

### 1.4 Interactive Emulator Loop

Wire up trained models into a playable real-time loop.

**Tasks:**
- [ ] Load trained autoencoder + dynamics model
- [ ] Bootstrap from a real Tetris start screen frame
- [ ] Map keyboard to NES action vector
- [ ] Run inference loop: `z_{t+1} = z_t + dynamics(z_t, action)` -> decode -> display at 30 FPS
- [ ] Add torch.compile() for GPU acceleration

**Success criteria:**
- Runs at stable 30 FPS on GPU
- Pressing LEFT/RIGHT visibly moves the piece in the correct direction
- Piece falls over time without input (gravity works)
- Emulator runs for 60+ seconds without visual degradation or artifacts

### 1.5 Quantitative Evaluation Harness

Build repeatable metrics so future improvements can be measured.

**Tasks:**
- [ ] Implement N-step rollout evaluation: run dynamics model for N steps from a real frame, compare to actual game sequence
- [ ] Metrics: L1 pixel error, SSIM, LPIPS (perceptual similarity) at steps 1, 5, 10, 30, 60
- [ ] Create a held-out test set of 20 episodes never seen during training
- [ ] Save baseline numbers to a results file

**Success criteria:**
- Evaluation script runs end-to-end and produces a table of metrics per rollout length
- 1-step SSIM > 0.90
- 10-step SSIM > 0.70
- Metrics are reproducible across runs (seeded)

---

## Phase 2: Sharper Reconstructions

The deterministic baseline will produce blurry outputs over time. Fix reconstruction quality first before tackling dynamics.

### 2.1 Perceptual Loss for Autoencoder

Replace pure L1 with a combination of L1 + perceptual (feature-matching) loss.

**Tasks:**
- [ ] Add a frozen VGG16 (or lightweight CNN) feature extractor
- [ ] Compute perceptual loss: L2 distance between features of original and reconstructed frames
- [ ] Combined loss: `L1 + 0.1 * perceptual`
- [ ] Retrain autoencoder, compare reconstructions to Phase 1 baseline

**Success criteria:**
- Sharper edges on game elements (blocks have crisp boundaries, not soft gradients)
- Score digits are individually distinguishable in reconstructions
- SSIM improvement of at least +0.02 over L1-only baseline

### 2.2 Increase Latent Capacity

Scale up the representation to handle more visual detail.

**Tasks:**
- [ ] Experiment with latent_dim: 32 -> 64 -> 128
- [ ] Add one more conv layer to encoder/decoder (4 layers total)
- [ ] Compare reconstruction quality at each latent_dim
- [ ] Pick the smallest latent_dim that achieves target quality (diminishing returns beyond some point)

**Success criteria:**
- Find the sweet spot: reconstruction quality plateaus (increasing latent_dim no longer helps)
- Inference still runs at 30 FPS with the chosen latent_dim
- Document chosen latent_dim and architecture in a config

### 2.3 Latent Noise Robustness

Make the autoencoder tolerant of slightly off-distribution latent vectors (which the dynamics model will produce).

**Tasks:**
- [ ] Enable latent noise during autoencoder training: add `N(0, noise_factor)` to z before decoding
- [ ] Sweep noise_factor: 0.01, 0.05, 0.1
- [ ] Retrain, measure reconstruction quality AND multi-step rollout quality

**Success criteria:**
- Reconstruction quality drops by no more than 5% vs noise-free
- 30-step rollout SSIM improves by at least +0.05 compared to no-noise baseline
- Visual quality no longer degrades noticeably during 60s of interactive play

---

## Phase 3: Stochastic Dynamics

The deterministic dynamics model averages over possible futures, causing blur. Replace it with a model that can sample from multiple plausible outcomes.

### 3.1 VAE-Style Dynamics Model

Predict a distribution over latent deltas instead of a single point estimate.

**Tasks:**
- [ ] Modify DynamicsModel to output `(mean, log_var)` -- 64 outputs instead of 32
- [ ] Implement reparameterization trick: `delta = mean + std * epsilon` where `epsilon ~ N(0,1)`
- [ ] Loss: `MSE(predicted_delta, actual_delta) + beta * KL(q || N(0,1))`
- [ ] Start with beta=0.0001, sweep upward: 0.001, 0.01
- [ ] Train on same latent dataset as Phase 1

**Success criteria:**
- With temperature=0 (using mean only), quality matches or exceeds deterministic baseline
- With temperature=1 (sampling), outputs are sharp (not blurry) but vary between runs
- Same starting state + same action produces visually different but plausible next frames across 10 samples
- KL loss is non-zero but bounded (model is actually using the stochastic component)

### 3.2 Temperature Control in Inference

Add user-controllable randomness during interactive play.

**Tasks:**
- [ ] Add temperature slider/key (0.0 = deterministic, 1.0 = full sampling)
- [ ] `delta = mean + temperature * std * epsilon`
- [ ] Display current temperature on screen

**Success criteria:**
- Temperature=0 produces identical frames given same input sequence (fully deterministic replay)
- Temperature=1 produces varied but coherent gameplay
- User can adjust temperature in real-time during play

### 3.3 Multi-Step Rollout Training

Train dynamics on its own predictions to reduce error accumulation.

**Tasks:**
- [ ] Implement scheduled sampling: with probability p, use model's own prediction as input for next step instead of ground truth
- [ ] Anneal p from 0 (all teacher-forced) to 0.5 over training
- [ ] Alternatively: train on 5-step rollouts with cumulative loss

**Success criteria:**
- 30-step rollout SSIM improves by at least +0.10 over single-step trained model
- 60-step rollout still produces recognizable Tetris gameplay (board, pieces, score visible)
- No mode collapse (model doesn't converge to a single static frame)

---

## Phase 4: Color and Higher Resolution

Move beyond grayscale 80x80 to handle real NES output.

### 4.1 Color Support (RGB)

Extend autoencoder from 1-channel grayscale to 3-channel RGB.

**Tasks:**
- [ ] Change IMAGE_CHANNELS from 1 to 3
- [ ] Adjust first conv layer: `Conv2d(3, 32, ...)` and last deconv: `ConvTranspose2d(32, 3, ...)`
- [ ] May need larger latent_dim to encode color information
- [ ] Retrain autoencoder on RGB frames
- [ ] Update pygame display to use RGB directly (remove grayscale-to-RGB replication)

**Success criteria:**
- Color reconstructions preserve correct hues (Tetris pieces are the right colors)
- No color bleeding between adjacent game elements
- Reconstruction SSIM > 0.90 (on RGB)

### 4.2 Higher Resolution (160x144 or 256x240)

Scale up to native NES resolution (256x240).

**Tasks:**
- [ ] Increase IMAGE_HEIGHT/WIDTH to target resolution
- [ ] Add encoder/decoder layers to handle larger spatial dimensions
- [ ] Increase latent_dim proportionally (likely 128-256)
- [ ] Ensure inference still hits real-time (may need to optimize decoder)

**Success criteria:**
- Native-resolution reconstructions where individual pixels of game sprites are distinguishable
- Inference at 30 FPS (may require torch.compile or TensorRT)
- Score/text on screen is fully legible at native resolution

---

## Phase 5: Multi-Game Support

Generalize from one game to many.

### 5.1 Second Game: Simple Side-Scroller

Add a game with scrolling backgrounds (fundamentally different visual dynamics).

**Tasks:**
- [ ] Pick a simple NES side-scroller (e.g., Excitebike, Balloon Fight)
- [ ] Collect dataset, train separate autoencoder + dynamics
- [ ] Compare latent_dim requirements vs Tetris
- [ ] Identify what breaks (scrolling, sprite overlap, animation cycles)

**Success criteria:**
- Autoencoder reconstructs scrolling backgrounds without tearing
- Dynamics model handles continuous motion (character moves smoothly)
- 10-step rollout SSIM > 0.65

### 5.2 Third Game: Enemy AI / Stochasticity

Add a game with non-deterministic elements (e.g., Ms. Pac-Man, Galaga).

**Tasks:**
- [ ] Collect large dataset (many more episodes needed to cover AI variance)
- [ ] Train with VAE dynamics (Phase 3) -- deterministic dynamics will fail here
- [ ] Evaluate: does sampling produce plausible but different enemy behaviors?

**Success criteria:**
- Ghost/enemy movement varies between rollouts (not frozen or averaged)
- Game feels "alive" -- enemies make varied but reasonable decisions
- Player can meaningfully react to enemy movement

### 5.3 Game Selection and Unified Architecture

Build a multi-game launcher and explore weight sharing.

**Tasks:**
- [ ] Create game selection menu (pygame UI or CLI arg)
- [ ] Per-game model loading (separate weights per game)
- [ ] Explore: can one autoencoder handle multiple games? (shared visual encoder)
- [ ] Explore: game-conditioned dynamics (one model, game ID as input)

**Success criteria:**
- User can switch between 3+ games from a menu
- Each game runs at 30 FPS with correct visuals and dynamics
- If shared encoder works: single model handles 3+ games with <10% quality drop vs per-game models

---

## Phase 6: Advanced Generation (Optional / Research)

Replace the VAE dynamics with more powerful generative models for maximum visual quality.

### 6.1 VQ-VAE Representation

Replace continuous latent space with discrete tokens.

**Tasks:**
- [ ] Implement VQ-VAE: encoder -> codebook quantization -> decoder
- [ ] Codebook size: 512-1024 tokens, each 64-dim
- [ ] Commitment loss + codebook EMA updates
- [ ] Compare reconstruction quality to continuous autoencoder

**Success criteria:**
- Reconstructions are at least as sharp as continuous VAE
- Codebook utilization > 80% (no codebook collapse)
- Tokens are interpretable (similar game states map to similar token sequences)

### 6.2 Transformer Dynamics on Discrete Tokens

Predict next frame's tokens autoregressively.

**Tasks:**
- [ ] Flatten frame tokens to sequence (e.g., 10x10 grid = 100 tokens)
- [ ] Train causal transformer: predict next frame's token sequence given current frame's tokens + action
- [ ] Sampling from token logits gives diverse, sharp predictions

**Success criteria:**
- Generated frames are sharper than VAE dynamics (no blur at all)
- Sampling produces varied but coherent futures
- Inference speed: at least 15 FPS (transformer decoding is sequential)

### 6.3 Flow Matching / Diffusion Dynamics

Use continuous generative models for maximum quality.

**Tasks:**
- [ ] Implement conditional flow matching: noise -> next_frame conditioned on (current_frame, action)
- [ ] Few-step generation (4-8 steps) for real-time inference
- [ ] Compare quality to VAE and VQ-VAE+Transformer approaches

**Success criteria:**
- Best visual quality of all approaches (measured by LPIPS)
- Real-time capable (15+ FPS with few-step sampling)
- Handles stochastic games naturally (different noise -> different futures)

---

## Quick Reference: What Exists vs What Needs Building

| Component | Status | Files |
|-----------|--------|-------|
| Dataset loading + pair construction | Done | `train.py:189-275` |
| Grayscale 80x80 autoencoder | Done | `train.py:60-125`, `main.py:37-90` |
| Deterministic dynamics model | Done | `train.py:128-146`, `main.py:93-108` |
| Training loop (AE + dynamics) | Done | `train.py:304-599` |
| Interactive pygame inference | Done | `main.py:209-258` |
| NES dataset + action mapping | **TODO** | Phase 1.1 |
| Evaluation harness + metrics | **TODO** | Phase 1.5 |
| Perceptual loss | **TODO** | Phase 2.1 |
| Latent noise training | Exists (disabled) | Phase 2.3, toggle in `train.py:25` |
| VAE dynamics (stochastic) | **TODO** | Phase 3.1 |
| Multi-step rollout training | **TODO** | Phase 3.3 |
| RGB support | **TODO** | Phase 4.1 |
| Higher resolution | **TODO** | Phase 4.2 |
| Multi-game support | **TODO** | Phase 5 |
| VQ-VAE / Transformer / Diffusion | **TODO** | Phase 6 |
