import os
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
import pygame
from PIL import Image
import numpy as np

nb_seed = 42
ds_id = "tsilva/GymnasiumRecording__Tetris_GameBoy"
model_compile = True
model_latent_dim = 32
model_latent_noise_factor = 0.0
train_n_epochs = 100
train_batch_size = 128
train_max_grad_norm = 0
train_weight_decay = 0
train_warmup_ratio = 0
train_loss_function = "l1"
train_learning_rate = 0.001
train_drift = 1
val_epochs = 10

# Model loading configuration
# Set use_local_models = True to load from ./models/ directory
# Set use_local_models = False to download from Hugging Face
use_local_models = False
models_dir = "./models"

image_channels = 1
output_channels = 1
image_height = 80
image_width = 80
use_bottleneck = True


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, image_height, image_width)
            dummy_output = self.encoder_conv(dummy_input)
            self._flattened_size = dummy_output.view(1, -1).shape[1]
            self._conv_output_shape = dummy_output.shape[1:]

        if use_bottleneck:
            self.fc_enc = nn.Linear(self._flattened_size, model_latent_dim)
            self.fc_dec = nn.Linear(model_latent_dim, self._flattened_size)

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder_conv(x)
        if use_bottleneck:
            x = x.view(x.size(0), -1)
            x = self.fc_enc(x)
        return x

    def decode(self, z):
        if use_bottleneck:
            z = self.fc_dec(z)
            z = z.view(z.size(0), *self._conv_output_shape)
        z = self.decoder_conv(z)
        return z

    def forward(self, x):
        z = self.encode(x)
        z_input = z
        if self.training and self.model_latent_noise_factor > 0:
            noise = torch.randn_like(z_input) * self.model_latent_noise_factor
            z_input += noise
        out = self.decode(z_input)
        return out, z


class DynamicsModel(nn.Module):
    def __init__(self, z_dim=32, n_actions=9):
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


# Model instantiation
use_bottleneck = model_latent_dim > 0
input_channels = image_channels
output_channels = image_channels

# Device selection: CUDA > MPS (Mac) > CPU
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using device: {device}")

representation_model = ConvAutoencoder().to(device)
if device.type == "cuda":
    representation_model = torch.compile(representation_model)

if use_local_models:
    # Load from local models directory
    model_path = os.path.join(
        models_dir, f"{ds_id.replace('/', '__')}-representation.pt"
    )
    print(f"Loading local model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Local model not found: {model_path}. Train first with train.py"
        )
else:
    # Download from Hugging Face
    model_path = hf_hub_download(
        repo_id=f"{ds_id}-representation",
        filename="model.pt",
    )
    print(f"Model downloaded to: {model_path}")

representation_model_state_dict = torch.load(
    model_path, map_location=device, weights_only=True
)
representation_model.load_state_dict(representation_model_state_dict)
representation_model.eval()

dynamics_model = DynamicsModel(z_dim=model_latent_dim, n_actions=9).to(device)
if device.type == "cuda":
    dynamics_model = torch.compile(dynamics_model)

if use_local_models:
    # Load from local models directory
    model_path = os.path.join(models_dir, f"{ds_id.replace('/', '__')}-dynamics.pt")
    print(f"Loading local model from: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Local model not found: {model_path}. Train first with train.py"
        )
else:
    # Download from Hugging Face
    model_path = hf_hub_download(
        repo_id=f"{ds_id}-dynamics",
        filename="model.pt",
    )
    print(f"Model downloaded to: {model_path}")

dynamics_model_state_dict = torch.load(
    model_path, map_location=device, weights_only=True
)
dynamics_model.load_state_dict(dynamics_model_state_dict)
dynamics_model.eval()

# --- Pygame setup ---
pygame.init()
win_size = (image_width, image_height)
screen = pygame.display.set_mode(win_size)
pygame.display.set_caption("Latent Dynamics Viewer")

# --- Load and preprocess start.jpg ---
img = Image.open("start.png").convert("L").resize(win_size)
img_np = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
img_tensor = (
    torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
)  # shape: (1, 1, H, W)

# --- Get initial latent vector ---
with torch.no_grad():
    latent = representation_model.encode(img_tensor)

# Render the initial frame (start.png) before the loop
recon_img = img_np  # Already normalized and loaded
recon_img_disp = np.clip(recon_img * 255, 0, 255).astype(np.uint8)
surf = pygame.surfarray.make_surface(
    np.repeat(recon_img_disp[:, :, None], 3, axis=2).swapaxes(0, 1)
)
screen.blit(surf, (0, 0))
pygame.display.flip()

running = True
clock = pygame.time.Clock()

while running:
    action = np.zeros(9, dtype=np.float32)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    # Use pygame.K_* constants directly, not their integer values
    if keys[pygame.K_z]:  # A
        action[0] = 1
    if keys[pygame.K_q]:  # SELECT
        action[2] = 1
    if keys[pygame.K_r]:  # START
        action[3] = 1
    if keys[pygame.K_UP]:
        action[4] = 1
    if keys[pygame.K_DOWN]:
        action[5] = 1
    if keys[pygame.K_LEFT]:
        action[6] = 1
    if keys[pygame.K_RIGHT]:
        action[7] = 1
    if keys[pygame.K_x]:  # B
        action[8] = 1

    print(action)

    # action = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0])
    action_tensor = torch.from_numpy(action).unsqueeze(0).to(device)  # shape: (1, 9)
    # print(action_tensor)
    z_and_a = torch.cat([latent, action_tensor], dim=1)

    with torch.no_grad():
        delta_latent = dynamics_model(z_and_a)
        next_latent = latent + delta_latent  # <-- Correct latent update
        recon = representation_model.decode(next_latent)
        recon_img = recon.squeeze().detach().cpu().numpy()
        recon_img = np.clip(recon_img * 255, 0, 255).astype(np.uint8)

    # Display the frame
    surf = pygame.surfarray.make_surface(
        np.repeat(recon_img[:, :, None], 3, axis=2).swapaxes(0, 1)
    )
    screen.blit(surf, (0, 0))
    pygame.display.flip()
    clock.tick(30)  # Limit to 30 FPS

    latent = next_latent  # Update latent for next step

pygame.quit()
