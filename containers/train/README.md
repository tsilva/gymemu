# Training containers

The Linux amd64 image installs Python 3.13 and the committed `uv.lock`, including
PyTorch's CUDA libraries. The host supplies the NVIDIA driver. GPU recipes require
native bfloat16 and a driver compatible with the locked CUDA runtime. The image
contains no datasets or credentials.

GitHub Actions builds and CPU-tests the image before publishing
`ghcr.io/tsilva/gymemu/train:sha-<commit>` and, on `main`, `:main`.
An explicitly dispatched branch build can publish only its commit tag.
Copy the **digest reference**
from the workflow summary for runs. Tags can move. Dependency build layers are
cached separately from application code.

## Build and verify

```bash
docker build --platform linux/amd64 -f containers/train/Dockerfile \
  --build-arg VCS_REF="$(git rev-parse HEAD)" -t gymemu-train:local .
docker run --rm --shm-size 1g -v "$PWD/runs/container-check:/workspace" \
  gymemu-train:local smoke --device cpu
docker run --rm --gpus all --shm-size 2g \
  -v "$PWD/runs/container-check:/workspace" \
  gymemu-train:local smoke --device cuda
```

The smoke trains direct, two-stage latent, and ball-state models on a tiny dataset
with unordered frame IDs and transition rows. It verifies frame caching,
checkpoints, saved scenes, empty startup, autoregressive playback, and reset.
The GPU check uses bf16 and compilation. Each check writes
`verification/<timestamp>/verification.json`. `--online` additionally checks W&B
and R2 with injected credentials. This verifies execution, not tracking quality.

## Run a recipe

```bash
export IMAGE='ghcr.io/tsilva/gymemu/train@sha256:REPLACE_WITH_WORKFLOW_DIGEST'
docker run --rm --gpus all --shm-size 2g \
  --env-file /path/to/private/gymemu.env -e GYMEMU_IMAGE_REF="$IMAGE" \
  -v /persistent/gymemu:/workspace \
  "$IMAGE" run recipe=breakout_ball
```

`run` checks CUDA, downloads the pinned dataset, builds a missing frame cache,
validates an existing cache, and starts training in a unique `/workspace/runs`
directory. It accepts Hydra overrides and `--recipe /workspace/recipe.yaml`.
`train` skips cache preparation. `cache`, `play`, and `upload` invoke the matching
root scripts. The default command is the bounded CPU smoke.

Persist `/workspace` for datasets, frame caches, compiled kernels, W&B logs, and
outputs. Inject these runtime variables through private env files or provider secrets:

- `WANDB_API_KEY`
- `GYMEMU_MODELS_R2_ENDPOINT_URL`
- `GYMEMU_MODELS_R2_ACCESS_KEY_ID`
- `GYMEMU_MODELS_R2_SECRET_ACCESS_KEY`
- `GYMEMU_PUBLIC_R2_ENDPOINT_URL`
- `GYMEMU_PUBLIC_R2_ACCESS_KEY_ID`
- `GYMEMU_PUBLIC_R2_SECRET_ACCESS_KEY`
- Optional `HF_TOKEN` for private datasets
- `GYMEMU_IMAGE_REF`, the exact digest saved in `reproduction.json`

Use separate R2 credentials scoped to the private `gymemu` and public
`gymemu-public` buckets. The macOS Keychain profiles are
unavailable inside Linux containers. For local-only runs, pass
`wandb.mode=disabled r2.enabled=false`. Keep secrets out of images and Hydra overrides.

## Beast-3 scheduling

Use the existing dstack 0.20.28 coordinator and `main` project through Gradlab's
private operator SSH tunnel. Set `DSTACK_SERVER_URL` and `DSTACK_TOKEN` privately.
The project needs the seven named online secrets above. Private GHCR images also
need registry pull credentials configured on the coordinator.

```bash
# Render without allocating a GPU.
uv run python ops/dstack/launch.py --image "$IMAGE" --name gymemu-smoke --smoke --offline
# Submit the bounded GPU check.
uv run python ops/dstack/launch.py --image "$IMAGE" --name gymemu-smoke --smoke --offline --submit
# Full ball recipe, with W&B and R2 enabled.
uv run python ops/dstack/launch.py --image "$IMAGE" --name gymemu-ball --submit
dstack ps --project main
dstack logs --project main gymemu-ball
```

Tasks reuse fleet `b3`, requesting one GPU, at least 12 CPUs and 40 GB RAM. They
cannot provision cloud instances. Host `~/.local/share/gymemu/container-workspace`
maps to `/workspace`; the host Hugging Face and Gymemu frame caches are reused.
Smokes have a one-hour limit and training has a 24-hour limit. There is no automatic
interruption retry because exact optimizer-state resume is not implemented.

Legacy jobs outside dstack are invisible to its capacity accounting. Check that
`~/.config/gymemu/training.lock` is free on Beast-3 before submitting, and do not
interrupt existing training. Route future Gymemu training through dstack so it
shares capacity accounting with Gradlab.

## Runpod

Create a GPU Pod template using the published image digest. Set entrypoint to
`["gymemu-container"]` and command to `["run", "recipe=breakout_ball"]`.
For a UI field that overrides the entire start command, enter
`gymemu-container run recipe=breakout_ball`. Check the
[Runpod template fields](https://docs.runpod.io/api-reference/templates/POST/templates)
when translating these arrays to UI settings.

Mount persistent storage at `/workspace`, inject the same secrets, and use a
bf16-capable GPU with at least 40 GB host RAM. Allow storage for the downloaded
dataset, the 7.8 GB frame cache, outputs, and the container image. Disable restart
loops that would repeat a completed training job.

First run `smoke --device cuda --online`, then a bounded real-data check, then the
full recipe. Verify W&B and the R2 manifest before deleting a Pod. Training process
exit alone does not stop Pod billing; explicitly stop or terminate it after the
artifacts are safe. Persistent storage still affects cost. Runpod provisioning and
cost benchmarking are separate from Beast-3 verification.
