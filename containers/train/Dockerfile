# syntax=docker/dockerfile:1.7
ARG PYTHON_IMAGE=python:3.13-slim-bookworm@sha256:ed86c82274b3c69b52fb5820f358f0bd7df0b603332063cb5c6e32bd220c3e6e
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.13@sha256:841c8e6fe30a8b07b4478d12d0c608cba6de66102d29d65d1cc423af86051563
FROM ${UV_IMAGE} AS uv
FROM ${PYTHON_IMAGE} AS dependencies
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates g++ git libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 UV_LINK_MODE=copy \
    UV_CACHE_DIR=/root/.cache/uv \
    UV_PROJECT_ENVIRONMENT=/opt/gymemu/.venv \
    PATH="/opt/gymemu/.venv/bin:$PATH" \
    PYTHONPATH=/opt/gymemu \
    HF_HOME=/workspace/huggingface \
    XDG_CACHE_HOME=/workspace/.cache \
    WANDB_DIR=/workspace/wandb \
    GYMEMU_WORKSPACE=/workspace
WORKDIR /opt/gymemu
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache
FROM dependencies AS runtime
COPY .dockerignore ./
COPY *.py ./
COPY gymemu ./gymemu
COPY configs ./configs
COPY start_states ./start_states
COPY containers/train ./containers/train
COPY README.md ./
COPY docs ./docs
COPY --chmod=755 containers/train/entrypoint.sh /usr/local/bin/gymemu-container
ARG VCS_REF=uncommitted
ENV GYMEMU_IMAGE_SOURCE_SHA=${VCS_REF}
LABEL org.opencontainers.image.title="Gymemu training runtime" \
    org.opencontainers.image.source="https://github.com/tsilva/gymemu" \
    org.opencontainers.image.revision=${VCS_REF}
WORKDIR /workspace
ENTRYPOINT ["gymemu-container"]
CMD ["smoke", "--device", "cpu"]
