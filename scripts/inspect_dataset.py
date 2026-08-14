#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import deque

import numpy as np
from datasets import load_dataset
from PIL import Image, ImageDraw

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dataset_utils import infer_history_length  # noqa: E402
from game_config import BREAKOUT_CONFIG, infer_game_config  # noqa: E402
from preprocessing import (  # noqa: E402
    encode_action,
    has_valid_black_background,
    preprocess_frame,
)

STACKED_SUFFIX_PATTERN = re.compile(r"_stack\d+(?:_[A-Za-z0-9_]+)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render visual spot checks for raw or deduped datasets using the same "
            "preprocessing path as training/runtime."
        )
    )
    parser.add_argument("--dataset", default=BREAKOUT_CONFIG.dataset_id)
    parser.add_argument("--split", default="train")
    parser.add_argument("--game", default=BREAKOUT_CONFIG.name)
    parser.add_argument("--source-dataset", default=None)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--history-length", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=80)
    parser.add_argument("--image-height", type=int, default=96)
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--sample-scale", type=int, default=2)
    parser.add_argument("--coverage-width", type=int, default=1200)
    parser.add_argument(
        "--emit-gifs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write one animated GIF per sampled transition to <output-dir>/gifs/",
    )
    parser.add_argument(
        "--gif-frame-duration-ms",
        type=int,
        default=220,
        help="Frame duration for emitted GIFs",
    )
    parser.add_argument(
        "--gif-final-hold-frames",
        type=int,
        default=4,
        help="How many extra times to repeat the final frame in each GIF",
    )
    parser.add_argument(
        "--output-dir",
        default=".cache/dataset_inspection",
        help="Directory where samples.png and coverage.png will be written",
    )
    return parser.parse_args()


def normalize_episode_id(raw_episode_id) -> str:
    if isinstance(raw_episode_id, bytes):
        return raw_episode_id.hex()
    return str(raw_episode_id)


def infer_source_dataset_id(dataset_id: str) -> str | None:
    stripped = STACKED_SUFFIX_PATTERN.sub("", dataset_id)
    if stripped != dataset_id:
        return stripped
    return None


def parse_action_id(action_value, game_config) -> int:
    encoded = encode_action(action_value, game_config)
    return int(np.argmax(encoded))


def action_label_map(game_config) -> dict[int, str]:
    labels = {0: "NOOP"}
    for binding in game_config.key_bindings:
        labels[binding.action_id] = binding.label
    return labels


def fixed_sample_indices(n_items: int, n_samples: int) -> np.ndarray:
    if n_items <= 0:
        return np.array([], dtype=np.int64)
    count = min(n_items, n_samples)
    if count == 1:
        return np.array([0], dtype=np.int64)
    return np.unique(np.linspace(0, n_items - 1, num=count, dtype=np.int64))


def load_transition_samples(
    dataset_id: str,
    split: str,
    game_config,
    history_length: int,
    image_width: int,
    image_height: int,
) -> tuple[list[dict[str, object]], bool]:
    dataset = load_dataset(dataset_id, split=split)
    uses_stacked_samples = {"history", "action", "next_frame"}.issubset(set(dataset.column_names))

    if uses_stacked_samples:
        samples = []
        for row in dataset:
            samples.append(
                {
                    "episode_id": normalize_episode_id(row.get("episode_id", "0")),
                    "source_index": int(row.get("source_index", -1)),
                    "history": np.asarray(row["history"], dtype=np.float32),
                    "action_id": parse_action_id(row["action"], game_config),
                    "next_frame": np.asarray(row["next_frame"], dtype=np.float32),
                }
            )
        return samples, True

    samples = []
    frame_history: deque[np.ndarray] = deque(maxlen=history_length)
    previous_action_id: int | None = None
    previous_episode_id: str | None = None
    previous_source_index: int | None = None

    for source_index, row in enumerate(dataset):
        episode_id = normalize_episode_id(row.get("episode_id", "0"))
        if previous_episode_id is not None and episode_id != previous_episode_id:
            frame_history.clear()
            previous_action_id = None
            previous_source_index = None

        observation = row["observations"]
        if not has_valid_black_background(observation, game_config):
            frame_history.clear()
            previous_action_id = None
            previous_source_index = None
            previous_episode_id = episode_id
            continue

        processed = preprocess_frame(
            observation,
            game_config,
            target_size=(image_width, image_height),
        ).astype(np.float32, copy=False)

        if previous_action_id is not None and len(frame_history) == history_length:
            samples.append(
                {
                    "episode_id": episode_id,
                    "source_index": previous_source_index
                    if previous_source_index is not None
                    else -1,
                    "history": np.stack(frame_history, axis=0),
                    "action_id": previous_action_id,
                    "next_frame": processed,
                }
            )

        frame_history.append(processed)
        previous_action_id = parse_action_id(row["actions"], game_config)
        previous_source_index = source_index
        previous_episode_id = episode_id

        if row.get("terminations") or row.get("truncations"):
            frame_history.clear()
            previous_action_id = None
            previous_source_index = None

    return samples, False


def make_frame_tile(frame: np.ndarray, scale: int) -> Image.Image:
    frame_uint8 = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    tile = Image.fromarray(frame_uint8, mode="L").convert("RGB")
    if scale != 1:
        tile = tile.resize(
            (tile.width * scale, tile.height * scale),
            resample=Image.Resampling.NEAREST,
        )
    return tile


def render_samples_image(
    samples: list[dict[str, object]],
    sample_indices: np.ndarray,
    labels: dict[int, str],
    output_path: str,
    scale: int,
) -> None:
    if sample_indices.size == 0:
        raise ValueError("Cannot render samples image for an empty dataset")

    sample = samples[int(sample_indices[0])]
    history = np.asarray(sample["history"], dtype=np.float32)
    tile_width = history.shape[2] * scale
    tile_height = history.shape[1] * scale
    history_length = history.shape[0]
    frame_gap = 4
    row_gap = 8
    header_height = 22
    label_width = 240
    columns = history_length + 1
    canvas_width = label_width + columns * tile_width + (columns - 1) * frame_gap + 16
    canvas_height = (
        header_height
        + len(sample_indices) * tile_height
        + max(0, len(sample_indices) - 1) * row_gap
        + 16
    )
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for column in range(history_length):
        x = label_width + column * (tile_width + frame_gap)
        draw.text((x, 4), f"h-{history_length - column - 1}", fill=(0, 0, 0))
    next_x = label_width + history_length * (tile_width + frame_gap)
    draw.text((next_x, 4), "next", fill=(0, 0, 0))

    for row_index, sample_index in enumerate(sample_indices):
        row_y = header_height + row_index * (tile_height + row_gap)
        sample = samples[int(sample_index)]
        action_id = int(sample["action_id"])
        source_index = int(sample["source_index"])
        label = (
            f"#{int(sample_index)}\n"
            f"src={source_index}\n"
            f"act={labels.get(action_id, str(action_id))}\n"
            f"ep={sample['episode_id']}"
        )
        draw.multiline_text((8, row_y + 4), label, fill=(0, 0, 0), spacing=2)

        history_frames = np.asarray(sample["history"], dtype=np.float32)
        for column, frame in enumerate(history_frames):
            x = label_width + column * (tile_width + frame_gap)
            canvas.paste(make_frame_tile(frame, scale), (x, row_y))

        x = label_width + history_length * (tile_width + frame_gap)
        canvas.paste(
            make_frame_tile(np.asarray(sample["next_frame"], dtype=np.float32), scale), (x, row_y)
        )

    canvas.save(output_path)


def render_transition_gif(
    sample: dict[str, object],
    sample_index: int,
    labels: dict[int, str],
    output_path: str,
    scale: int,
    frame_duration_ms: int,
    final_hold_frames: int,
) -> None:
    history_frames = np.asarray(sample["history"], dtype=np.float32)
    next_frame = np.asarray(sample["next_frame"], dtype=np.float32)
    action_id = int(sample["action_id"])
    action_label = labels.get(action_id, str(action_id))
    source_index = int(sample["source_index"])
    frame_sequence = [*history_frames, next_frame]
    frame_labels = [
        *(f"h-{history_frames.shape[0] - offset - 1}" for offset in range(history_frames.shape[0])),
        "next",
    ]

    frames: list[Image.Image] = []
    durations = [frame_duration_ms] * len(frame_sequence)
    if final_hold_frames > 0:
        durations[-1] = frame_duration_ms * (final_hold_frames + 1)

    for frame_label, frame in zip(frame_labels, frame_sequence):
        tile = make_frame_tile(frame, scale)
        canvas = Image.new("RGB", (tile.width + 16, tile.height + 44), color=(255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 4), f"#{sample_index} src={source_index}", fill=(0, 0, 0))
        draw.text((8, 18), f"{frame_label} act={action_label}", fill=(0, 0, 0))
        canvas.paste(tile, (8, 36))
        frames.append(canvas)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )


def render_transition_gifs(
    samples: list[dict[str, object]],
    sample_indices: np.ndarray,
    labels: dict[int, str],
    output_dir: str,
    scale: int,
    frame_duration_ms: int,
    final_hold_frames: int,
) -> list[str]:
    gif_dir = os.path.join(output_dir, "gifs")
    os.makedirs(gif_dir, exist_ok=True)
    output_paths = []

    for sample_index in sample_indices:
        sample = samples[int(sample_index)]
        action_label = labels.get(int(sample["action_id"]), str(sample["action_id"]))
        safe_action_label = action_label.lower().replace("+", "_").replace("/", "_")
        output_path = os.path.join(
            gif_dir,
            f"sample_{int(sample_index):05d}_src_{int(sample['source_index']):06d}_{safe_action_label}.gif",
        )
        render_transition_gif(
            sample,
            int(sample_index),
            labels,
            output_path,
            scale,
            frame_duration_ms,
            final_hold_frames,
        )
        output_paths.append(output_path)

    return output_paths


def bucket_mask(mask: np.ndarray, width: int) -> np.ndarray:
    if mask.size == 0:
        return np.zeros(width, dtype=np.float32)

    bucketed = np.zeros(width, dtype=np.float32)
    edges = np.linspace(0, mask.size, num=width + 1, dtype=np.int64)
    for bucket_index in range(width):
        start = int(edges[bucket_index])
        end = int(edges[bucket_index + 1])
        if end <= start:
            end = min(mask.size, start + 1)
        if end <= start:
            continue
        bucketed[bucket_index] = float(mask[start:end].mean())
    return bucketed


def render_coverage_image(
    source_dataset_id: str,
    source_split: str,
    kept_source_indices: list[int],
    game_config,
    output_path: str,
    image_width: int,
) -> dict[str, int | float]:
    source_dataset = load_dataset(source_dataset_id, split=source_split)
    total_rows = len(source_dataset)
    valid_mask = np.zeros(total_rows, dtype=np.float32)
    kept_mask = np.zeros(total_rows, dtype=np.float32)

    kept_count = 0
    for source_index in kept_source_indices:
        if 0 <= source_index < total_rows and not kept_mask[source_index]:
            kept_mask[source_index] = 1.0
            kept_count += 1

    valid_count = 0
    for row_index, row in enumerate(source_dataset):
        if has_valid_black_background(row["observations"], game_config):
            valid_mask[row_index] = 1.0
            valid_count += 1

    valid_bucket = bucket_mask(valid_mask, image_width)
    kept_bucket = bucket_mask(kept_mask, image_width)
    dropped_bucket = np.clip(valid_bucket - kept_bucket, 0.0, 1.0)

    label_width = 120
    row_height = 22
    row_gap = 10
    top_margin = 44
    bottom_margin = 18
    canvas_height = top_margin + 3 * row_height + 2 * row_gap + bottom_margin
    canvas_width = label_width + image_width + 16
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    draw.text((8, 8), f"source={source_dataset_id} [{source_split}]", fill=(0, 0, 0))
    draw.text(
        (8, 24),
        f"rows={total_rows} valid={valid_count} kept={kept_count}",
        fill=(0, 0, 0),
    )

    rows = [
        ("valid raw", valid_bucket, (210, 210, 210)),
        ("kept deduped", kept_bucket, (60, 160, 60)),
        ("valid dropped", dropped_bucket, (220, 160, 40)),
    ]
    for row_index, (label, bucket_values, color) in enumerate(rows):
        y = top_margin + row_index * (row_height + row_gap)
        draw.text((8, y + 4), label, fill=(0, 0, 0))
        bar_x = label_width
        draw.rectangle(
            (bar_x, y, bar_x + image_width - 1, y + row_height - 1),
            outline=(150, 150, 150),
            fill=(248, 248, 248),
        )
        for column, value in enumerate(bucket_values):
            if value <= 0.0:
                continue
            intensity = max(40, round(255 * float(value)))
            pixel = tuple(round(channel * intensity / 255) for channel in color)
            draw.line(
                (bar_x + column, y + 1, bar_x + column, y + row_height - 2),
                fill=pixel,
            )

    canvas.save(output_path)

    valid_indices = np.flatnonzero(valid_mask > 0.0)
    kept_indices = np.flatnonzero(kept_mask > 0.0)
    largest_gap = 0
    if kept_indices.size >= 2:
        largest_gap = int(np.diff(kept_indices).max())

    return {
        "source_rows": total_rows,
        "valid_rows": valid_count,
        "kept_rows": kept_count,
        "valid_keep_ratio": kept_count / max(valid_count, 1),
        "first_kept_source_index": int(kept_indices[0]) if kept_indices.size else -1,
        "last_kept_source_index": int(kept_indices[-1]) if kept_indices.size else -1,
        "largest_kept_gap": largest_gap,
        "first_valid_source_index": int(valid_indices[0]) if valid_indices.size else -1,
        "last_valid_source_index": int(valid_indices[-1]) if valid_indices.size else -1,
    }


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    game_config = infer_game_config(dataset_id=args.dataset, game=args.game)
    history_length = infer_history_length(args.dataset, args.history_length)
    samples, uses_stacked_samples = load_transition_samples(
        args.dataset,
        args.split,
        game_config,
        history_length,
        args.image_width,
        args.image_height,
    )
    if not samples:
        raise RuntimeError(f"No transition samples found in {args.dataset} [{args.split}]")

    labels = action_label_map(game_config)
    sample_indices = fixed_sample_indices(len(samples), args.num_samples)
    samples_path = os.path.join(args.output_dir, "samples.png")
    render_samples_image(samples, sample_indices, labels, samples_path, args.sample_scale)
    gif_paths: list[str] = []
    if args.emit_gifs:
        gif_paths = render_transition_gifs(
            samples,
            sample_indices,
            labels,
            args.output_dir,
            args.sample_scale,
            args.gif_frame_duration_ms,
            args.gif_final_hold_frames,
        )

    coverage_stats = None
    coverage_path = None
    source_dataset_id = args.source_dataset
    if source_dataset_id is None and uses_stacked_samples:
        source_dataset_id = infer_source_dataset_id(args.dataset)

    if source_dataset_id is not None:
        kept_source_indices = [
            int(sample["source_index"]) for sample in samples if int(sample["source_index"]) >= 0
        ]
        if kept_source_indices:
            coverage_path = os.path.join(args.output_dir, "coverage.png")
            coverage_stats = render_coverage_image(
                source_dataset_id,
                args.source_split,
                kept_source_indices,
                game_config,
                coverage_path,
                args.coverage_width,
            )

    print(f"dataset={args.dataset} split={args.split}")
    print(f"transitions={len(samples)}")
    print(f"stacked_dataset={uses_stacked_samples}")
    print(f"samples_image={samples_path}")
    if gif_paths:
        print(f"gif_dir={os.path.dirname(gif_paths[0])}")
        print(f"gif_count={len(gif_paths)}")
    if coverage_path is not None and coverage_stats is not None:
        print(f"coverage_image={coverage_path}")
        for key, value in coverage_stats.items():
            if isinstance(value, float):
                print(f"{key}={value:.6f}")
            else:
                print(f"{key}={value}")
    elif source_dataset_id is None:
        print("coverage_image=not_generated")
        print("coverage_reason=no_source_dataset")
    else:
        print("coverage_image=not_generated")
        print("coverage_reason=no_valid_source_index_metadata")


if __name__ == "__main__":
    main()
