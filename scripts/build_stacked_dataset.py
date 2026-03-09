#!/usr/bin/env python3
"""Build a stacked-frame Breakout dataset and optionally push it to Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from datasets import Array2D, Array3D, Dataset, DatasetInfo, Features, Value, load_dataset

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from game_config import BREAKOUT_CONFIG, infer_game_config  # noqa: E402
from preprocessing import has_valid_black_background, preprocess_frame  # noqa: E402

DEFAULT_SOURCE_DATASET = BREAKOUT_CONFIG.dataset_id
DEFAULT_HISTORY_LENGTH = 4
DEFAULT_IMAGE_WIDTH = 80
DEFAULT_IMAGE_HEIGHT = 96


@dataclass
class BuildStats:
    rows_seen: int = 0
    rows_valid: int = 0
    rows_invalid: int = 0
    transitions_seen: int = 0
    yielded_samples: int = 0
    duplicate_samples: int = 0
    resets_episode_boundary: int = 0
    resets_invalid_frame: int = 0
    resets_terminated: int = 0


def default_target_dataset_id(source_dataset: str, history_length: int) -> str:
    return f"{source_dataset}_stack{history_length}_deduped"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a filtered dataset of stacked preprocessed frames and exact deduped "
            "(history, action, next_frame) transitions."
        )
    )
    parser.add_argument(
        "--source-dataset",
        default=DEFAULT_SOURCE_DATASET,
        help="Source Hugging Face dataset ID",
    )
    parser.add_argument(
        "--target-dataset",
        default=None,
        help=(
            "Target Hugging Face dataset ID. Defaults to "
            "<source>_stack4_deduped style naming."
        ),
    )
    parser.add_argument(
        "--game",
        default=BREAKOUT_CONFIG.name,
        help="Game profile used for preprocessing",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=DEFAULT_HISTORY_LENGTH,
        help="Number of preprocessed frames per input history",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=DEFAULT_IMAGE_WIDTH,
        help="Preprocessed frame width",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=DEFAULT_IMAGE_HEIGHT,
        help="Preprocessed frame height",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/hf_datasets",
        help="Cache directory used while materializing the generated dataset",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap for smoke testing without processing the full source dataset",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="Print progress every N source rows",
    )
    parser.add_argument(
        "--max-shard-size",
        default="200MB",
        help="Maximum parquet shard size when pushing to Hugging Face",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the uploaded dataset as private",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Build the dataset locally without pushing it to Hugging Face",
    )
    return parser.parse_args()


def extract_action_id(action_value: int | Sequence[int] | np.ndarray, n_actions: int) -> int:
    if isinstance(action_value, np.ndarray):
        if action_value.ndim == 0:
            action_id = int(action_value)
        else:
            flat = action_value.reshape(-1)
            if flat.size == n_actions and np.isin(flat, [0, 1]).all():
                action_id = int(np.argmax(flat))
            elif flat.size == 0:
                action_id = 0
            else:
                action_id = int(flat[0])
    elif isinstance(action_value, Sequence) and not isinstance(action_value, (str, bytes)):
        if len(action_value) == 0:
            action_id = 0
        elif len(action_value) == n_actions and all(
            value in (0, 1, 0.0, 1.0, False, True) for value in action_value
        ):
            action_id = int(np.argmax(np.asarray(action_value, dtype=np.uint8)))
        else:
            action_id = int(action_value[0])
    else:
        action_id = int(action_value)

    if not 0 <= action_id < n_actions:
        raise ValueError(f"Action id {action_id} is outside [0, {n_actions - 1}]")
    return action_id


def transition_hash(history: np.ndarray, action_id: int, next_frame: np.ndarray) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(history.tobytes())
    digest.update(action_id.to_bytes(2, byteorder="little", signed=False))
    digest.update(next_frame.tobytes())
    return digest.digest()


def dataset_features(history_length: int, image_height: int, image_width: int) -> Features:
    return Features(
        {
            "episode_id": Value("string"),
            "source_index": Value("int64"),
            "history": Array3D(
                shape=(history_length, image_height, image_width), dtype="uint8"
            ),
            "action": Value("int64"),
            "next_frame": Array2D(shape=(image_height, image_width), dtype="uint8"),
        }
    )


def build_generator(
    source_dataset: str,
    game: str,
    history_length: int,
    image_width: int,
    image_height: int,
    stats: BuildStats,
    progress_every: int,
    max_rows: int | None,
) -> Iterable[dict[str, object]]:
    game_config = infer_game_config(dataset_id=source_dataset, game=game)
    dataset = load_dataset(source_dataset, split="train")

    frame_window: deque[tuple[int, np.ndarray]] = deque(maxlen=history_length)
    current_episode_id: str | None = None
    previous_row: tuple[str, int, int, bool] | None = None
    seen_hashes: set[bytes] = set()

    for source_index, sample in enumerate(dataset):
        if max_rows is not None and source_index >= max_rows:
            break

        stats.rows_seen += 1

        raw_episode_id = sample.get("episode_id", "0")
        if isinstance(raw_episode_id, bytes):
            episode_id = raw_episode_id.hex()
        else:
            episode_id = str(raw_episode_id)
        if current_episode_id is None or episode_id != current_episode_id:
            if frame_window:
                stats.resets_episode_boundary += 1
            frame_window.clear()
            previous_row = None
            current_episode_id = episode_id

        observation = np.asarray(sample["observations"], dtype=np.uint8)
        if not has_valid_black_background(observation, game_config):
            stats.rows_invalid += 1
            if frame_window:
                stats.resets_invalid_frame += 1
            frame_window.clear()
            previous_row = None
            continue

        stats.rows_valid += 1
        processed_frame = preprocess_frame(
            observation,
            game_config,
            target_size=(image_width, image_height),
        ).astype(np.uint8, copy=False)

        if previous_row is not None and len(frame_window) == history_length:
            prev_episode_id, prev_source_index, prev_action_id, prev_cut = previous_row
            if not prev_cut and prev_episode_id == episode_id:
                stats.transitions_seen += 1
                history = np.stack([frame for _, frame in frame_window], axis=0)
                sample_hash = transition_hash(history, prev_action_id, processed_frame)
                if sample_hash in seen_hashes:
                    stats.duplicate_samples += 1
                else:
                    seen_hashes.add(sample_hash)
                    stats.yielded_samples += 1
                    yield {
                        "episode_id": prev_episode_id,
                        "source_index": prev_source_index,
                        "history": history,
                        "action": prev_action_id,
                        "next_frame": processed_frame,
                    }

        frame_window.append((source_index, processed_frame))

        transition_cut = bool(sample.get("terminations") or sample.get("truncations"))
        if transition_cut:
            stats.resets_terminated += 1

        action_id = extract_action_id(sample["actions"], game_config.n_actions)
        previous_row = (
            episode_id,
            source_index,
            action_id,
            transition_cut,
        )

        if progress_every > 0 and stats.rows_seen % progress_every == 0:
            print(
                f"[builder] rows={stats.rows_seen} valid={stats.rows_valid} "
                f"samples={stats.yielded_samples} duplicates={stats.duplicate_samples}"
            )


def main() -> None:
    args = parse_args()
    target_dataset = args.target_dataset or default_target_dataset_id(
        args.source_dataset, args.history_length
    )
    os.makedirs(args.cache_dir, exist_ok=True)

    stats = BuildStats()
    features = dataset_features(args.history_length, args.image_height, args.image_width)
    info = DatasetInfo(
        description=(
            "Preprocessed Breakout transitions with a stacked frame history. "
            "Each sample stores the last N binarized frames, the discrete action taken "
            "from the most recent frame in that history, and the resulting next frame. "
            "Exact duplicate (history, action, next_frame) tuples are removed."
        ),
        features=features,
    )

    print(f"Source dataset: {args.source_dataset}")
    print(f"Target dataset: {target_dataset}")
    print(f"History length: {args.history_length}")
    print(f"Image size: {args.image_width}x{args.image_height}")
    print(f"Cache dir: {args.cache_dir}")
    if args.max_rows is not None:
        print(f"Max rows: {args.max_rows}")

    dataset = Dataset.from_generator(
        build_generator,
        features=features,
        cache_dir=args.cache_dir,
        keep_in_memory=False,
        gen_kwargs={
            "source_dataset": args.source_dataset,
            "game": args.game,
            "history_length": args.history_length,
            "image_width": args.image_width,
            "image_height": args.image_height,
            "stats": stats,
            "progress_every": args.progress_every,
            "max_rows": args.max_rows,
        },
        info=info,
    )

    print("\nBuild complete")
    print(f"Rows seen: {stats.rows_seen}")
    print(f"Valid rows: {stats.rows_valid}")
    print(f"Invalid rows: {stats.rows_invalid}")
    print(f"Transitions considered: {stats.transitions_seen}")
    print(f"Unique samples written: {stats.yielded_samples}")
    print(f"Duplicate samples skipped: {stats.duplicate_samples}")
    print(f"Episode-boundary resets: {stats.resets_episode_boundary}")
    print(f"Invalid-frame resets: {stats.resets_invalid_frame}")
    print(f"Terminal/truncation resets: {stats.resets_terminated}")
    print(f"Generated dataset rows: {len(dataset)}")

    if args.skip_upload:
        print("Upload skipped.")
        return

    commit_info = dataset.push_to_hub(
        target_dataset,
        split="train",
        private=args.private,
        max_shard_size=args.max_shard_size,
        commit_message=(
            f"Add stacked-{args.history_length} preprocessed Breakout dataset with exact dedupe"
        ),
        commit_description=(
            f"Source dataset: {args.source_dataset}\n"
            f"History length: {args.history_length}\n"
            f"Image size: {args.image_width}x{args.image_height}\n"
            f"Rows seen: {stats.rows_seen}\n"
            f"Valid rows: {stats.rows_valid}\n"
            f"Unique samples written: {stats.yielded_samples}\n"
            f"Duplicate samples skipped: {stats.duplicate_samples}\n"
        ),
    )
    print(f"Uploaded to https://huggingface.co/datasets/{target_dataset}")
    print(f"Commit: {commit_info.oid}")


if __name__ == "__main__":
    main()
