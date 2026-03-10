#!/usr/bin/env python3
"""Build a train-ready rollout dataset from a raw Gymnasium recording dataset."""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from game_config import BREAKOUT_CONFIG, infer_game_config  # noqa: E402
from rollout_dataset import (  # noqa: E402
    build_rollout_sequences_from_raw_dataset,
    create_prepared_rollout_dataset,
    default_prepared_dataset_id,
    load_dataset_with_fallback,
)

DEFAULT_IMAGE_WIDTH = 80
DEFAULT_IMAGE_HEIGHT = 96
DEFAULT_HISTORY_LENGTH = 4
DEFAULT_UNROLL_STEPS = 8
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_SEQUENCE_STRIDE = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize the exact rollout windows used by train.py and optionally "
            "push them to Hugging Face as a new dataset."
        )
    )
    parser.add_argument(
        "--source-dataset",
        default=BREAKOUT_CONFIG.dataset_id,
        help="Source Hugging Face dataset ID",
    )
    parser.add_argument(
        "--target-dataset",
        default=None,
        help=(
            "Target Hugging Face dataset ID. Defaults to "
            "<source>_stack4_unroll8_train_ready style naming."
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
        help="Number of preprocessed frames per model input",
    )
    parser.add_argument(
        "--unroll-steps",
        type=int,
        default=DEFAULT_UNROLL_STEPS,
        help="Number of rollout targets stored per sample",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=DEFAULT_VAL_SPLIT,
        help="Validation split ratio used while materializing rollout sequences",
    )
    parser.add_argument(
        "--sequence-stride",
        type=int,
        default=DEFAULT_SEQUENCE_STRIDE,
        help="Keep every Nth rollout window from each valid segment",
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
        help="Print progress every N source rows while grouping samples",
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


def main() -> None:
    load_dotenv()
    args = parse_args()

    if args.history_length < 1:
        raise ValueError("history-length must be at least 1")
    if args.unroll_steps < 1:
        raise ValueError("unroll-steps must be at least 1")
    if args.sequence_stride < 1:
        raise ValueError("sequence-stride must be at least 1")
    if not 0 <= args.val_split < 1:
        raise ValueError("val-split must be in [0, 1)")

    target_dataset = args.target_dataset or default_prepared_dataset_id(
        args.source_dataset,
        args.history_length,
        args.unroll_steps,
    )
    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"Source dataset: {args.source_dataset}")
    print(f"Target dataset: {target_dataset}")
    print(f"History length: {args.history_length}")
    print(f"Unroll steps: {args.unroll_steps}")
    print(f"Validation split: {args.val_split}")
    print(f"Sequence stride: {args.sequence_stride}")
    print(f"Image size: {args.image_width}x{args.image_height}")
    print(f"Cache dir: {args.cache_dir}")
    if args.max_rows is not None:
        print(f"Max rows: {args.max_rows}")

    game_config = infer_game_config(dataset_id=args.source_dataset, game=args.game)
    dataset = load_dataset_with_fallback(
        args.source_dataset,
        split="train",
        cache_dir=args.cache_dir,
    )
    train_sequences, val_sequences, dataset_stats = build_rollout_sequences_from_raw_dataset(
        dataset,
        game_config,
        history_length=args.history_length,
        image_width=args.image_width,
        image_height=args.image_height,
        unroll_steps=args.unroll_steps,
        val_split=args.val_split,
        sequence_stride=args.sequence_stride,
        max_rows=args.max_rows,
        progress_every=args.progress_every,
    )

    if not train_sequences and not val_sequences:
        raise RuntimeError(
            "No rollout windows were produced from the selected source rows. "
            "Increase --max-rows or verify that the source dataset contains valid gameplay frames."
        )
    if not train_sequences:
        raise RuntimeError(
            "No training rollout windows were produced. Increase --max-rows or reduce --val-split."
        )
    if not val_sequences:
        raise RuntimeError(
            "No validation rollout windows were produced. Increase --max-rows or increase --val-split."
        )

    prepared_dataset = create_prepared_rollout_dataset(
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        history_length=args.history_length,
        unroll_steps=args.unroll_steps,
        image_height=args.image_height,
        image_width=args.image_width,
        n_actions=game_config.n_actions,
    )

    print("\nBuild complete")
    for key in (
        "data/raw_samples",
        "data/episode_count",
        "data/train_episodes",
        "data/val_episodes",
        "data/train_sequences",
        "data/val_sequences",
        "data/skipped_invalid_background",
        "data/skipped_terminal_pairs",
    ):
        print(f"{key}: {dataset_stats[key]}")

    if args.skip_upload:
        print("Upload skipped.")
        return

    commit_info = prepared_dataset.push_to_hub(
        target_dataset,
        private=args.private,
        max_shard_size=args.max_shard_size,
        commit_message=(
            f"Add train-ready rollout dataset (stack{args.history_length}, unroll{args.unroll_steps})"
        ),
        commit_description=(
            f"Source dataset: {args.source_dataset}\n"
            f"History length: {args.history_length}\n"
            f"Unroll steps: {args.unroll_steps}\n"
            f"Validation split: {args.val_split}\n"
            f"Sequence stride: {args.sequence_stride}\n"
            f"Image size: {args.image_width}x{args.image_height}\n"
            f"Raw samples: {dataset_stats['data/raw_samples']}\n"
            f"Train sequences: {dataset_stats['data/train_sequences']}\n"
            f"Validation sequences: {dataset_stats['data/val_sequences']}\n"
        ),
    )
    print(f"Uploaded to https://huggingface.co/datasets/{target_dataset}")
    print(f"Commit: {commit_info.oid}")


if __name__ == "__main__":
    main()
