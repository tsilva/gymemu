from __future__ import annotations

import hashlib

import numpy as np
from datasets import Array2D, Array3D, Dataset, DatasetDict, Features, load_dataset

from preprocessing import encode_action, has_valid_black_background, preprocess_frame

PREPARED_ROLLOUT_COLUMNS = {"history", "action_seq", "target_frames"}
STACKED_TRANSITION_COLUMNS = {"history", "action", "next_frame"}
DEFAULT_SPLIT_SEED = 42


def is_prepared_rollout_dataset(column_names) -> bool:
    return PREPARED_ROLLOUT_COLUMNS.issubset(set(column_names))


def is_stacked_transition_dataset(column_names) -> bool:
    return STACKED_TRANSITION_COLUMNS.issubset(set(column_names))


def default_prepared_dataset_id(source_dataset: str, history_length: int, unroll_steps: int) -> str:
    return f"{source_dataset}_stack{history_length}_unroll{unroll_steps}_train_ready"


def load_dataset_with_fallback(dataset_id: str, split: str | None = None, **load_kwargs):
    try:
        return (
            load_dataset(dataset_id, split=split, **load_kwargs)
            if split
            else load_dataset(dataset_id, **load_kwargs)
        )
    except Exception as exc:
        print(f"Error loading dataset: {exc}")
        print("Attempting to load with trust_remote_code=True...")
        retry_kwargs = dict(load_kwargs)
        retry_kwargs["trust_remote_code"] = True
        return (
            load_dataset(dataset_id, split=split, **retry_kwargs)
            if split
            else load_dataset(dataset_id, **retry_kwargs)
        )


def prepared_rollout_features(
    history_length: int,
    unroll_steps: int,
    image_height: int,
    image_width: int,
    n_actions: int,
) -> Features:
    return Features(
        {
            "history": Array3D(shape=(history_length, image_height, image_width), dtype="uint8"),
            "action_seq": Array2D(shape=(unroll_steps, n_actions), dtype="float32"),
            "target_frames": Array3D(
                shape=(unroll_steps, image_height, image_width), dtype="uint8"
            ),
        }
    )


def prepared_rollout_dimensions(dataset_split) -> dict[str, int]:
    features = dataset_split.features
    history_shape = tuple(features["history"].shape)
    action_seq_shape = tuple(features["action_seq"].shape)
    target_shape = tuple(features["target_frames"].shape)
    return {
        "history_length": int(history_shape[0]),
        "image_height": int(history_shape[1]),
        "image_width": int(history_shape[2]),
        "unroll_steps": int(action_seq_shape[0]),
        "n_actions": int(action_seq_shape[1]),
        "target_unroll_steps": int(target_shape[0]),
    }


def build_rollout_sequences_from_raw_dataset(
    dataset,
    game_config,
    history_length: int,
    image_width: int,
    image_height: int,
    unroll_steps: int,
    val_split: float,
    sequence_stride: int,
    max_rows: int | None = None,
    progress_every: int = 0,
    split_seed: int = DEFAULT_SPLIT_SEED,
):
    rows_seen = 0
    train_rollouts = []
    val_rollouts = []
    skipped_invalid_frames = 0
    skipped_terminal_pairs = 0
    episode_count = 0
    val_episode_count = 0
    train_episode_count = 0
    current_episode_id = None
    current_episode_split = "train"
    segment_frames = []
    segment_actions = []

    def flush_segment():
        if len(segment_frames) < history_length + unroll_steps:
            segment_frames.clear()
            segment_actions.clear()
            return

        n_windows = len(segment_frames) - history_length - unroll_steps + 1
        for window_index in range(n_windows):
            if window_index % sequence_stride != 0:
                continue

            history_frames = np.stack(
                segment_frames[window_index : window_index + history_length],
                axis=0,
            )
            action_seq = np.stack(
                segment_actions[
                    window_index + history_length - 1 : window_index
                    + history_length
                    - 1
                    + unroll_steps
                ],
                axis=0,
            )
            target_frames = np.stack(
                segment_frames[
                    window_index + history_length : window_index + history_length + unroll_steps
                ],
                axis=0,
            )

            target_list = val_rollouts if current_episode_split == "validation" else train_rollouts

            target_list.append((history_frames, action_seq, target_frames))

        segment_frames.clear()
        segment_actions.clear()

    for row_index, sample in enumerate(dataset):
        if max_rows is not None and row_index >= max_rows:
            break
        rows_seen += 1
        episode_id = _normalize_episode_id(sample.get("episode_id", 0))

        if current_episode_id is None or episode_id != current_episode_id:
            if current_episode_id is not None:
                flush_segment()
            current_episode_id = episode_id
            current_episode_split = _episode_split(episode_id, val_split, split_seed)
            episode_count += 1
            if current_episode_split == "validation":
                val_episode_count += 1
            else:
                train_episode_count += 1

        if progress_every > 0 and rows_seen % progress_every == 0:
            print(f"[builder] grouped_rows={rows_seen} episodes={episode_count}")

        frame = sample["observations"]
        if not has_valid_black_background(frame, game_config):
            skipped_invalid_frames += 1
            flush_segment()
            continue

        processed = preprocess_frame(
            frame,
            game_config,
            target_size=(image_width, image_height),
        ).astype(np.uint8, copy=False)
        try:
            action_array = encode_action(sample["actions"], game_config)
        except ValueError:
            skipped_invalid_frames += 1
            flush_segment()
            continue

        segment_frames.append(processed)
        segment_actions.append(action_array.astype(np.float32, copy=False))

        if sample.get("terminations") or sample.get("truncations"):
            skipped_terminal_pairs += 1
            flush_segment()

    if current_episode_id is not None:
        flush_segment()

    if episode_count == 1 and val_split > 0:
        combined_rollouts = train_rollouts + val_rollouts
        train_rollouts = []
        val_rollouts = []
        for sequence_index, rollout in enumerate(combined_rollouts):
            if _use_single_episode_val_slot(sequence_index, val_split):
                val_rollouts.append(rollout)
            else:
                train_rollouts.append(rollout)
        train_episode_count = 1 if train_rollouts else 0
        val_episode_count = 1 if val_rollouts else 0

    dataset_stats = {
        "data/raw_samples": rows_seen,
        "data/episode_count": episode_count,
        "data/train_episodes": train_episode_count,
        "data/val_episodes": val_episode_count,
        "data/train_sequences": len(train_rollouts),
        "data/val_sequences": len(val_rollouts),
        "data/skipped_invalid_background": skipped_invalid_frames,
        "data/skipped_terminal_pairs": skipped_terminal_pairs,
    }
    return train_rollouts, val_rollouts, dataset_stats


def create_prepared_rollout_dataset(
    train_sequences,
    val_sequences,
    history_length: int,
    unroll_steps: int,
    image_height: int,
    image_width: int,
    n_actions: int,
):
    features = prepared_rollout_features(
        history_length=history_length,
        unroll_steps=unroll_steps,
        image_height=image_height,
        image_width=image_width,
        n_actions=n_actions,
    )
    return DatasetDict(
        {
            "train": Dataset.from_dict(_rollout_dict(train_sequences), features=features),
            "validation": Dataset.from_dict(_rollout_dict(val_sequences), features=features),
        }
    )


def load_prepared_rollout_splits(dataset_id: str):
    dataset_dict = load_dataset_with_fallback(dataset_id)
    if "train" not in dataset_dict or "validation" not in dataset_dict:
        raise ValueError(
            "Prepared rollout datasets must expose both 'train' and 'validation' splits."
        )

    train_split = dataset_dict["train"]
    val_split = dataset_dict["validation"]
    if not is_prepared_rollout_dataset(train_split.column_names):
        raise ValueError(f"{dataset_id} does not look like a prepared rollout dataset.")
    if not is_prepared_rollout_dataset(val_split.column_names):
        raise ValueError(f"{dataset_id} validation split does not match the prepared schema.")

    train_sequences = [
        (
            np.asarray(row["history"], dtype=np.float32),
            np.asarray(row["action_seq"], dtype=np.float32),
            np.asarray(row["target_frames"], dtype=np.float32),
        )
        for row in train_split
    ]
    val_sequences = [
        (
            np.asarray(row["history"], dtype=np.float32),
            np.asarray(row["action_seq"], dtype=np.float32),
            np.asarray(row["target_frames"], dtype=np.float32),
        )
        for row in val_split
    ]
    return train_sequences, val_sequences, prepared_rollout_dimensions(train_split)


def _rollout_dict(sequences):
    histories = []
    action_sequences = []
    target_rollouts = []
    for history, action_seq, target_frames in sequences:
        histories.append(history.astype(np.uint8, copy=False))
        action_sequences.append(action_seq.astype(np.float32, copy=False))
        target_rollouts.append(target_frames.astype(np.uint8, copy=False))
    return {
        "history": histories,
        "action_seq": action_sequences,
        "target_frames": target_rollouts,
    }


def _use_single_episode_val_slot(sequence_index: int, val_split: float) -> bool:
    if val_split <= 0:
        return False
    return int((sequence_index + 1) * val_split) > int(sequence_index * val_split)


def _normalize_episode_id(raw_episode_id) -> str:
    if isinstance(raw_episode_id, bytes):
        return raw_episode_id.hex()
    return str(raw_episode_id)


def _episode_split(episode_id: str, val_split: float, split_seed: int) -> str:
    if val_split <= 0:
        return "train"
    payload = f"{split_seed}:{episode_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    bucket = int.from_bytes(digest, byteorder="big") / float(1 << 64)
    return "validation" if bucket < val_split else "train"
