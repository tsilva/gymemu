import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from context_corruption import (
    apply_seed_history_corruption,
    sample_history_corruption_strengths,
)
from dataset_utils import infer_history_length
from device_utils import configure_torch, get_device, move_batch_tensor, prepare_conv_module
from game_config import BREAKOUT_CONFIG, infer_game_config
from rollout_dataset import (
    default_prepared_dataset_id,
    is_prepared_rollout_dataset,
    load_dataset_with_fallback,
    prepared_rollout_dimensions,
)
from rollout_feedback import feedback_from_logits
from spatial_model import (
    SpatialLatentWorldModel,
    load_spatial_model_state_dict,
    normalized_spatial_model_state_dict,
)
from wandb_utils import TrainingTracker, make_image_grid

SEED = 42
IMAGE_WIDTH = 80
IMAGE_HEIGHT = 96
HISTORY_LENGTH = 1
TRAIN_N_EPOCHS = 50
TRAIN_BATCH_SIZE = 16
TRAIN_LEARNING_RATE = 0.001
TRAIN_WEIGHT_DECAY = 0.0
TRAIN_MAX_GRAD_NORM = 0.0
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.0
MAX_FOREGROUND_LOSS_WEIGHT = 64.0
FRAME_CHANGE_LOSS_WEIGHT = 24.0
FRAME_CHANGE_MAP_LOSS_WEIGHT = 32.0
FRAME_CHANGE_BCE_LOSS_WEIGHT = 4.0
FRAME_CHANGE_DICE_LOSS_WEIGHT = 1.0
SPATIAL_LATENT_CHANNELS = 32
SPATIAL_REFINE_BLOCKS = 4
UNROLL_STEPS = 16
FEEDBACK_MODE = "soft"
HISTORY_CORRUPTION_DEFAULT = True
HISTORY_CORRUPTION_MAX_STRENGTH = 0.08
HISTORY_CORRUPTION_FOREGROUND_DROPOUT_MAX = 0.06
RARE_ACTION_SAMPLING_POWER = 0.5
MAX_SEQUENCE_SAMPLE_WEIGHT = 8.0
ROLLOUT_SAMPLES_PER_EPOCH = 1024
MODEL_COMPILE_DEFAULT = True
DEFAULT_PREPARED_TRAIN_DATASET = default_prepared_dataset_id(
    BREAKOUT_CONFIG.dataset_id,
    4,
    UNROLL_STEPS,
)

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = get_device()
configure_torch(device)
print(f"Using device: {device}")


def maybe_compile(model, enabled, current_device):
    if enabled and current_device.type == "cuda":
        return torch.compile(model)
    return model


def empty_device_cache():
    if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def dataloader_kwargs(current_device):
    return {
        "num_workers": 0,
        "pin_memory": current_device.type == "cuda",
    }


def metric_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    return float(value)


def compute_grad_norm(parameters):
    grad_norm_sq = 0.0
    has_grad = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        has_grad = True
        grad_norm_sq += float(parameter.grad.detach().pow(2).sum().cpu().item())
    if not has_grad:
        return 0.0
    return grad_norm_sq ** 0.5


def add_weighted_metric_sums(metric_sums, metrics, weight):
    for key, value in metrics.items():
        metric_sums[key] = metric_sums.get(key, 0.0) + metric_value(value) * weight


def normalize_metric_sums(metric_sums, total_weight):
    if total_weight <= 0:
        return {key: 0.0 for key in metric_sums}
    return {key: value / total_weight for key, value in metric_sums.items()}


def is_val_improved(current_loss, best_loss, min_delta):
    return current_loss < (best_loss - min_delta)


def action_histogram_dict(n_actions):
    return {action_id: 0 for action_id in range(n_actions)}


def log_action_histogram(target_hist, action_array):
    action_id = int(np.argmax(action_array))
    target_hist[action_id] += 1


def action_fraction_metrics(prefix, histogram):
    total = sum(histogram.values())
    metrics = {}
    for action_id, count in histogram.items():
        denominator = total if total > 0 else 1
        metrics[f"{prefix}/action_{action_id}"] = count / denominator
    return metrics


def compute_sequence_sample_weights(
    sequence_action_ids,
    n_actions,
    rarity_power=RARE_ACTION_SAMPLING_POWER,
    max_weight=MAX_SEQUENCE_SAMPLE_WEIGHT,
):
    if not sequence_action_ids:
        return None, None

    action_hist = np.zeros(n_actions, dtype=np.float64)
    for action_ids in sequence_action_ids:
        np.add.at(action_hist, action_ids, 1)

    total = action_hist.sum()
    if total <= 0:
        return None, None

    action_freq = action_hist / total
    base_freq = action_freq[action_freq > 0].max()
    action_weights = np.ones(n_actions, dtype=np.float64)
    nonzero_mask = action_freq > 0
    action_weights[nonzero_mask] = np.clip(
        (base_freq / action_freq[nonzero_mask]) ** rarity_power,
        1.0,
        max_weight,
    )

    sequence_weights = []
    for action_ids in sequence_action_ids:
        sequence_weights.append(float(action_weights[action_ids].max()))

    return torch.as_tensor(sequence_weights, dtype=torch.double), action_weights


def fixed_sample_indices(n_items, n_samples):
    if n_items <= 0:
        return np.array([], dtype=np.int64)
    count = min(n_items, n_samples)
    rng = np.random.default_rng(SEED)
    return np.sort(rng.choice(n_items, size=count, replace=False))


class RolloutDataset(Dataset):
    def __init__(self, dataset_split):
        self.dataset_split = dataset_split

    def __len__(self):
        return len(self.dataset_split)

    def __getitem__(self, idx):
        row = self.dataset_split[int(idx)]
        history_tensor = torch.as_tensor(row["history"], dtype=torch.float32)
        action_tensor = torch.as_tensor(row["action_seq"], dtype=torch.float32)
        target_tensor = torch.as_tensor(row["target_frames"], dtype=torch.float32).unsqueeze(1)
        return history_tensor, action_tensor, target_tensor


def frame_prediction_components(logits, target, current_frame):
    current_frame = current_frame.detach()
    positive_ratio = target.mean(dim=(1, 2, 3), keepdim=True)
    foreground_weight = ((1.0 - positive_ratio) / positive_ratio.clamp_min(1e-6)).clamp(
        1.0,
        MAX_FOREGROUND_LOSS_WEIGHT,
    )
    change_mask = (target - current_frame).abs()
    weights = 1.0 + (foreground_weight - 1.0) * target + FRAME_CHANGE_LOSS_WEIGHT * change_mask
    probs = torch.sigmoid(logits)
    binary = (probs >= 0.5).float()
    predicted_change = (probs - current_frame).abs().clamp(1e-4, 1.0 - 1e-4)
    change_weights = 1.0 + FRAME_CHANGE_MAP_LOSS_WEIGHT * change_mask
    change_bce = F.binary_cross_entropy(predicted_change, change_mask, weight=change_weights)
    change_intersection = (predicted_change * change_mask).sum(dim=(1, 2, 3))
    change_union = predicted_change.sum(dim=(1, 2, 3)) + change_mask.sum(dim=(1, 2, 3))
    change_dice = ((2.0 * change_intersection + 1e-6) / (change_union + 1e-6)).mean()
    intersection = (binary * target).sum(dim=(1, 2, 3))
    union = binary.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = ((2.0 * intersection + 1e-6) / (union + 1e-6)).mean()
    foreground_pred_ratio = probs.mean()
    foreground_target_ratio = target.mean()
    loss_bce = F.binary_cross_entropy_with_logits(logits, target, weight=weights)
    loss_total = (
        loss_bce
        + FRAME_CHANGE_BCE_LOSS_WEIGHT * change_bce
        + FRAME_CHANGE_DICE_LOSS_WEIGHT * (1.0 - change_dice)
    )
    return {
        "loss_total": loss_total,
        "loss_frame_bce": loss_bce,
        "loss_change_bce": change_bce,
        "loss_change_dice": 1.0 - change_dice,
        "foreground_weight_mean": foreground_weight.mean(),
        "change_weight_mean": change_mask.mean(),
        "change_target_ratio": change_mask.mean(),
        "change_pred_ratio": predicted_change.mean(),
        "weight_mean": weights.mean(),
        "dice": dice,
        "change_dice": change_dice,
        "foreground_pred_ratio": foreground_pred_ratio,
        "foreground_target_ratio": foreground_target_ratio,
        "foreground_ratio_error": (foreground_pred_ratio - foreground_target_ratio).abs(),
    }


def make_rollout_media_grid(history_frames, predicted_rollouts, target_rollouts):
    rows = []
    for history_frame, predicted_frames, target_frames in zip(
        history_frames, predicted_rollouts, target_rollouts
    ):
        rows.append([history_frame, *predicted_frames])
        rows.append([history_frame, *target_frames])
    return make_image_grid(rows)


def build_rollout_eval_examples(val_split, n_samples=4):
    indices = fixed_sample_indices(len(val_split), n_samples)
    examples = []
    for index in indices:
        row = val_split[int(index)]
        examples.append(
            (
                np.asarray(row["history"], dtype=np.float32),
                np.asarray(row["action_seq"], dtype=np.float32),
                np.asarray(row["target_frames"], dtype=np.float32),
            )
        )
    return examples


def collect_split_action_data(dataset_split, n_actions):
    histogram = action_histogram_dict(n_actions)
    sequence_action_ids = []
    for row in dataset_split:
        action_ids = np.asarray(row["action_seq"], dtype=np.float32).argmax(axis=1)
        sequence_action_ids.append(action_ids)
        for action_id in action_ids:
            histogram[int(action_id)] += 1
    return histogram, sequence_action_ids


def load_rollout_dataset(
    dataset_id,
    game_config,
    history_length,
    image_width,
    image_height,
    unroll_steps,
):
    print(f"\nLoading dataset: {dataset_id}")
    dataset_dict = load_dataset_with_fallback(dataset_id)
    if "train" not in dataset_dict or "validation" not in dataset_dict:
        raise ValueError(
            "train.py only accepts prepared rollout datasets with 'train' and 'validation' "
            "splits. Build one with scripts/build_training_dataset.py."
        )
    if not is_prepared_rollout_dataset(dataset_dict["train"].column_names):
        raise ValueError(
            "train.py only accepts prepared rollout datasets built by "
            "scripts/build_training_dataset.py."
        )

    print(f"Train split rows: {len(dataset_dict['train'])}")
    print(f"Validation split rows: {len(dataset_dict['validation'])}")
    train_split = dataset_dict["train"]
    val_split = dataset_dict["validation"]
    prepared_dims = prepared_rollout_dimensions(train_split)
    if prepared_dims["history_length"] != history_length:
        raise ValueError(
            f"Prepared dataset history length is {prepared_dims['history_length']}, "
            f"but training was configured for {history_length}."
        )
    if prepared_dims["unroll_steps"] != unroll_steps:
        raise ValueError(
            f"Prepared dataset unroll steps is {prepared_dims['unroll_steps']}, "
            f"but training was configured for {unroll_steps}."
        )
    if prepared_dims["n_actions"] != game_config.n_actions:
        raise ValueError(
            f"Prepared dataset action size is {prepared_dims['n_actions']}, "
            f"but game profile expects {game_config.n_actions}."
        )
    if (
        prepared_dims["image_width"] != image_width
        or prepared_dims["image_height"] != image_height
    ):
        print(
            "Prepared dataset frame size differs from the requested preprocessing size: "
            f"{prepared_dims['image_width']}x{prepared_dims['image_height']} vs "
            f"{image_width}x{image_height}. Using the stored rollout tensors as-is."
        )

    train_action_hist, train_sequence_action_ids = collect_split_action_data(
        train_split,
        game_config.n_actions,
    )
    val_action_hist, _ = collect_split_action_data(val_split, game_config.n_actions)

    print(f"Train rollout sequences: {len(train_split)}")
    print(f"Validation rollout sequences: {len(val_split)}")
    dataset_stats = {
        "data/prepared_rollout_dataset": 1.0,
        "data/train_sequences": len(train_split),
        "data/val_sequences": len(val_split),
    }
    dataset_stats.update(action_fraction_metrics("data/train_action_fraction", train_action_hist))
    dataset_stats.update(action_fraction_metrics("data/val_action_fraction", val_action_hist))
    train_weights, action_weights = compute_sequence_sample_weights(
        train_sequence_action_ids,
        game_config.n_actions,
    )
    return train_split, val_split, dataset_stats, train_weights, action_weights


def make_train_loader(train_dataset, train_weights, batch_size, rollout_samples_per_epoch):
    sampler = None
    if train_weights is not None and rollout_samples_per_epoch > 0:
        num_samples = max(1, min(int(rollout_samples_per_epoch), len(train_dataset)))
        sampler = WeightedRandomSampler(
            train_weights,
            num_samples=num_samples,
            replacement=False,
        )
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        **dataloader_kwargs(device),
    )
    return loader


def maybe_corrupt_seed_history(history_frames, args):
    if not args.history_corruption:
        return history_frames, torch.zeros(history_frames.size(0), device=history_frames.device)

    strengths = sample_history_corruption_strengths(
        history_frames.size(0),
        max_strength=args.history_corruption_max_strength,
        device=history_frames.device,
    )
    corrupted = apply_seed_history_corruption(
        history_frames,
        strengths,
        foreground_dropout_max=args.history_corruption_foreground_dropout_max,
    )
    return corrupted, strengths


def train_spatial_model(
    args,
    game_config,
    train_split,
    val_split,
    train_weights,
    action_weights,
    tracker=None,
):
    dataset_name = args.dataset.replace("/", "__")
    model = prepare_conv_module(
        SpatialLatentWorldModel(
            history_length=args.history_length,
            n_actions=game_config.n_actions,
            latent_channels=args.spatial_latent_channels,
            refine_blocks=args.spatial_refine_blocks,
        ),
        device,
    )
    model = maybe_compile(model, args.model_compile, device)

    if args.spatial_dynamics_path:
        load_result = load_spatial_model_state_dict(
            model,
            torch.load(args.spatial_dynamics_path, map_location=device, weights_only=True),
        )
        print(f"Loaded spatial dynamics model from: {args.spatial_dynamics_path}")
        for note in load_result["notes"]:
            print(f"  -> {note}")
        if load_result["missing_keys"]:
            print(f"Missing keys after partial load: {len(load_result['missing_keys'])}")
        if load_result["unexpected_keys"]:
            print(f"Unexpected keys after partial load: {len(load_result['unexpected_keys'])}")

    train_dataset = RolloutDataset(train_split)
    val_dataset = RolloutDataset(val_split)
    train_loader = make_train_loader(
        train_dataset,
        train_weights,
        args.batch_size,
        args.rollout_samples_per_epoch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_kwargs(device),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    eval_examples = build_rollout_eval_examples(val_split, n_samples=4)
    best_val_loss = float("inf")
    best_model_path = None
    best_epoch = None
    epochs_without_improvement = 0
    optimizer_step = 0

    if action_weights is not None:
        for action_id, weight in enumerate(action_weights):
            print(f"Sampling weight action {action_id}: {weight:.3f}")

    for epoch in range(args.epochs):
        model.train()
        train_metric_sums = {}
        train_sample_count = 0
        train_epoch_start = time.perf_counter()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Train]")
        for batch_idx, (history_frames, action_seq, target_frames) in enumerate(pbar, start=1):
            history_frames = move_batch_tensor(
                history_frames, device, non_blocking=device.type == "cuda"
            )
            action_seq = action_seq.to(device, non_blocking=device.type == "cuda")
            target_frames = move_batch_tensor(
                target_frames, device, non_blocking=device.type == "cuda"
            )
            batch_size = history_frames.size(0)

            optimizer.zero_grad(set_to_none=True)

            rollout_history, corruption_strengths = maybe_corrupt_seed_history(history_frames, args)
            n_steps = action_seq.size(1)
            loss = torch.zeros((), device=device)
            batch_metric_sums = {}
            first_step_metrics = None
            last_step_metrics = None

            for step_idx in range(n_steps):
                logits = model(rollout_history, action_seq[:, step_idx, :])
                target_frame = target_frames[:, step_idx, :, :, :]
                step_metrics = frame_prediction_components(
                    logits,
                    target_frame,
                    rollout_history[:, -1:, :, :],
                )
                loss = loss + step_metrics["loss_total"]
                add_weighted_metric_sums(batch_metric_sums, step_metrics, 1.0)
                if step_idx == 0:
                    first_step_metrics = step_metrics
                if step_idx == n_steps - 1:
                    last_step_metrics = step_metrics
                _, _, next_input = feedback_from_logits(logits, args.feedback_mode)
                rollout_history = torch.cat([rollout_history[:, 1:, :, :], next_input], dim=1)

            loss = loss / n_steps
            loss.backward()

            grad_norm = compute_grad_norm(model.parameters())
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            optimizer_step += 1
            train_sample_count += batch_size
            batch_metrics = normalize_metric_sums(batch_metric_sums, n_steps)
            batch_metrics["loss_total"] = metric_value(loss)
            batch_metrics["loss_step_1"] = metric_value(first_step_metrics["loss_total"])
            batch_metrics["loss_step_last"] = metric_value(last_step_metrics["loss_total"])
            batch_metrics["loss_last_over_first"] = batch_metrics["loss_step_last"] / max(
                batch_metrics["loss_step_1"], 1e-8
            )
            batch_metrics["dice_step_1"] = metric_value(first_step_metrics["dice"])
            batch_metrics["dice_step_last"] = metric_value(last_step_metrics["dice"])
            batch_metrics["history_corruption_strength"] = metric_value(corruption_strengths.mean())
            add_weighted_metric_sums(train_metric_sums, batch_metrics, batch_size)

            if batch_idx == 1 or batch_idx % 20 == 0:
                pbar.set_postfix({"loss": f"{metric_value(loss):.6f}"})
                if tracker is not None:
                    tracker.log_batch(
                        "spatial_dynamics",
                        {
                            "train/loss_total": batch_metrics["loss_total"],
                            "train/loss_step_1": batch_metrics["loss_step_1"],
                            "train/loss_step_last": batch_metrics["loss_step_last"],
                            "train/loss_last_over_first": batch_metrics["loss_last_over_first"],
                            "train/dice_step_1": batch_metrics["dice_step_1"],
                            "train/dice_step_last": batch_metrics["dice_step_last"],
                            "train/history_corruption_strength": batch_metrics[
                                "history_corruption_strength"
                            ],
                            "train/change_dice": batch_metrics["change_dice"],
                            "train/loss_frame_bce": batch_metrics["loss_frame_bce"],
                            "train/loss_change_bce": batch_metrics["loss_change_bce"],
                            "train/loss_change_dice": batch_metrics["loss_change_dice"],
                            "train/foreground_ratio_error": batch_metrics[
                                "foreground_ratio_error"
                            ],
                            "train/foreground_weight_mean": batch_metrics[
                                "foreground_weight_mean"
                            ],
                            "train/change_weight_mean": batch_metrics["change_weight_mean"],
                            "train/change_target_ratio": batch_metrics["change_target_ratio"],
                            "train/change_pred_ratio": batch_metrics["change_pred_ratio"],
                            "train/weight_mean": batch_metrics["weight_mean"],
                            "train/grad_norm": grad_norm,
                            "train/lr": optimizer.param_groups[0]["lr"],
                        },
                        optimizer_step,
                    )

        train_metrics = {
            f"train/{key}": value
            for key, value in normalize_metric_sums(train_metric_sums, train_sample_count).items()
        }
        train_epoch_time = time.perf_counter() - train_epoch_start
        train_metrics["train/epoch_time_s"] = train_epoch_time
        train_metrics["train/samples_per_s"] = (
            train_sample_count / train_epoch_time if train_epoch_time > 0 else 0.0
        )
        avg_train_loss = train_metrics["train/loss_total"]

        model.eval()
        val_metric_sums = {}
        val_sample_count = 0
        val_epoch_start = time.perf_counter()

        with torch.inference_mode():
            for history_frames, action_seq, target_frames in tqdm(
                val_loader, desc=f"Epoch {epoch + 1}/{args.epochs} [Val]", leave=False
            ):
                history_frames = move_batch_tensor(
                    history_frames, device, non_blocking=device.type == "cuda"
                )
                action_seq = action_seq.to(device, non_blocking=device.type == "cuda")
                target_frames = move_batch_tensor(
                    target_frames, device, non_blocking=device.type == "cuda"
                )
                batch_size = history_frames.size(0)

                rollout_history = history_frames
                n_steps = action_seq.size(1)
                loss = torch.zeros((), device=device)
                batch_metric_sums = {}
                first_step_metrics = None
                last_step_metrics = None

                for step_idx in range(n_steps):
                    logits = model(rollout_history, action_seq[:, step_idx, :])
                    target_frame = target_frames[:, step_idx, :, :, :]
                    step_metrics = frame_prediction_components(
                        logits,
                        target_frame,
                        rollout_history[:, -1:, :, :],
                    )
                    loss = loss + step_metrics["loss_total"]
                    add_weighted_metric_sums(batch_metric_sums, step_metrics, 1.0)
                    if step_idx == 0:
                        first_step_metrics = step_metrics
                    if step_idx == n_steps - 1:
                        last_step_metrics = step_metrics
                    _, _, next_input = feedback_from_logits(logits, args.feedback_mode)
                    rollout_history = torch.cat([rollout_history[:, 1:, :, :], next_input], dim=1)

                loss = loss / n_steps
                batch_metrics = normalize_metric_sums(batch_metric_sums, n_steps)
                batch_metrics["loss_total"] = metric_value(loss)
                batch_metrics["loss_step_1"] = metric_value(first_step_metrics["loss_total"])
                batch_metrics["loss_step_last"] = metric_value(last_step_metrics["loss_total"])
                batch_metrics["loss_last_over_first"] = batch_metrics["loss_step_last"] / max(
                    batch_metrics["loss_step_1"], 1e-8
                )
                batch_metrics["dice_step_1"] = metric_value(first_step_metrics["dice"])
                batch_metrics["dice_step_last"] = metric_value(last_step_metrics["dice"])
                add_weighted_metric_sums(val_metric_sums, batch_metrics, batch_size)
                val_sample_count += batch_size

        val_metrics = {
            f"val/{key}": value
            for key, value in normalize_metric_sums(val_metric_sums, val_sample_count).items()
        }
        val_epoch_time = time.perf_counter() - val_epoch_start
        val_metrics["val/epoch_time_s"] = val_epoch_time
        val_metrics["val/samples_per_s"] = (
            val_sample_count / val_epoch_time if val_epoch_time > 0 else 0.0
        )
        avg_val_loss = val_metrics["val/loss_total"]

        print(
            "Epoch "
            f"{epoch + 1}/{args.epochs} - Train Loss: {avg_train_loss:.6f}, "
            f"Val Loss: {avg_val_loss:.6f}"
        )

        best_val_improved = False
        if is_val_improved(avg_val_loss, best_val_loss, args.early_stopping_min_delta):
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_model_path = os.path.join(args.output_dir, f"{dataset_name}-spatial-dynamics.pt")
            torch.save(normalized_spatial_model_state_dict(model.state_dict()), best_model_path)
            print(f"  -> Saved best model (val_loss: {best_val_loss:.6f})")
            best_val_improved = True
        elif args.early_stopping_patience > 0:
            epochs_without_improvement += 1
            print(
                "  -> No validation improvement "
                f"({epochs_without_improvement}/{args.early_stopping_patience})"
            )
            if epochs_without_improvement >= args.early_stopping_patience:
                print(
                    "  -> Early stopping triggered for spatial dynamics "
                    f"after epoch {epoch + 1}"
                )
                empty_device_cache()
                break

        epoch_metrics = {}
        epoch_metrics.update(train_metrics)
        epoch_metrics.update(val_metrics)
        epoch_metrics["health/generalization_gap_ratio"] = (
            avg_val_loss / avg_train_loss if avg_train_loss > 0 else 0.0
        )
        epoch_metrics["health/best_val_improved"] = float(best_val_improved)
        epoch_metrics["health/epochs_without_improvement"] = epochs_without_improvement
        if tracker is not None:
            tracker.log_epoch("spatial_dynamics", epoch_metrics, epoch)

        if tracker is not None and eval_examples and (best_val_improved or (epoch + 1) % 5 == 0):
            predicted_rollouts = []
            target_rollouts = []
            history_last_frames = []
            with torch.inference_mode():
                for history_frames, action_seq, target_frames in eval_examples:
                    history_tensor = (
                        torch.from_numpy(history_frames).float().unsqueeze(0).to(device)
                    )
                    action_tensor = torch.from_numpy(action_seq).float().unsqueeze(0).to(device)
                    rollout_history = history_tensor
                    rollout_predictions = []
                    for step_idx in range(action_tensor.size(1)):
                        logits = model(rollout_history, action_tensor[:, step_idx, :])
                        probs, _, next_input = feedback_from_logits(logits, args.feedback_mode)
                        rollout_predictions.append(probs[0, 0].detach().cpu().numpy())
                        rollout_history = torch.cat([rollout_history[:, 1:, :, :], next_input], dim=1)
                    predicted_rollouts.append(rollout_predictions)
                    target_rollouts.append([frame for frame in target_frames])
                    history_last_frames.append(history_frames[-1])

            grid = make_rollout_media_grid(
                history_last_frames,
                predicted_rollouts,
                target_rollouts,
            )
            tracker.log_media(
                "spatial_dynamics",
                {
                    "media/rollout_strip": tracker.image(
                        grid,
                        caption=f"Epoch {epoch + 1} spatial dynamics rollouts",
                    )
                },
                epoch,
            )

        empty_device_cache()

    return {
        "best_model_path": best_model_path,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the spatial latent world model")
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_PREPARED_TRAIN_DATASET,
        help="Prepared Hugging Face dataset ID built by scripts/build_training_dataset.py",
    )
    parser.add_argument(
        "--game",
        type=str,
        default=BREAKOUT_CONFIG.name,
        help="Game profile for preprocessing and controls",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=None,
        help="Number of recent frames fed into the world model",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Optional square image size override applied to both width and height",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=IMAGE_WIDTH,
        help="Preprocessed frame width",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=IMAGE_HEIGHT,
        help="Preprocessed frame height",
    )
    parser.add_argument("--epochs", type=int, default=TRAIN_N_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=TRAIN_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=TRAIN_WEIGHT_DECAY)
    parser.add_argument("--max-grad-norm", type=float, default=TRAIN_MAX_GRAD_NORM)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=EARLY_STOPPING_MIN_DELTA,
    )
    parser.add_argument("--unroll-steps", type=int, default=UNROLL_STEPS)
    parser.add_argument(
        "--rollout-samples-per-epoch",
        type=int,
        default=ROLLOUT_SAMPLES_PER_EPOCH,
        help="Number of weighted rollout windows to sample per training epoch",
    )
    parser.add_argument(
        "--feedback-mode",
        choices=("soft", "hard", "ste"),
        default=FEEDBACK_MODE,
        help="How predicted frames are fed back into history during training",
    )
    parser.add_argument(
        "--history-corruption",
        action=argparse.BooleanOptionalAction,
        default=HISTORY_CORRUPTION_DEFAULT,
        help="Corrupt the seed history during training with mild noise and foreground dropout",
    )
    parser.add_argument(
        "--history-corruption-max-strength",
        type=float,
        default=HISTORY_CORRUPTION_MAX_STRENGTH,
        help="Maximum per-sequence Gaussian noise scale for seed-history corruption",
    )
    parser.add_argument(
        "--history-corruption-foreground-dropout-max",
        type=float,
        default=HISTORY_CORRUPTION_FOREGROUND_DROPOUT_MAX,
        help="Maximum per-foreground-pixel dropout probability for seed-history corruption",
    )
    parser.add_argument(
        "--spatial-latent-channels",
        type=int,
        default=SPATIAL_LATENT_CHANNELS,
        help="Channel count of the spatial latent map used by the world model",
    )
    parser.add_argument(
        "--spatial-refine-blocks",
        type=int,
        default=SPATIAL_REFINE_BLOCKS,
        help="Number of action-conditioned residual blocks in the spatial latent model",
    )
    parser.add_argument(
        "--spatial-dynamics-path",
        default=None,
        help="Optional path to a pre-trained spatial dynamics checkpoint to resume from",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./models",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--model-compile",
        action=argparse.BooleanOptionalAction,
        default=MODEL_COMPILE_DEFAULT,
        help="Enable torch.compile when running on CUDA",
    )
    return parser.parse_args()


def main():
    load_dotenv()

    args = parse_args()
    args.model_family = "spatial"
    args.history_length = infer_history_length(args.dataset, args.history_length)
    if args.image_size is not None:
        args.image_width = args.image_size
        args.image_height = args.image_size

    if args.history_length < 1:
        raise ValueError("history-length must be at least 1")
    if args.unroll_steps < 1:
        raise ValueError("unroll-steps must be at least 1")

    os.makedirs(args.output_dir, exist_ok=True)

    game_config = infer_game_config(dataset_id=args.dataset, game=args.game)
    print(f"Game profile: {game_config.name}")
    print(f"Dataset: {args.dataset}")
    print(f"Frame size: {args.image_width}x{args.image_height}")
    print(f"History length: {args.history_length}")
    print(f"Unroll steps: {args.unroll_steps}")
    print(f"Feedback mode: {args.feedback_mode}")
    print(f"History corruption: {args.history_corruption}")
    print(f"History corruption max strength: {args.history_corruption_max_strength}")
    print(
        "History corruption foreground dropout max: "
        f"{args.history_corruption_foreground_dropout_max}"
    )
    print(f"Spatial latent channels: {args.spatial_latent_channels}")
    print(f"Spatial refine blocks: {args.spatial_refine_blocks}")
    print(f"Spatial dynamics resume path: {args.spatial_dynamics_path}")

    train_split, val_split, dataset_stats, train_weights, action_weights = load_rollout_dataset(
        args.dataset,
        game_config,
        history_length=args.history_length,
        image_width=args.image_width,
        image_height=args.image_height,
        unroll_steps=args.unroll_steps,
    )

    tracker = TrainingTracker()
    tracker.init_run(args, dataset_stats)
    tracker.log_run_metrics(dataset_stats)

    result = train_spatial_model(
        args,
        game_config,
        train_split,
        val_split,
        train_weights,
        action_weights,
        tracker=tracker,
    )

    summary = {
        "spatial_dynamics/best_val_loss": result["best_val_loss"],
        "spatial_dynamics/best_epoch": result["best_epoch"],
        "spatial_dynamics/best_model_path": result["best_model_path"],
    }
    tracker.finish(summary)

    print("\nTraining complete.")
    print(f"Best model path: {result['best_model_path']}")
    print(f"Best validation loss: {result['best_val_loss']:.6f}")
    print(f"Best epoch: {result['best_epoch']}")


if __name__ == "__main__":
    main()
