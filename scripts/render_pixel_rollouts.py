import argparse
from collections import deque

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image, ImageDraw

from game_config import BREAKOUT_CONFIG, infer_game_config
from pixel_feedback import (
    feedback_from_logits,
    paddle_motion_mask,
    shift_paddle_frames,
    static_noop_mask,
)
from pixel_model import FrameDynamicsModel
from preprocessing import has_valid_black_background, preprocess_frame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render scripted rollouts for a pixel dynamics model"
    )
    parser.add_argument("--dataset", default=BREAKOUT_CONFIG.dataset_id)
    parser.add_argument("--game", default=BREAKOUT_CONFIG.name)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--image-width", type=int, default=80)
    parser.add_argument("--image-height", type=int, default=96)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--pixel-refine-blocks", type=int, default=0)
    parser.add_argument(
        "--pixel-static-noop-hold",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pixel-static-history-threshold", type=float, default=40.0)
    parser.add_argument(
        "--pixel-static-predicted-diff-threshold",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--pixel-paddle-motion-hold",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pixel-paddle-motion-threshold", type=float, default=24.0)
    parser.add_argument("--pixel-paddle-shift", type=int, default=2)
    parser.add_argument(
        "--feedback",
        choices=("soft", "hard"),
        default="soft",
        help="Whether to feed sigmoid probabilities or thresholded frames back into history",
    )
    parser.add_argument(
        "--start-valid-offset",
        type=int,
        default=0,
        help="Skip this many valid history windows before choosing the rollout seed",
    )
    parser.add_argument(
        "--output",
        default="/tmp/pixel_rollouts.png",
        help="Path for the rendered montage",
    )
    return parser.parse_args()


def make_action_vector(action_id, n_actions):
    action = np.zeros(n_actions, dtype=np.float32)
    action[action_id] = 1.0
    return action


def build_policies(game_config, n_actions):
    policies = [("NOOP", [0])]
    labels_to_ids = {binding.label: binding.action_id for binding in game_config.key_bindings}
    if "FIRE" in labels_to_ids:
        policies.append(("FIRE", [labels_to_ids["FIRE"]] + [0]))
    if "RIGHT" in labels_to_ids:
        policies.append(("RIGHT", [labels_to_ids["RIGHT"]]))
    if "LEFT" in labels_to_ids:
        policies.append(("LEFT", [labels_to_ids["LEFT"]]))

    action_vectors = {}
    for label, action_ids in policies:
        action_vectors[label] = [
            make_action_vector(action_id, n_actions) for action_id in action_ids
        ]
    return action_vectors


def fetch_seed_history(args, game_config):
    dataset = load_dataset(args.dataset, split="train", streaming=True)
    history = deque(maxlen=args.history_length)
    valid_window_index = 0
    first_window = None

    for sample in dataset:
        frame = sample["observations"]
        if not has_valid_black_background(frame, game_config):
            history.clear()
            continue

        processed = preprocess_frame(
            frame,
            game_config,
            target_size=(args.image_width, args.image_height),
        ).astype(np.float32, copy=False)
        history.append(processed)
        if len(history) < args.history_length:
            continue

        window = np.stack(history, axis=0)
        if first_window is None:
            first_window = window

        motion_pixels = 0.0
        if args.history_length >= 2:
            motion_pixels = np.abs(window[-1] - window[-2]).sum()

        if valid_window_index >= args.start_valid_offset and motion_pixels > 0:
            return window

        valid_window_index += 1

    if first_window is None:
        raise RuntimeError(
            f"Could not find {args.history_length} consecutive valid frames in {args.dataset}"
        )
    return first_window


def rollout_policy(
    model,
    seed_history,
    policy_actions,
    steps,
    feedback,
    static_noop_hold,
    static_history_threshold,
    static_predicted_diff_threshold,
    paddle_motion_hold,
    paddle_motion_threshold,
    paddle_shift,
):
    history = torch.from_numpy(seed_history).unsqueeze(0).float()
    frames = [seed_history[-1]]

    for step in range(steps):
        action_index = min(step, len(policy_actions) - 1)
        action = torch.from_numpy(policy_actions[action_index]).unsqueeze(0).float()
        with torch.inference_mode():
            current_frame = history[:, -1:, :, :]
            current_binary = (current_frame >= 0.5).float()
            logits = model(history, action)
            _, binary, next_input = feedback_from_logits(logits, feedback)
            if static_noop_hold:
                hold_mask = static_noop_mask(history, action, static_history_threshold)
                if bool(hold_mask.any()):
                    predicted_diff = (binary - current_binary).abs().sum(dim=(1, 2, 3))
                    hold_mask = hold_mask & (
                        predicted_diff <= static_predicted_diff_threshold
                    )
                    if bool(hold_mask.any()):
                        hold_mask = hold_mask[:, None, None, None]
                        binary = torch.where(hold_mask, current_binary, binary)
                        next_input = torch.where(hold_mask, current_frame, next_input)
            if paddle_motion_hold:
                paddle_mask = paddle_motion_mask(history, action, paddle_motion_threshold)
                if bool(paddle_mask.any()):
                    shifted_binary, applied_mask = shift_paddle_frames(
                        current_binary,
                        action,
                        shift_pixels=paddle_shift,
                    )
                    paddle_mask = paddle_mask & applied_mask
                    if bool(paddle_mask.any()):
                        paddle_mask = paddle_mask[:, None, None, None]
                        binary = torch.where(paddle_mask, shifted_binary, binary)
                        next_input = torch.where(paddle_mask, shifted_binary, next_input)
        history = torch.cat([history[:, 1:, :, :], next_input], dim=1)
        frames.append(binary[0, 0].cpu().numpy())

    return np.stack(frames)


def sample_frame_indices(total_frames):
    candidates = [0, 1, 2, 4, 8, 16, 32]
    return [idx for idx in candidates if idx < total_frames]


def render_montage(rollouts, output_path):
    any_frames = next(iter(rollouts.values()))
    frame_indices = sample_frame_indices(any_frames.shape[0])
    frame_height, frame_width = any_frames.shape[1:]
    label_width = 56
    header_height = 18
    gap = 4
    canvas_width = label_width + len(frame_indices) * frame_width + (len(frame_indices) - 1) * gap
    canvas_height = header_height + len(rollouts) * frame_height + (len(rollouts) - 1) * gap
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for col, frame_index in enumerate(frame_indices):
        x = label_width + col * (frame_width + gap)
        draw.text((x, 2), f"t={frame_index}", fill=(0, 0, 0))

    for row, (label, frames) in enumerate(rollouts.items()):
        y = header_height + row * (frame_height + gap)
        draw.text((4, y + 2), label, fill=(0, 0, 0))
        for col, frame_index in enumerate(frame_indices):
            x = label_width + col * (frame_width + gap)
            tile = Image.fromarray(
                (frames[frame_index] * 255).astype(np.uint8),
                mode="L",
            ).convert("RGB")
            canvas.paste(tile, (x, y))

    canvas.save(output_path)
    return output_path


def frame_mae(frame_a, frame_b):
    return float(np.abs(frame_a - frame_b).mean())


def main():
    args = parse_args()
    game_config = infer_game_config(dataset_id=args.dataset, game=args.game)
    model = FrameDynamicsModel(
        history_length=args.history_length,
        n_actions=game_config.n_actions,
        refine_blocks=args.pixel_refine_blocks,
    )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True))
    model.eval()

    seed_history = fetch_seed_history(args, game_config)
    policies = build_policies(game_config, game_config.n_actions)
    rollouts = {
        label: rollout_policy(
            model,
            seed_history,
            actions,
            args.steps,
            args.feedback,
            args.pixel_static_noop_hold,
            args.pixel_static_history_threshold,
            args.pixel_static_predicted_diff_threshold,
            args.pixel_paddle_motion_hold,
            args.pixel_paddle_motion_threshold,
            args.pixel_paddle_shift,
        )
        for label, actions in policies.items()
    }

    for label, frames in rollouts.items():
        avg_step_change = np.mean(
            [frame_mae(frames[idx], frames[idx + 1]) for idx in range(frames.shape[0] - 1)]
        )
        drift_from_start = frame_mae(frames[0], frames[-1])
        print(
            f"{label}: avg_step_change={avg_step_change:.6f}, "
            f"drift_from_start={drift_from_start:.6f}"
        )

    labels = list(rollouts)
    base_label = labels[0]
    for other_label in labels[1:]:
        separation = frame_mae(rollouts[base_label][-1], rollouts[other_label][-1])
        print(f"{base_label} vs {other_label}: final_frame_mae={separation:.6f}")

    output_path = render_montage(rollouts, args.output)
    print(output_path)


if __name__ == "__main__":
    main()
