import torch


def feedback_from_logits(logits: torch.Tensor, mode: str):
    probs = torch.sigmoid(logits)
    binary = (probs >= 0.5).float()

    if mode == "soft":
        next_input = probs
    elif mode == "hard":
        next_input = binary
    elif mode == "ste":
        next_input = probs + (binary - probs).detach()
    else:
        raise ValueError(f"Unsupported pixel feedback mode '{mode}'")

    return probs, binary, next_input


def static_noop_mask(
    history_frames: torch.Tensor,
    action: torch.Tensor,
    history_motion_threshold: float = 0.0,
):
    if history_frames.size(1) < 2:
        history_motion = torch.zeros(history_frames.size(0), device=history_frames.device)
    else:
        history_motion = (
            history_frames[:, 1:, :, :] - history_frames[:, :-1, :, :]
        ).abs().sum(dim=(1, 2, 3))
    noop_action = action.argmax(dim=1) == 0
    return noop_action & (history_motion <= history_motion_threshold)


def paddle_motion_mask(
    history_frames: torch.Tensor,
    action: torch.Tensor,
    outside_motion_threshold: float = 4.0,
    paddle_top: int = 83,
    paddle_bottom: int = 88,
):
    if history_frames.size(1) < 2:
        outside_motion = torch.zeros(history_frames.size(0), device=history_frames.device)
    else:
        diffs = (history_frames[:, 1:, :, :] - history_frames[:, :-1, :, :]).abs().clone()
        diffs[:, :, paddle_top:paddle_bottom, :] = 0
        outside_motion = diffs.sum(dim=(1, 2, 3))
    action_ids = action.argmax(dim=1)
    horizontal_action = (action_ids == 2) | (action_ids == 3)
    return horizontal_action & (outside_motion <= outside_motion_threshold)


def _find_row_runs(row: torch.Tensor):
    cols = torch.where(row > 0.5)[0]
    if cols.numel() == 0:
        return []

    runs = []
    start = int(cols[0].item())
    prev = start
    for col in cols[1:]:
        current = int(col.item())
        if current == prev + 1:
            prev = current
            continue
        runs.append((start, prev))
        start = current
        prev = current
    runs.append((start, prev))
    return runs


def shift_paddle_frames(
    current_binary: torch.Tensor,
    action: torch.Tensor,
    shift_pixels: int = 2,
    paddle_top: int = 83,
    paddle_bottom: int = 88,
    playfield_left: int = 4,
    playfield_right: int = 75,
    min_width: int = 4,
    max_width: int = 14,
):
    shifted = current_binary.clone()
    applied_mask = torch.zeros(
        current_binary.size(0),
        dtype=torch.bool,
        device=current_binary.device,
    )

    for batch_index in range(current_binary.size(0)):
        action_id = int(action[batch_index].argmax().item())
        if action_id == 2:
            delta = shift_pixels
        elif action_id == 3:
            delta = -shift_pixels
        else:
            continue

        frame = current_binary[batch_index, 0].clone()
        candidates = []
        for y in range(paddle_top, min(paddle_bottom, frame.size(0))):
            for start, end in _find_row_runs(frame[y]):
                width = end - start + 1
                if playfield_left <= start and end <= playfield_right:
                    if min_width <= width <= max_width:
                        candidates.append((y, start, end))
                    continue

                # When the paddle reaches a wall, it can merge into the wall run.
                if start < playfield_left <= end:
                    split_start = playfield_left
                    split_end = end
                    split_width = split_end - split_start + 1
                    if min_width <= split_width <= max_width:
                        candidates.append((y, split_start, split_end))

                if start <= playfield_right < end:
                    split_start = start
                    split_end = playfield_right
                    split_width = split_end - split_start + 1
                    if min_width <= split_width <= max_width:
                        candidates.append((y, split_start, split_end))

        if not candidates:
            continue

        candidates.sort(key=lambda item: (item[2] - item[1] + 1, item[0]), reverse=True)
        ref_y, ref_start, ref_end = candidates[0]
        ref_center = (ref_start + ref_end) / 2.0

        paddle_rows = []
        for y, start, end in candidates:
            center = (start + end) / 2.0
            if abs(center - ref_center) <= 2.0:
                paddle_rows.append((y, start, end))

        if not paddle_rows:
            continue

        width = round(sum(end - start + 1 for _, start, end in paddle_rows) / len(paddle_rows))
        base_start = round(sum(start for _, start, _ in paddle_rows) / len(paddle_rows))
        new_start = min(max(base_start + delta, playfield_left), playfield_right - width + 1)
        new_end = new_start + width - 1

        for y, start, end in paddle_rows:
            frame[y, start : end + 1] = 0
            frame[y, new_start : new_end + 1] = 1

        shifted[batch_index, 0] = frame
        applied_mask[batch_index] = True

    return shifted, applied_mask
