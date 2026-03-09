from dataclasses import dataclass

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


def _paddle_candidates(
    frame: torch.Tensor,
    paddle_top: int = 83,
    paddle_bottom: int = 88,
    playfield_left: int = 4,
    playfield_right: int = 75,
    min_width: int = 4,
    max_width: int = 14,
):
    candidates = []
    for y in range(paddle_top, min(paddle_bottom, frame.size(0))):
        for start, end in _find_row_runs(frame[y]):
            width = end - start + 1
            if playfield_left <= start and end <= playfield_right:
                if min_width <= width <= max_width:
                    candidates.append((y, start, end))
                continue

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
    return candidates


def detect_paddle_span(
    frame: torch.Tensor,
    paddle_top: int = 83,
    paddle_bottom: int = 88,
    playfield_left: int = 4,
    playfield_right: int = 75,
    min_width: int = 4,
    max_width: int = 14,
):
    candidates = _paddle_candidates(
        frame,
        paddle_top=paddle_top,
        paddle_bottom=paddle_bottom,
        playfield_left=playfield_left,
        playfield_right=playfield_right,
        min_width=min_width,
        max_width=max_width,
    )
    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[2] - item[1] + 1, item[0]), reverse=True)
    ref_y, ref_start, ref_end = candidates[0]
    ref_center = (ref_start + ref_end) / 2.0

    paddle_rows = []
    for y, start, end in candidates:
        center = (start + end) / 2.0
        if abs(center - ref_center) <= 2.0:
            paddle_rows.append((y, start, end))

    if not paddle_rows:
        return None

    width = round(sum(end - start + 1 for _, start, end in paddle_rows) / len(paddle_rows))
    base_start = round(sum(start for _, start, _ in paddle_rows) / len(paddle_rows))
    return {
        "rows": paddle_rows,
        "start": base_start,
        "end": base_start + width - 1,
        "center": base_start + (width - 1) / 2.0,
    }


@dataclass
class BreakoutBallState:
    x: int
    y: int
    vx: int
    vy: int
    attached: bool


def detect_ball_position(
    frame: torch.Tensor,
    playfield_left: int = 4,
    playfield_right: int = 75,
    paddle_top: int = 83,
    max_area: int = 6,
):
    visited = torch.zeros_like(frame, dtype=torch.bool)
    height, width = frame.shape
    best = None

    for y in range(height):
        for x in range(width):
            if frame[y, x] <= 0.5 or bool(visited[y, x]):
                continue
            stack = [(y, x)]
            visited[y, x] = True
            points = []
            while stack:
                cy, cx = stack.pop()
                points.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width:
                        if frame[ny, nx] > 0.5 and not bool(visited[ny, nx]):
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            area = len(points)
            xs = [px for _, px in points]
            ys = [py for py, _ in points]
            if area > max_area:
                continue
            if max(xs) < playfield_left + 1 or min(xs) > playfield_right - 1:
                continue
            if min(ys) >= paddle_top:
                continue
            best = {
                "x": round(sum(xs) / len(xs)),
                "y": round(sum(ys) / len(ys)),
                "area": area,
            }
            return best

    return best


def clear_ball_like_components(
    frame: torch.Tensor,
    playfield_left: int = 4,
    playfield_right: int = 75,
    paddle_top: int = 83,
    max_area: int = 6,
):
    out = frame.clone()
    visited = torch.zeros_like(frame, dtype=torch.bool)
    height, width = frame.shape

    for y in range(height):
        for x in range(width):
            if frame[y, x] <= 0.5 or bool(visited[y, x]):
                continue
            stack = [(y, x)]
            visited[y, x] = True
            points = []
            while stack:
                cy, cx = stack.pop()
                points.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width:
                        if frame[ny, nx] > 0.5 and not bool(visited[ny, nx]):
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            area = len(points)
            xs = [px for _, px in points]
            ys = [py for py, _ in points]
            if area > max_area:
                continue
            if max(xs) < playfield_left + 1 or min(xs) > playfield_right - 1:
                continue
            if min(ys) >= paddle_top:
                continue
            for py, px in points:
                out[py, px] = 0.0

    return out


def init_breakout_ball_state(
    frame: torch.Tensor,
    launch_vx: int = 1,
    launch_vy: int = -1,
):
    ball = detect_ball_position(frame)
    if ball is not None:
        return BreakoutBallState(
            x=ball["x"],
            y=ball["y"],
            vx=launch_vx,
            vy=launch_vy,
            attached=False,
        )

    paddle = detect_paddle_span(frame)
    if paddle is None:
        return None

    return BreakoutBallState(
        x=round(paddle["center"]),
        y=min(y for y, _, _ in paddle["rows"]) - 1,
        vx=launch_vx,
        vy=launch_vy,
        attached=True,
    )


def advance_breakout_ball_state(
    ball_state: BreakoutBallState | None,
    action_id: int,
    frame: torch.Tensor,
    launch_action_id: int = 1,
    launch_vx: int = 1,
    launch_vy: int = -1,
    left_wall: int = 4,
    right_wall: int = 75,
    top_wall: int = 0,
    bottom_wall: int = 95,
):
    paddle = detect_paddle_span(frame)
    if ball_state is None:
        return init_breakout_ball_state(
            frame,
            launch_vx=launch_vx,
            launch_vy=launch_vy,
        )

    if ball_state.attached:
        if paddle is not None:
            ball_state.x = round(paddle["center"])
            ball_state.y = min(y for y, _, _ in paddle["rows"]) - 1
        if action_id != launch_action_id:
            return ball_state
        ball_state.attached = False
        ball_state.vx = launch_vx
        ball_state.vy = launch_vy

    next_x = ball_state.x + ball_state.vx
    next_y = ball_state.y + ball_state.vy

    if next_x < left_wall:
        next_x = left_wall
        ball_state.vx = abs(ball_state.vx)
    elif next_x > right_wall:
        next_x = right_wall
        ball_state.vx = -abs(ball_state.vx)

    if next_y < top_wall:
        next_y = top_wall
        ball_state.vy = abs(ball_state.vy)

    if paddle is not None and ball_state.vy > 0:
        paddle_top = min(y for y, _, _ in paddle["rows"])
        if next_y >= paddle_top - 1 and paddle["start"] - 1 <= next_x <= paddle["end"] + 1:
            next_y = paddle_top - 1
            ball_state.vy = -abs(ball_state.vy)
            offset = next_x - paddle["center"]
            if offset > 1:
                ball_state.vx = abs(ball_state.vx)
            elif offset < -1:
                ball_state.vx = -abs(ball_state.vx)

    if next_y > bottom_wall:
        if paddle is not None:
            ball_state.attached = True
            ball_state.x = round(paddle["center"])
            ball_state.y = min(y for y, _, _ in paddle["rows"]) - 1
            ball_state.vx = launch_vx
            ball_state.vy = launch_vy
            return ball_state
        return None

    ball_state.x = next_x
    ball_state.y = next_y
    return ball_state


def overlay_breakout_ball(frame: torch.Tensor, ball_state: BreakoutBallState | None):
    if ball_state is None:
        return frame

    out = frame.clone()
    for dy in (0, 1):
        y = ball_state.y + dy
        if 0 <= y < out.size(0) and 0 <= ball_state.x < out.size(1):
            out[y, ball_state.x] = 1.0
    return out


def erase_breakout_ball(frame: torch.Tensor, ball_state: BreakoutBallState | None):
    if ball_state is None:
        return frame

    out = frame.clone()
    for dy in (0, 1):
        y = ball_state.y + dy
        if 0 <= y < out.size(0) and 0 <= ball_state.x < out.size(1):
            out[y, ball_state.x] = 0.0
    return out


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
        paddle = detect_paddle_span(
            frame,
            paddle_top=paddle_top,
            paddle_bottom=paddle_bottom,
            playfield_left=playfield_left,
            playfield_right=playfield_right,
            min_width=min_width,
            max_width=max_width,
        )
        if paddle is None:
            continue

        width = paddle["end"] - paddle["start"] + 1
        new_start = min(max(paddle["start"] + delta, playfield_left), playfield_right - width + 1)
        new_end = new_start + width - 1

        for y, start, end in paddle["rows"]:
            frame[y, start : end + 1] = 0
            frame[y, new_start : new_end + 1] = 1

        shifted[batch_index, 0] = frame
        applied_mask[batch_index] = True

    return shifted, applied_mask
