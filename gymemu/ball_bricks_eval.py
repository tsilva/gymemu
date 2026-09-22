"""Ball and brick-layout rollouts with explicit reference segment boundaries."""

import numpy as np
import torch


def evaluate_ball_bricks(
    model,
    data,
    horizons=(1, 8, 32, 128),
    modes=("both",),
    *,
    contact=False,
    hit_count=False,
    paddle_width=False,
    paddle_charge=False,
    paddle_position=False,
):
    """Start at every eligible source; retain supplied other state fields.

    Exact reference-input predictions can be reused without changing results.
    A window is scored only if its whole horizon stays inside a contiguous life.
    With contact=True, output column 112 is contact and "both" feeds it back.
    With hit_count=True, output column 113 is count and "both" feeds it back.
    With paddle_width=True, output 114 is width and "both" feeds it back.
    With paddle_charge=True, action is source 118 and charge output 115 feeds
    source 6 in joint mode. Each step uses its recorded action without feedback.
    With paddle_position=True, output 116 feeds source paddle x column 4.
    Input rows must already exclude terminal transitions and carry life IDs.
    """
    source, target = data["x"], data["target"]
    n = len(source)
    if hit_count and not contact:
        raise ValueError("Count feedback requires the contact output contract")
    if paddle_width and not hit_count:
        raise ValueError("Width feedback requires the count output contract")
    if paddle_charge and not paddle_width:
        raise ValueError("Charge feedback requires the width output contract")
    if paddle_position and not paddle_charge:
        raise ValueError("Position feedback requires the charge output contract")
    inputs_count = 119 if paddle_charge else 118
    if source.shape != (n, inputs_count):
        raise ValueError(f"Expected N-by-{inputs_count} source rows")
    outputs = (
        112
        + int(contact)
        + int(hit_count)
        + int(paddle_width)
        + int(paddle_charge)
        + int(paddle_position)
    )
    if not n or target.shape != (n, outputs) or min(horizons) < 1:
        raise ValueError(f"Expected nonempty sources, N-by-{outputs} targets and positive horizons")
    if not modes or set(modes) - {"reference", "ball", "bricks", "both"}:
        raise ValueError("Unknown ball feedback mode")
    continuous = (
        (data["episodes"][1:] == data["episodes"][:-1])
        & (data["steps"][1:] == data["steps"][:-1] + 1)
        & (data["starts"][1:] == data["starts"][:-1])
    )
    remaining = np.ones(n, np.int64)
    for i in range(n - 2, -1, -1):
        if continuous[i]:
            remaining[i] = remaining[i + 1] + 1
    model.eval()
    with torch.no_grad():
        teacher = np.concatenate(
            [
                model.predict(torch.from_numpy(source[i : i + 2048])).cpu().numpy()
                for i in range(0, n, 2048)
            ]
        )
    wrong = teacher != target
    first_cases = [
        dict(
            episode=int(data["episodes"][i]),
            step=int(data["steps"][i]),
            source=source[i, :10].tolist(),
            target=target[i].tolist(),
            predicted=teacher[i].tolist(),
            event=int(data["events"][i]),
        )
        for i in np.flatnonzero(wrong.any(1))
    ]
    masks = {"all": np.ones(n, bool), "ordinary": data["events"] == 0}
    masks.update(
        {
            name: (data["events"] & bit) != 0
            for name, bit in [("side_wall", 1), ("paddle", 2), ("brick", 4), ("ceiling", 8)]
        }
    )
    one_step = {
        name: dict(
            n=int(m.sum()),
            x_errors=int(wrong[m, 0].sum()),
            y_errors=int(wrong[m, 1].sum()),
            vx_errors=int(wrong[m, 2].sum()),
            vy_errors=int(wrong[m, 3].sum()),
            ball_errors=int(wrong[m, :4].any(1).sum()),
            layout_errors=int(wrong[m, 4:112].any(1).sum()),
            brick_cell_errors=int(wrong[m, 4:112].sum()),
            **({"contact_errors": int(wrong[m, 112].sum())} if contact else {}),
            **({"count_errors": int(wrong[m, 113].sum())} if hit_count else {}),
            **({"width_errors": int(wrong[m, 114].sum())} if paddle_width else {}),
            **(
                {
                    "charge_errors": int(wrong[m, 115].sum()),
                    "charge_mae": float(np.abs(teacher[m, 115] - target[m, 115]).mean())
                    if m.any()
                    else None,
                    "invalid_charge_predictions": int(
                        ((teacher[m, 115] < 0) | (teacher[m, 115] >= 3856)).sum()
                    ),
                }
                if paddle_charge
                else {}
            ),
            **(
                {
                    "paddle_x_errors": int(wrong[m, 116].sum()),
                    "paddle_x_mae": float(np.abs(teacher[m, 116] - target[m, 116]).mean())
                    if m.any()
                    else None,
                    "paddle_x_max_error": float(np.abs(teacher[m, 116] - target[m, 116]).max())
                    if m.any()
                    else None,
                }
                if paddle_position
                else {}
            ),
            joint_errors=int(wrong[m].any(1).sum()),
        )
        for name, m in masks.items()
    }
    results = {}
    with torch.no_grad():
        for mode in modes:
            feedback = np.zeros((n, outputs), np.float32)
            first = np.zeros(n, np.int64)
            summary = {}
            for h in range(1, max(horizons) + 1):
                starts = np.flatnonzero(remaining >= h)
                if not len(starts):
                    break
                rows = starts + h - 1
                prediction = teacher[rows].copy()
                if mode != "reference" and h > 1:
                    inputs = source[rows].copy()
                    previous = feedback[starts]
                    if mode in ("ball", "both"):
                        inputs[:, 0] = previous[:, 0]
                        inputs[:, 2] = previous[:, 2]
                    if mode in ("ball", "both"):
                        inputs[:, 1] = np.floor(previous[:, 1])
                        inputs[:, 8] = np.rint((previous[:, 1] % 1) * 8)
                        inputs[:, 3] = previous[:, 3]
                    if mode in ("bricks", "both"):
                        inputs[:, 10:118] = previous[:, 4:112]
                    if contact and mode == "both":
                        inputs[:, 9] = previous[:, 112]
                    if hit_count and mode == "both":
                        inputs[:, 7] = previous[:, 113]
                    if paddle_width and mode == "both":
                        inputs[:, 5] = previous[:, 114]
                    if paddle_charge and mode == "both":
                        inputs[:, 6] = previous[:, 115]
                    if paddle_position and mode == "both":
                        inputs[:, 4] = previous[:, 116]
                    changed = np.flatnonzero(np.any(inputs != source[rows], axis=1))
                    for batch in np.array_split(changed, max(1, (len(changed) + 2047) // 2048)):
                        if len(batch):
                            prediction[batch] = (
                                model.predict(torch.from_numpy(inputs[batch])).cpu().numpy()
                            )
                feedback[starts] = prediction
                errors = prediction != target[rows]
                new = (first[starts] == 0) & errors.any(1)
                first[starts[new]] = h
                if h in horizons:
                    failed = first[starts] > 0
                    first_rows = starts[failed] + first[starts[failed]] - 1
                    summary[str(h)] = dict(
                        windows=len(starts),
                        x_errors=int(errors[:, 0].sum()),
                        y_errors=int(errors[:, 1].sum()),
                        vx_errors=int(errors[:, 2].sum()),
                        vy_errors=int(errors[:, 3].sum()),
                        ball_errors=int(errors[:, :4].any(1).sum()),
                        layout_errors=int(errors[:, 4:112].any(1).sum()),
                        brick_cell_errors=int(errors[:, 4:112].sum()),
                        **({"contact_errors": int(errors[:, 112].sum())} if contact else {}),
                        **({"count_errors": int(errors[:, 113].sum())} if hit_count else {}),
                        **({"width_errors": int(errors[:, 114].sum())} if paddle_width else {}),
                        **(
                            {
                                "charge_errors": int(errors[:, 115].sum()),
                                "charge_mae": float(
                                    np.abs(prediction[:, 115] - target[rows, 115]).mean()
                                ),
                                "charge_max_error": float(
                                    np.abs(prediction[:, 115] - target[rows, 115]).max()
                                ),
                                "invalid_charge_predictions": int(
                                    ((prediction[:, 115] < 0) | (prediction[:, 115] >= 3856)).sum()
                                ),
                            }
                            if paddle_charge
                            else {}
                        ),
                        **(
                            {
                                "paddle_x_errors": int(errors[:, 116].sum()),
                                "paddle_x_mae": float(
                                    np.abs(prediction[:, 116] - target[rows, 116]).mean()
                                ),
                                "paddle_x_max_error": float(
                                    np.abs(prediction[:, 116] - target[rows, 116]).max()
                                ),
                            }
                            if paddle_position
                            else {}
                        ),
                        joint_errors=int(errors.any(1).sum()),
                        endpoint_joint_accuracy=float(1 - errors.any(1).mean()),
                        exact_entire_window=int((~failed).sum()),
                        entire_window_accuracy=float((~failed).mean()),
                        x_mae=float(np.abs(prediction[:, 0] - target[rows, 0]).mean()),
                        x_max_error=float(np.abs(prediction[:, 0] - target[rows, 0]).max()),
                        y_mae=float(np.abs(prediction[:, 1] - target[rows, 1]).mean()),
                        y_max_error=float(np.abs(prediction[:, 1] - target[rows, 1]).max()),
                        median_first_error_step=float(np.median(first[starts[failed]]))
                        if failed.any()
                        else None,
                        unique_first_error_transitions=len(np.unique(first_rows)),
                        first_error_events={
                            name: int(m[first_rows].sum()) for name, m in masks.items()
                        },
                    )
            results[mode] = summary
    return dict(
        sources=n,
        segments=int(1 + (~continuous).sum()),
        one_step=one_step,
        rollouts=results,
        teacher_errors=first_cases,
        scope=(
            "Selected ball/brick fields and optional contact/count/width/charge/paddle x fed back; "
            "others supplied; reference boundaries"
        ),
    )
