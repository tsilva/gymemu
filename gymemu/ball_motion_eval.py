"""Four-variable ball rollouts with explicit reference segment boundaries."""

import numpy as np
import torch


def evaluate_ball_motion(model, data, horizons=(1, 8, 32, 128), modes=("both",)):
    """Start at every eligible source; retain supplied non-ball state fields.

    Exact reference-input predictions can be reused without changing results.
    A window is scored only if its whole horizon stays inside a contiguous life.
    Input rows must already exclude terminal transitions and carry life IDs.
    """
    source, target = data["x"], data["target"]
    n = len(source)
    if not n or target.shape != (n, 4) or min(horizons) < 1:
        raise ValueError("Expected nonempty sources, N-by-4 targets and positive horizons")
    if not modes or set(modes) - {"reference", "horizontal", "vertical", "both"}:
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
            joint_errors=int(wrong[m].any(1).sum()),
        )
        for name, m in masks.items()
    }
    results = {}
    with torch.no_grad():
        for mode in modes:
            feedback = np.zeros((n, 4), np.float32)
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
                    if mode in ("horizontal", "both"):
                        inputs[:, 0] = previous[:, 0]
                        inputs[:, 2] = previous[:, 2]
                    if mode in ("vertical", "both"):
                        inputs[:, 1] = np.floor(previous[:, 1])
                        inputs[:, 8] = np.rint((previous[:, 1] % 1) * 8)
                        inputs[:, 3] = previous[:, 3]
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
        scope="Selected ball fields fed back; all others supplied; reference boundaries",
    )
