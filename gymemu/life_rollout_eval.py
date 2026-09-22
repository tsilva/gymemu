"""Score learned stopping without crossing lives or scoring respawn states."""

import numpy as np
import torch


def _feedback(source, previous):
    result = source.copy()
    result[:, 0] = previous[:, 0]
    result[:, 1] = np.floor(previous[:, 1])
    result[:, 8] = np.rint(previous[:, 1] % 1 * 8)
    result[:, 2:4] = previous[:, 2:4]
    result[:, 10:118] = previous[:, 4:112]
    result[:, 9] = previous[:, 112]
    result[:, 7] = previous[:, 113]
    result[:, 5] = previous[:, 114]
    result[:, 6] = previous[:, 115]
    result[:, 4] = previous[:, 116]
    return result


def evaluate_life_rollouts(model, data, horizons=(1, 8, 32, 128), *, full_segments=False):
    """Feed all state back and stop inference at the first predicted stop.

    Bounded windows start at every source. Death-ended windows shorter than the
    horizon are included; short censored windows are excluded. Full-segment mode
    starts once per segment and includes censored segments. If a model misses a
    recorded death, record that miss and censor there: later stop timing cannot
    be measured without crossing into a different life. Terminal successor state
    is ignored, even when its recorded fields contain a respawn or NaNs.
    """
    source, target = data["x"], data["target"]
    terminal = np.asarray(data["terminal"])
    n = len(source)
    if not n or source.shape != (n, 119) or target.shape != (n, 117):
        raise ValueError("Expected nonempty N-by-119 sources and N-by-117 state targets")
    if terminal.shape != (n,) or terminal.dtype != np.bool_:
        raise ValueError("Expected one boolean terminal label per source")
    if not np.isfinite(target[~terminal]).all():
        raise ValueError("Nonterminal state targets must be finite")
    if not horizons or min(horizons) < 1:
        raise ValueError("Expected positive horizons")
    continuous = (
        (data["episodes"][1:] == data["episodes"][:-1])
        & (data["steps"][1:] == data["steps"][:-1] + 1)
        & (data["starts"][1:] == data["starts"][:-1])
    )
    if np.any(terminal[:-1] & continuous):
        raise ValueError("A terminal transition must end its life segment")
    remaining = np.ones(n, np.int64)
    for i in range(n - 2, -1, -1):
        if continuous[i]:
            remaining[i] = remaining[i + 1] + 1
    roots = np.r_[0, np.flatnonzero(~continuous) + 1] if full_segments else np.arange(n)
    lengths = remaining[roots]
    ends_in_death = terminal[roots + lengths - 1]
    model.eval()
    with torch.no_grad():
        teacher = np.concatenate(
            [
                model.predict(torch.from_numpy(source[i : i + 2048])).cpu().numpy()
                for i in range(0, n, 2048)
            ]
        )
    if teacher.shape != (n, 118) or not np.isin(teacher[:, 117], [0, 1]).all():
        raise ValueError("Expected 117 state predictions and a binary stop flag")
    predicted_stop = teacher[:, 117].astype(bool)
    comparable = ~terminal & ~predicted_stop
    wrong = np.zeros(n, bool)
    wrong[comparable] = (teacher[comparable, :117] != target[comparable]).any(1)
    joint_wrong = wrong | (predicted_stop != terminal)
    one_step = dict(
        n=n,
        deaths=int(terminal.sum()),
        true_positives=int((predicted_stop & terminal).sum()),
        false_positives=int((predicted_stop & ~terminal).sum()),
        false_negatives=int((~predicted_stop & terminal).sum()),
        true_negatives=int((~predicted_stop & ~terminal).sum()),
        state_compared=int(comparable.sum()),
        nonterminal_state_errors=int(wrong.sum()),
        joint_errors=int(joint_wrong.sum()),
        exact_accuracy=float(1 - joint_wrong.mean()),
    )
    windows = len(roots)
    feedback = np.zeros((windows, 117), np.float32)
    alive = np.ones(windows, bool)
    failed = np.zeros(windows, bool)
    first_stop = np.zeros(windows, np.int64)
    state_errors = np.zeros(windows, np.int64)
    calls = np.zeros(windows, np.int64)
    stop_horizon = int(lengths.max()) if full_segments else max(horizons)
    summaries = {}
    with torch.no_grad():
        for h in range(1, stop_horizon + 1):
            active = np.flatnonzero(alive & (lengths >= h))
            rows = roots[active] + h - 1
            prediction = teacher[rows].copy()
            if h > 1 and len(active):
                inputs = _feedback(source[rows], feedback[active])
                changed = np.flatnonzero(np.any(inputs != source[rows], axis=1))
                for batch in np.array_split(changed, max(1, (len(changed) + 2047) // 2048)):
                    if len(batch):
                        prediction[batch] = (
                            model.predict(torch.from_numpy(inputs[batch])).cpu().numpy()
                        )
            stopped = prediction[:, 117].astype(bool)
            comparable = ~terminal[rows] & ~stopped
            wrong = np.zeros(len(rows), bool)
            wrong[comparable] = (prediction[comparable, :117] != target[rows[comparable]]).any(1)
            failed[active] |= wrong | (stopped != terminal[rows])
            state_errors[active] += wrong
            calls[active] += 1
            first_stop[active[stopped]] = h
            alive[active[stopped]] = False
            feedback[active[~stopped]] = prediction[~stopped, :117]
            if (not full_segments and h in horizons) or (full_segments and h == stop_horizon):
                eligible = (
                    np.ones(windows, bool) if full_segments else (lengths >= h) | ends_in_death
                )
                expected_death = ends_in_death & (lengths <= h)
                has_stop = first_stop > 0
                exact_death = expected_death & (first_stop == lengths)
                early = expected_death & has_stop & (first_stop < lengths)
                missed = expected_death & ~has_stop
                false_survival = ~expected_death & has_stop
                count = int(eligible.sum())
                summaries["segment_end" if full_segments else str(h)] = dict(
                    windows=count,
                    death_windows=int((eligible & expected_death).sum()),
                    survival_or_censored_windows=int((eligible & ~expected_death).sum()),
                    exact_death_step=int((eligible & exact_death).sum()),
                    premature_death_stop=int((eligible & early).sum()),
                    missed_death=int((eligible & missed).sum()),
                    false_stop_without_reference_death=int((eligible & false_survival).sum()),
                    stop_timing_errors=int((eligible & (early | missed | false_survival)).sum()),
                    exact_entire_window=int((eligible & ~failed).sum()),
                    entire_window_accuracy=float((~failed[eligible]).mean()) if count else None,
                    state_transition_errors=int(state_errors[eligible].sum()),
                    simulated_transitions=int(calls[eligible].sum()),
                    reference_end_censored_misses=int((eligible & missed).sum()),
                )
    return dict(
        one_step=one_step,
        rollouts=summaries,
        segments=int(1 + (~continuous).sum()),
        full_segments=full_segments,
        teacher_error_sources=[
            dict(
                episode=int(data["episodes"][i]),
                step=int(data["steps"][i]),
                terminal=bool(terminal[i]),
                predicted_stop=bool(predicted_stop[i]),
            )
            for i in np.flatnonzero(joint_wrong)
        ],
        scope=(
            "All state fed back; recorded actions; predicted stops halt inference. "
            "Missed deaths censored at reference end; no respawn state is used."
        ),
    )
