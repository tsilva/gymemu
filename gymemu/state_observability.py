"""Exact-input, per-target contradiction audit on the training split only."""

import argparse
import hashlib
from pathlib import Path

import numpy as np

from gymemu.state_data import FIELDS, NATIVE_SCALES, StateWindows, write_json


def fingerprints(data, history, past_actions):
    """Hash only model-visible values. None uses the whole virtual-episode prefix."""
    indices, a = data.indices, data.arrays
    keys = np.empty(len(indices), dtype="V32")
    if history is None:
        previous = None
        digest = b""
        for j, i in enumerate(indices):
            if i == a["starts"][i]:
                digest = b""
            elif previous != i - 1:
                raise ValueError("Full-prefix audit requires ordered, complete segments")
            digest = hashlib.sha256(
                digest + a["states"][i].tobytes() + a["actions"][i].tobytes()
            ).digest()
            keys[j] = digest
            previous = i
    else:
        for offset in range(0, len(indices), 2048):
            batch = data.batch(indices[offset : offset + 2048], history, past_actions)
            hs, hv, aa = (batch[k].numpy() for k in ("history", "history_valid", "action_history"))
            for j, (state, valid, actions) in enumerate(zip(hs, hv, aa, strict=True)):
                keys[offset + j] = hashlib.sha256(
                    state.tobytes() + valid.tobytes() + actions.tobytes()
                ).digest()
    return keys


def contradictions(keys, target, terminal, identities, *, example_limit=2):
    """Empirical contradictions, not a population Bayes-error estimate.

    State targets on terminal transitions are undefined and excluded per field.
    Nonterminal/terminal disagreement remains visible in the terminal audit.
    """
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind="stable")
    offsets = np.r_[0, counts.cumsum()]
    names = [*FIELDS, "brick_grid", "terminal"]
    fields = {
        name: {"conflicting_groups": 0, "examples_in_conflicting_groups": 0, "examples": []}
        for name in names
    }
    square_floor = np.zeros(7)
    spread = np.zeros(7)
    classification_floor = {"brick_grid": 0, "terminal": 0}
    scales = np.r_[NATIVE_SCALES, 16.0]
    for group in np.flatnonzero(counts > 1):
        ix = order[offsets[group] : offsets[group + 1]]
        live = ix[~terminal[ix]]
        candidates = {}
        if len(live) > 1:
            values = target[live, :7].astype(np.float64) * scales
            square_floor += np.square(values - values.mean(0)).sum(0)
            spread = np.maximum(spread, np.ptp(values, axis=0))
            for k, name in enumerate(FIELDS):
                if np.ptp(values[:, k]) > 0:
                    candidates[name] = (
                        live,
                        int(live[np.argmin(values[:, k])]),
                        int(live[np.argmax(values[:, k])]),
                    )
            bricks = target[live, 7:]
            classification_floor["brick_grid"] += int(
                np.minimum(bricks.sum(0), len(live) - bricks.sum(0)).sum()
            )
            differs = np.any(bricks != bricks[0], axis=1)
            if differs.any():
                candidates["brick_grid"] = (
                    live,
                    int(live[0]),
                    int(live[np.flatnonzero(differs)[0]]),
                )
        positives = int(terminal[ix].sum())
        classification_floor["terminal"] += min(positives, len(ix) - positives)
        if 0 < positives < len(ix):
            candidates["terminal"] = (ix, int(ix[~terminal[ix]][0]), int(ix[terminal[ix]][0]))
        for name, (members, first, second) in candidates.items():
            row = fields[name]
            row["conflicting_groups"] += 1
            row["examples_in_conflicting_groups"] += len(members)
            if len(row["examples"]) < example_limit:
                if name in FIELDS:
                    k = FIELDS.index(name)
                    outcomes = (target[[first, second], k] * scales[k]).tolist()
                elif name == "terminal":
                    outcomes = terminal[[first, second]].tolist()
                else:
                    outcomes = target[[first, second], 7:].astype(int).tolist()
                row["examples"].append(
                    {"episode_and_step": identities[[first, second]].tolist(), "targets": outcomes}
                )
    live_count = int((~terminal).sum())
    for k, name in enumerate(FIELDS):
        fields[name].update(
            max_conflicting_spread_native=float(spread[k]),
            empirical_mse_floor_native=float(square_floor[k] / max(1, live_count)),
        )
    for name, errors in classification_floor.items():
        fields[name]["minimum_empirical_classification_errors"] = errors
    return {
        "samples": len(keys),
        "nonterminal_samples": live_count,
        "unique_inputs": len(counts),
        "repeated_input_groups": int((counts > 1).sum()),
        "examples_in_repeated_groups": int(counts[counts > 1].sum()),
        "fields": fields,
    }


def audit(cache, output, contexts):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    data = StateWindows(cache, "train")
    a, ii = data.arrays, data.indices
    target = np.array(a["states"][ii + 1])
    terminal = np.array(a["terminal"][ii])
    identities = np.column_stack((a["episode_ids"][ii], a["steps"][ii]))
    result = {
        "cache_identity": data.manifest["identity"],
        "split": "train",
        "interpretation": (
            "A conflict disproves exact prediction from these inputs. "
            "No observed conflict does not prove sufficiency."
        ),
        "contexts": [],
    }
    for history, past in contexts:
        keys = fingerprints(data, history, past)
        row = contradictions(keys, target, terminal, identities)
        row.update(history=history, past_actions=past)
        if history is not None:
            full = ii - a["starts"][ii] >= max(history - 1, past)
            row["full_context_only"] = contradictions(
                keys[full], target[full], terminal[full], identities[full]
            )
        result["contexts"].append(row)
        write_json(output / "audit.json", result)
        print(
            history,
            past,
            {k: v["conflicting_groups"] for k, v in row["fields"].items()},
            flush=True,
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", default="1:7,8:7,16:15,32:31,full")
    args = parser.parse_args(argv)
    contexts = [
        (None, None) if s == "full" else tuple(map(int, s.split(":")))
        for s in args.contexts.split(",")
    ]
    audit(args.cache, args.output, contexts)


if __name__ == "__main__":
    main()
