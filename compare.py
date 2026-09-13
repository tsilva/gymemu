"""Compare completed runs with identical held-out RGB evaluation contracts."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def collect_runs(roots):
    paths = set()
    for root in roots:
        root = Path(root)
        if root.is_file():
            paths.add(root)
        elif root.is_dir():
            paths.update(root.rglob("summary.json"))
        else:
            raise ValueError(f"Run path does not exist: {root}")
    groups = {}
    for path in sorted(paths):
        summary = json.loads(path.read_text())
        if summary.get("status") != "complete":
            continue
        key = hashlib.sha256(json.dumps(summary["evaluation"], sort_keys=True).encode()).hexdigest()
        row = {
            "evaluation_group": key[:12],
            "run": str(path.parent),
            **{
                k: summary[k]
                for k in [
                    "name",
                    "approach",
                    "seed",
                    "history",
                    "best_mse",
                    "parameters",
                    "optimizer_steps",
                    "train_samples_seen",
                    "seconds",
                ]
            },
        }
        groups.setdefault(key, []).append(row)
    if not groups:
        raise ValueError("No completed runs with summary.json found")
    return {key: sorted(rows, key=lambda r: r["best_mse"]) for key, rows in groups.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument(
        "--csv", type=Path, help="Write comparison rows for plotting or spreadsheets"
    )
    args = parser.parse_args(argv)
    try:
        groups = collect_runs(args.runs)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    for key, rows in groups.items():
        print(f"Evaluation group {key[:12]} (lower RGB MSE is better)")
        for row in rows:
            print(
                f"  {row['best_mse']:.8f}  {row['approach']} seed={row['seed']} "
                f"history={row['history']} steps={row['optimizer_steps']}  {row['run']}"
            )
    if len(groups) > 1:
        print("Different datasets or evaluated targets are shown in separate groups.")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as stream:
            rows = [row for group in groups.values() for row in group]
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
