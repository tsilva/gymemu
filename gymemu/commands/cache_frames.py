"""Build a reusable lossless frame cache for faster training, without changing pixels."""

import argparse
from pathlib import Path

from gymemu.cache import build_cache
from gymemu.data import resolve_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root, _ = resolve_dataset(args.dataset, args.revision)
    print(build_cache(root, args.output, args.workers))


if __name__ == "__main__":
    main()
