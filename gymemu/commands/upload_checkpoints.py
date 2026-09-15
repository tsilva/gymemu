"""Retry R2 publication for an existing Gymemu run without retraining."""

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from gymemu.storage import CheckpointStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Local run directory containing resolved.yaml")
    args = parser.parse_args(argv)
    output = args.run.expanduser().resolve()
    config = OmegaConf.to_container(OmegaConf.load(output / "resolved.yaml"), resolve=True)
    if not config.get("r2", {}).get("enabled", False):
        parser.error("This run did not enable R2 uploads")
    store = CheckpointStore(config, output)
    store.sync(final=(output / "summary.json").is_file())
    if store.state["status"] == "pending":
        raise SystemExit(
            "R2 upload is still pending; inspect credentials and connectivity, then retry"
        )
    print(f"Uploaded run manifest: {store.uri}")


if __name__ == "__main__":
    main()
