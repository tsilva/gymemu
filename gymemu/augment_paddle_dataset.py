"""Add deterministic paddle controller states to every recorded transition."""

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gymemu.cache import file_hash
from gymemu.state_data import decode_labels, write_json
from gymemu.state_hidden_context import (
    NATIVE_COMMIT,
    PaddleController,
    controller_native_frames,
)

VERSION = "breakout-paddle-controller-v1"
FIELDS = ("paddle_charge", "paddle_measure", "paddle_repeat", "paddle_held")
SOURCE_SHA256 = "a2a47d8ec876657697cb2113ddcb5545bfd8ed3d2c394649423759247213e4c7"


def load_thresholds(native_source):
    if file_hash(native_source) != SOURCE_SHA256:
        raise ValueError("Expected the pinned native source snapshot")
    block = Path(native_source).read_text().split("const PADDLE_MEASURE_THRESHOLDS:")[1]
    values = [
        tuple(map(int, p)) for p in re.findall(r"\((\d+),\s*(\d+)\)", block.split("];", 1)[0])
    ]
    if len(values) != 89:
        raise ValueError("Unexpected controller threshold table")
    return values


def initial_controller(seed, thresholds):
    controller = PaddleController(thresholds)
    noops = int(np.random.default_rng(seed).integers(1, 31, dtype=np.uint64))
    for _ in range(noops):
        controller.step(0)
    return controller


def add_values(table, values, prefix=""):
    values = np.asarray(values, dtype=np.int32).reshape(-1, 4)
    for j, field in enumerate(FIELDS):
        name = prefix + field
        if name in table.column_names:
            raise ValueError(f"Refusing to overwrite {name}")
        dtype = pa.bool_() if field == "paddle_held" else pa.int32()
        column = values[:, j].astype(bool) if field == "paddle_held" else values[:, j]
        table = table.append_column(name, pa.array(column, type=dtype))
    return table


class EpisodeReplay:
    """Keep an independent controller and continuity check for each original episode."""

    def __init__(self, metadata, thresholds):
        self.metadata = metadata
        self.controllers = {
            eid: initial_controller(row["seed"], thresholds) for eid, row in metadata.items()
        }
        self.steps = dict.fromkeys(metadata, 0)
        self.previous = dict.fromkeys(metadata)
        self.initial = {eid: controller.state() for eid, controller in self.controllers.items()}
        self.one_frame_stops = 0

    def transition(self, row):
        eid, step = row["episode_id"], row["step"]
        if eid not in self.controllers or step != self.steps[eid]:
            raise ValueError(f"Missing, duplicate, or out-of-order transition {eid}:{step}")
        if row["configured_frame_skip"] != 2:
            raise ValueError("Only frameskip 2 is supported")
        action = json.loads(row["native_action_json"])
        if type(action) is not int or action not in (0, 1, 2):
            raise ValueError("Unexpected executed action")
        controller = self.controllers[eid]
        before = controller.state()
        frames = controller_native_frames(self.previous[eid])
        if row["elapsed_native_frames"] not in (None, frames):
            raise ValueError(f"Native frame count mismatch at {eid}:{step}")
        for _ in range(frames):
            controller.step(action + 1)
        labels = decode_labels(row["record_json"])
        if controller.x != labels["paddle_x"] / 65536 or controller.vx != round(
            labels["paddle_vx_normalized"] * 160
        ):
            raise ValueError(f"Controller replay mismatch at {eid}:{step}")
        self.one_frame_stops += int(frames == 1)
        self.steps[eid] += 1
        self.previous[eid] = labels
        return before, controller.state()

    def finish(self):
        for eid, row in self.metadata.items():
            if self.steps[eid] != row["length"]:
                raise ValueError(f"Incomplete episode {eid}")


def write_checked(original, annotated, source, destination, root):
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(annotated, destination, compression="zstd", row_group_size=8192)
    reloaded = pq.read_table(destination)
    if not reloaded.equals(annotated) or not reloaded.select(original.column_names).equals(
        original
    ):
        raise ValueError(f"Parquet roundtrip changed values: {destination}")
    return {
        "relative": str(destination.relative_to(root)),
        "rows": len(original),
        "original_sha256": file_hash(source),
        "sha256": file_hash(destination),
    }


def augment(dataset, output, native_source, *, frames=None):
    dataset, output, native_source = map(Path, (dataset, output, native_source))
    dataset, output = dataset.resolve(), output.resolve()
    staging = output.with_name(output.name + ".incomplete")
    if output.exists() or staging.exists() or output.is_relative_to(dataset):
        raise ValueError("Output must be new and outside the source dataset")
    thresholds = load_thresholds(native_source)
    manifest = json.loads((dataset / "manifest.json").read_text())
    environment = manifest["contract"]["environment"]
    if (
        environment["env_provider"] != "env-breakoutatari2600-turbo-native"
        or environment["frame_skip"] != 2
        or environment["sticky_action_prob"] != 0
        or environment["env_args"]["noop_reset_max"] != 30
        or environment["env_args"]["use_fire_reset"]
    ):
        raise ValueError("Unsupported collection contract")
    if VERSION in manifest.get("annotations", {}):
        raise ValueError("Dataset already has paddle annotations")
    frame_root = Path(frames).resolve() if frames else dataset / "frames"
    if frames and not (frame_root / "assets").is_dir():
        raise ValueError("Expected a frames directory containing assets")
    staging.mkdir(parents=True)
    started = time.monotonic()
    receipts, summary = [], {}
    source_manifest_hash = file_hash(dataset / "manifest.json")
    if (dataset / "annotations").exists():
        shutil.copytree(dataset / "annotations", staging / "annotations")
    if frame_root.is_dir():
        (staging / "frames").symlink_to(frame_root, target_is_directory=True)
    for split in ("train", "heldout"):
        episode_paths = sorted((dataset / "episodes" / split).glob("*.parquet"))
        transition_paths = sorted((dataset / "transitions" / split).glob("*.parquet"))
        if not episode_paths or not transition_paths:
            raise ValueError(f"Missing split {split}")
        metadata = {}
        for path in episode_paths:
            for row in pq.read_table(path, columns=["episode_id", "seed", "length"]).to_pylist():
                if row["episode_id"] in metadata:
                    raise ValueError("Duplicate episode metadata")
                metadata[row["episode_id"]] = row
        replay = EpisodeReplay(metadata, thresholds)
        rows = 0
        for index, path in enumerate(transition_paths, 1):
            original = pq.read_table(path)
            columns = [
                "episode_id",
                "step",
                "configured_frame_skip",
                "elapsed_native_frames",
                "native_action_json",
                "record_json",
            ]
            pairs = [replay.transition(row) for row in original.select(columns).to_pylist()]
            annotated = add_values(original, [p[0] for p in pairs], "source_")
            annotated = add_values(annotated, [p[1] for p in pairs])
            receipts.append(
                write_checked(
                    original, annotated, path, staging / path.relative_to(dataset), staging
                )
            )
            rows += len(original)
            if index % 25 == 0 or index == len(transition_paths):
                print(
                    f"{split}: {index}/{len(transition_paths)} shards, {rows:,} verified rows, "
                    f"{time.monotonic() - started:.1f}s",
                    flush=True,
                )
        replay.finish()
        for path in episode_paths:
            original = pq.read_table(path)
            annotated = add_values(
                original,
                [replay.initial[eid] for eid in original["episode_id"].to_pylist()],
                "initial_",
            )
            receipts.append(
                write_checked(
                    original, annotated, path, staging / path.relative_to(dataset), staging
                )
            )
        summary[split] = {
            "transitions": rows,
            "episodes": len(metadata),
            "controller_mismatches": 0,
            "last_life_one_frame_stops": replay.one_frame_stops,
        }
    # Verify source files did not change while the build ran.
    for receipt in receipts:
        if file_hash(dataset / receipt["relative"]) != receipt["original_sha256"]:
            raise ValueError("Source dataset changed during annotation")
    if file_hash(dataset / "manifest.json") != source_manifest_hash:
        raise ValueError("Source manifest changed during annotation")
    schema = {
        "version": VERSION,
        "fields": list(FIELDS),
        "types": ["int32", "int32", "int32", "bool"],
        "transition_timing": {
            "source_": "Before the row's action; inputs use only reset metadata and past actions",
            "unprefixed": "After the row's action; aligned with record_json successor labels",
        },
        "episode_timing": "initial_ fields describe the state after reset no-ops, before step 0",
        "normalization_divisors": [3856, 235, 60, 1],
        "life_boundary": "Persists across life loss; resets only at original episode reset",
        "actions": "Executed native_action_json; custom discrete 0/1/2 maps to native 1/2/3",
        "validation": "Every successor paddle position and velocity matches the original labels",
        "native_frames": "2, except 1 at final-life loss determined from preceding source labels",
        "excluded": ["paddle_hits"],
    }
    annotation_root = staging / "annotations" / VERSION
    annotation_root.mkdir(parents=True)
    report = {
        "version": VERSION,
        "status": "validated",
        "summary": summary,
        "source_dataset": str(dataset),
        "source_manifest_sha256": source_manifest_hash,
        "native_commit": NATIVE_COMMIT,
        "native_source_sha256": SOURCE_SHA256,
        "frames": str(frame_root.resolve()) if frame_root.is_dir() else None,
        "seconds": time.monotonic() - started,
        "prior_annotation_receipts": "Historical input receipts; current file hashes are here",
    }
    for source in (
        Path(__file__),
        Path(__file__).with_name("state_hidden_context.py"),
        native_source,
    ):
        shutil.copyfile(source, annotation_root / source.name)
    report["code_sha256"] = {p.name: file_hash(p) for p in annotation_root.iterdir() if p.is_file()}
    write_json(annotation_root / "schema.json", schema)
    write_json(annotation_root / "validation.json", report)
    write_json(annotation_root / "files.json", receipts)
    for name in ("manifest.json", "dataset-summary.json"):
        value = json.loads((dataset / name).read_text()) if (dataset / name).exists() else {}
        value.setdefault("annotations", {})[VERSION] = {
            "schema": schema,
            "summary": summary,
            "receipts": f"annotations/{VERSION}",
        }
        write_json(staging / name, value)
    card = (dataset / "README.md").read_text() if (dataset / "README.md").exists() else ""
    card += (
        "\n\n## Paddle controller annotations\n\n"
        "Four exact integer/boolean controller values are included in the tables: "
        "`paddle_charge`, `paddle_measure`, `paddle_repeat`, `paddle_held`. "
        "Unprefixed transition columns describe the successor; `source_` columns describe "
        "the input state. Episode `initial_` columns describe the state after reset no-ops. "
        "The controller persists across life loss. Reconstruction uses the pinned native "
        "controller, episode seeds, and recorded executed actions. Every recorded successor "
        "paddle position and velocity was checked for exact agreement. Existing columns and "
        "split memberships are unchanged. Paddle-hit counts are not included. "
        f"See `annotations/{VERSION}/` for timing, provenance, checksums, and validation. "
        "Earlier annotation receipts describe their historical input files. "
        "The local frames directory, when present, links to existing unchanged image assets.\n"
    )
    (staging / "README.md").write_text(card)
    staging.rename(output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument(
        "--frames", type=Path, help="Existing frames directory to link without copying"
    )
    args = parser.parse_args()
    print(
        json.dumps(
            augment(args.dataset, args.output, args.native_source, frames=args.frames), indent=2
        )
    )


if __name__ == "__main__":
    main()
