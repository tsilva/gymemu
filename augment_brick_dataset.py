"""Build and validate additive Breakout brick annotations before Hub publication.

All outputs go to a separate directory. Run with the source revision and its
verified frame cache. Scratch arrays and completed shards support restart.
"""

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from gymemu import brick_labels
from gymemu.brick_labels import Issue, episode_issues, extract_wall_batch
from gymemu.cache import file_hash
from gymemu.data import Frames

ROOT = None


def atomic_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def array(name, mode="r"):
    return np.load(ROOT / "scratch" / f"{name}.npy", mmap_mode=mode)


def initialize(root):
    global ROOT
    ROOT = Path(root)
    pa.set_cpu_count(1)


def extract_job(job):
    receipt = ROOT / "scratch" / f"frame-{job['index']:04d}.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    started = time.monotonic()
    with pa.memory_map(job["source"]) as mmap:
        table = pa.ipc.open_file(mmap).read_all()
        grids, support, flags = (array(k, "r+") for k in ("grids", "support", "visual_flags"))
        begin = job["offset"]
        for first in range(0, len(table), 256):
            batch = table.slice(first, 256)
            walls = []
            for value in batch["image"]:
                decoded = pa.decompress(value.as_buffer(), 3 * 210 * 160, codec="lz4")
                rgb = np.frombuffer(decoded, dtype=np.uint8).reshape(3, 210, 160)
                walls.append(rgb[:, 57:93, 8:152].transpose(1, 2, 0))
            g, s, f = extract_wall_batch(np.stack(walls))
            target = slice(begin + first, begin + first + len(batch))
            grids[target], support[target], flags[target] = g, s, f
        for value in (grids, support, flags):
            value.flush()
    result = dict(index=job["index"], rows=job["rows"], seconds=time.monotonic() - started)
    atomic_json(receipt, result)
    return result


def metadata_job(job):
    receipt = ROOT / "scratch" / f"metadata-{job['index']:04d}.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    columns = ["episode_id", "step", "source_frame_id", "successor_frame_id", "record_json"]
    table = pq.read_table(job["source"], columns=columns)
    data = array("transitions", "r+")
    part = data[job["offset"] : job["offset"] + job["rows"]]
    for key in columns[:-1]:
        part[key] = table[key].to_numpy()
    part["split"] = job["split"]
    for i, value in enumerate(table["record_json"]):
        tree = dict(json.loads(json.loads(value.as_py())["structure"])[1])
        labels = dict(tree["labels"][1])
        for field in ("bricks_remaining", "walls_cleared", "score"):
            part[field][i] = labels[field][1]
    data.flush()
    result = dict(index=job["index"], rows=job["rows"])
    atomic_json(receipt, result)
    return result


def matrix_array(values):
    flat = pa.array(np.asarray(values).reshape(-1), type=pa.int8())
    return pa.FixedSizeListArray.from_arrays(pa.FixedSizeListArray.from_arrays(flat, 18), 6)


def add_columns(table, frame_indices, flags, initial, native_counts=None, prefix=""):
    grids = np.asarray(array("grids")[frame_indices])
    supports = np.asarray(array("support")[frame_indices])
    present = (grids == 1).sum((1, 2)).astype(np.uint8)
    unknown = (grids == -1).sum((1, 2)).astype(np.uint8)
    values = {
        "brick_grid": matrix_array(grids),
        "brick_grid_suspect": pa.array(flags != 0),
        "brick_grid_quality_flags": pa.array(flags, type=pa.uint16()),
        "brick_count_visible": pa.array(present),
        "brick_grid_unknown_cells": pa.array(unknown),
        "brick_grid_min_present_support": pa.array(
            np.where(grids == 1, supports, 48).min((1, 2)).astype(np.uint8)
        ),
        "brick_count_mismatch": (
            pa.array([None] * len(table), type=pa.bool_())
            if native_counts is None
            else pa.array(present != native_counts)
        ),
        "is_initial_brick_layout": pa.array(initial),
    }
    for name, value in values.items():
        column = prefix + name
        if prefix and name == "is_initial_brick_layout":
            column = prefix + "is_brick_layout"
        if column in table.column_names:
            raise ValueError(f"Refusing to overwrite an existing annotation: {column}")
        table = table.append_column(column, value)
    return table


def write_job(job):
    destination = ROOT / "dataset" / job["relative"]
    receipt = ROOT / "scratch" / f"write-{job['index']:04d}.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if file_hash(destination) == saved["sha256"]:
            return saved
        raise ValueError(f"Completed shard changed: {destination}")
    original = pq.read_table(job["source"])
    target = slice(job["offset"], job["offset"] + job["rows"])
    metadata = array("transitions")[target]
    table = add_columns(
        original,
        array("target_indices")[target],
        array("issues")[target],
        array("initial")[target],
        metadata["bricks_remaining"],
    )
    table = table.append_column(
        "source_is_initial_brick_layout", pa.array(array("source_initial")[target])
    )
    table = table.append_column(
        "source_brick_grid_suspect", pa.array(array("source_issues")[target] != 0)
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    pq.write_table(table, temp, compression="zstd", row_group_size=8192)
    reloaded = pq.read_table(temp)
    if not reloaded.select(original.column_names).equals(original):
        raise ValueError(f"Original columns changed during roundtrip: {job['relative']}")
    if not reloaded.equals(table):
        raise ValueError(f"Annotation roundtrip failed: {job['relative']}")
    temp.replace(destination)
    result = dict(
        relative=job["relative"],
        rows=len(table),
        sha256=file_hash(destination),
        original_sha256=file_hash(job["source"]),
    )
    atomic_json(receipt, result)
    return result


def parallel(stage, function, jobs, workers):
    total = sum(j["rows"] for j in jobs)
    completed = 0
    started = time.monotonic()
    results = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize,
        initargs=(str(ROOT),),
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        for i, result in enumerate(pool.map(function, jobs), 1):
            results.append(result)
            completed += result["rows"]
            status = dict(
                stage=stage,
                jobs_done=i,
                jobs_total=len(jobs),
                rows_done=completed,
                rows_total=total,
                seconds=time.monotonic() - started,
            )
            atomic_json(ROOT / "progress.json", status)
            if i % 10 == 0 or i == len(jobs):
                print(json.dumps(status), flush=True)
    return results


def create_array(name, dtype, shape):
    path = ROOT / "scratch" / f"{name}.npy"
    if not path.exists():
        out = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
        out.flush()


def annotate(root, jobs):
    if (ROOT / "scratch" / "annotation-summary.json").exists():
        return json.loads((ROOT / "scratch" / "annotation-summary.json").read_text())
    data = array("transitions")
    if set(data["episode_id"][data["split"] == 0]) & set(data["episode_id"][data["split"] == 1]):
        raise ValueError("Train/held-out episode overlap")
    if np.any((data["bricks_remaining"] < 0) | (data["bricks_remaining"] > 108)):
        raise ValueError("Native brick counts outside expected [0,108]")
    frame_ids = array("frame_ids")
    order = np.argsort(frame_ids)
    sorted_ids = frame_ids[order]

    def lookup(ids):
        positions = np.searchsorted(sorted_ids, ids)
        if np.any(positions == len(sorted_ids)) or not np.array_equal(sorted_ids[positions], ids):
            raise ValueError("Unknown frame ID")
        return order[positions]

    target_indices = array("target_indices", "r+")
    target_indices[:] = lookup(data["successor_frame_id"])
    grids, visual = array("grids"), array("visual_flags")
    issues, source_issues = array("issues", "r+"), array("source_issues", "r+")
    initial, source_initial = array("initial", "r+"), array("source_initial", "r+")
    episode_order = np.lexsort((data["step"], data["episode_id"], data["split"]))
    ordered = data[episode_order]
    boundaries = (
        np.flatnonzero((np.diff(ordered["episode_id"]) != 0) | (np.diff(ordered["split"]) != 0)) + 1
    )
    episode_values = {}
    for indices in np.split(episode_order, boundaries):
        rows = data[indices]
        if not np.array_equal(rows["step"], np.arange(len(rows))):
            raise ValueError("Episode steps must be unique, contiguous, and start at zero")
        if not np.array_equal(rows["source_frame_id"][1:], rows["successor_frame_id"][:-1]):
            raise ValueError("Broken episode frame chain")
        frames = np.r_[lookup(rows["source_frame_id"][:1]), target_indices[indices]]
        flags, layout = episode_issues(
            grids[frames],
            visual[frames],
            rows["bricks_remaining"],
            rows["walls_cleared"],
            rows["score"],
        )
        issues[indices], source_issues[indices] = flags[1:], flags[:-1]
        initial[indices], source_initial[indices] = layout[1:], layout[:-1]
        key = (int(rows["split"][0]), int(rows["episode_id"][0]))
        episode_values[key] = (
            int(frames[0]),
            int(flags[0]),
            bool(layout[0]),
            len(rows),
            int(frame_ids[frames[0]]),
        )
    for values in (target_indices, issues, source_issues, initial, source_initial):
        values.flush()
    for split_index, split in enumerate(("train", "heldout")):
        for path in sorted((root / "episodes" / split).glob("*.parquet")):
            table = pq.read_table(path)
            values = [
                episode_values[(split_index, int(eid))] for eid in table["episode_id"].to_pylist()
            ]
            if [v[3] for v in values] != table["length"].to_pylist():
                raise ValueError("Episode lengths differ from transition chains")
            if [v[4] for v in values] != table["initial_frame_id"].to_pylist():
                raise ValueError("Episode initial frame ID mismatch")
            annotated = add_columns(
                table,
                np.array([v[0] for v in values]),
                np.array([v[1] for v in values], dtype=np.uint16),
                np.array([v[2] for v in values]),
                prefix="initial_",
            )
            destination = ROOT / "dataset" / path.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(annotated, destination, compression="zstd")
            checked = pq.read_table(destination)
            if not checked.select(table.column_names).equals(table) or not checked.equals(
                annotated
            ):
                raise ValueError("Episode annotation roundtrip failed")
    summary = {}
    for split_index, split in enumerate(("train", "heldout")):
        mask = data["split"] == split_index
        values = issues[mask]
        summary[split] = dict(
            transitions=int(mask.sum()),
            suspect=int((values != 0).sum()),
            initial_layout=int(initial[mask].sum()),
            source_initial_layout=int(source_initial[mask].sum()),
            issues={issue.name: int(((values & issue.value) != 0).sum()) for issue in Issue},
        )
    summary["episodes"] = len(episode_values)
    summary["initial_frames_layout"] = sum(v[2] for v in episode_values.values())
    summary["initial_frames_issues"] = {
        issue.name: sum(bool(v[1] & issue.value) for v in episode_values.values())
        for issue in Issue
    }
    atomic_json(ROOT / "scratch" / "annotation-summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--through", choices=["extract", "metadata", "annotate", "write"], default="write"
    )
    args = parser.parse_args()
    initialize(args.output.resolve())
    (ROOT / "scratch").mkdir(parents=True, exist_ok=True)
    source_hashes = {p.name: file_hash(p) for p in (Path(__file__), Path(brick_labels.__file__))}
    identity = dict(
        source_dataset=str(args.dataset.resolve()),
        source_revision=args.dataset.name,
        version=brick_labels.VERSION,
        code=source_hashes,
    )
    identity_path = ROOT / "build-identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Source/code changed. Use a new output directory.")
    atomic_json(identity_path, identity)
    frame_manifest = json.loads((args.cache / "manifest.json").read_text())
    frame_jobs, offset = [], 0
    for i, shard in enumerate(frame_manifest["shards"]):
        frame_jobs.append(
            dict(
                index=i,
                rows=shard["rows"],
                offset=offset,
                source=str((args.cache / shard["name"]).resolve()),
            )
        )
        offset += shard["rows"]
    create_array("grids", np.int8, (offset, 6, 18))
    create_array("support", np.uint8, (offset, 6, 18))
    create_array("visual_flags", np.uint16, (offset,))
    create_array("frame_ids", np.int64, (offset,))
    # Verifies original image files, all cache hashes, and exact frame-ID order.
    frames = Frames(args.dataset, compact=True, cache=args.cache)
    array("frame_ids", "r+")[:] = frames.images._open()["frame_id"].to_numpy()
    del frames
    parallel("extract_unique_frames", extract_job, frame_jobs, args.workers)
    if args.through == "extract":
        return
    jobs, offset = [], 0
    for split_index, split in enumerate(("train", "heldout")):
        for path in sorted((args.dataset / "transitions" / split).glob("*.parquet")):
            count = pq.ParquetFile(path).metadata.num_rows
            jobs.append(
                dict(
                    index=len(jobs),
                    split=split_index,
                    rows=count,
                    offset=offset,
                    source=str(path.resolve()),
                    relative=str(path.relative_to(args.dataset)),
                )
            )
            offset += count
    dtype = np.dtype(
        [
            (k, "i8")
            for k in (
                "episode_id",
                "step",
                "source_frame_id",
                "successor_frame_id",
                "bricks_remaining",
                "walls_cleared",
                "score",
            )
        ]
        + [("split", "i1")]
    )
    create_array("transitions", dtype, (offset,))
    for name, dtype in [
        ("target_indices", np.int64),
        ("issues", np.uint16),
        ("source_issues", np.uint16),
        ("initial", bool),
        ("source_initial", bool),
    ]:
        create_array(name, dtype, (offset,))
    parallel("read_native_labels", metadata_job, jobs, args.workers)
    if args.through == "metadata":
        return
    summary = annotate(args.dataset, jobs)
    print(json.dumps(summary, indent=2), flush=True)
    if args.through == "annotate":
        return
    written = parallel("write_validate_transitions", write_job, jobs, args.workers)
    annotation_dir = ROOT / "dataset" / "annotations" / brick_labels.VERSION
    annotation_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(annotation_dir / "summary.json", summary)
    atomic_json(annotation_dir / "files.json", written)
    atomic_json(annotation_dir / "provenance.json", identity)
    atomic_json(ROOT / "progress.json", dict(stage="validated", summary=summary))


if __name__ == "__main__":
    main()
