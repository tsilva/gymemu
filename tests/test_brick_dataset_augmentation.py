import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from gymemu.brick_labels import COLORS
from gymemu.cache import build_cache


def test_additive_dataset_augmentation_alignment_and_initial_flags(tmp_path):
    root, cache, output = (tmp_path / name for name in ("source", "cache", "output"))
    full = np.zeros((210, 160, 3), dtype=np.uint8)
    for row, color in enumerate(COLORS):
        full[57 + row * 6 : 63 + row * 6, 8:152] = color
    partial = full.copy()
    partial[57:93, 8:144] = 0
    damaged = full.copy()
    damaged[87:93, 8:16] = 0
    images = []
    # IDs deliberately differ from storage offsets and storage order.
    for fid, rgb in [(700, damaged), (100, partial), (300, full)]:
        stream = io.BytesIO()
        Image.fromarray(rgb).save(stream, format="PNG")
        images.append(dict(frame_id=fid, image=dict(bytes=stream.getvalue(), path=None)))
    (root / "frames/assets").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(images), root / "frames/assets/00000.parquet")
    for split, eid, ids, counts in [
        ("train", 7, [100, 300, 700], [108, 107]),
        ("heldout", 25, [300, 300], [108]),
    ]:
        transitions = []
        for step, (source, target, count) in enumerate(zip(ids[:-1], ids[1:], counts, strict=True)):
            labels = [
                ("bricks_remaining", ["scalar", count]),
                ("walls_cleared", ["scalar", 0]),
                ("score", ["scalar", 108 - count]),
            ]
            record = json.dumps(
                dict(structure=json.dumps(["dict", [["labels", ["dict", labels]]]]))
            )
            transitions.append(
                dict(
                    episode_id=eid,
                    step=step,
                    source_frame_id=source,
                    successor_frame_id=target,
                    record_json=record,
                    keep_me="original",
                )
            )
        (root / "transitions" / split).mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(transitions), root / "transitions" / split / "00000.parquet"
        )
        (root / "episodes" / split).mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist(
                [dict(episode_id=eid, initial_frame_id=ids[0], length=len(counts))]
            ),
            root / "episodes" / split / "00000.parquet",
        )
    build_cache(root, cache, workers=1)
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "augment_brick_dataset.py"),
            "--dataset",
            str(root),
            "--cache",
            str(cache),
            "--output",
            str(output),
            "--workers",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    rows = pq.read_table(output / "dataset/transitions/train/00000.parquet").to_pylist()
    assert rows[0]["source_is_initial_brick_layout"]
    assert not rows[0]["is_initial_brick_layout"]
    assert not rows[1]["source_is_initial_brick_layout"]
    assert rows[1]["brick_grid"][5][0] == 0
    assert not rows[1]["brick_grid_suspect"]
    assert rows[1]["keep_me"] == "original"
    initial = pq.read_table(output / "dataset/episodes/train/00000.parquet").to_pylist()[0]
    assert initial["initial_is_brick_layout"]
    assert initial["initial_brick_count_mismatch"] is None
    assert initial["initial_brick_grid_suspect"]
