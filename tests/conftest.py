import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image


def write(root, kind, split, rows):
    path = root / kind / split / "00000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


@pytest.fixture
def snapshot(tmp_path):
    root = tmp_path / "dataset"
    frames = []
    for frame_id in [30, 10, 70, 20, 40, 60, 50]:  # IDs deliberately differ from row offsets.
        buffer = io.BytesIO()
        Image.fromarray(np.full((21, 17, 3), frame_id, dtype=np.uint8)).save(
            buffer, format="WEBP", lossless=True
        )
        frames.append({"frame_id": frame_id, "image": {"bytes": buffer.getvalue(), "path": None}})
    write(root, "frames", "assets", frames)
    for split, episodes in [
        ("train", [(1, [10, 20, 30], [2, 0]), (3, [70, 10], [0])]),
        ("heldout", [(2, [40, 50, 60], [0, 2])]),
    ]:
        metadata, steps = [], []
        for eid, ids, actions in episodes:
            metadata.append({"episode_id": eid, "initial_frame_id": ids[0], "length": len(actions)})
            for step, action in enumerate(actions):
                steps.append(
                    {
                        "episode_id": eid,
                        "step": step,
                        "source_frame_id": ids[step],
                        "successor_frame_id": ids[step + 1],
                        "native_action_json": str(action),
                    }
                )
        write(root, "episodes", split, metadata)
        write(root, "transitions", split, list(reversed(steps)))
    return root
