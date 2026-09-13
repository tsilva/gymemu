import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from gymemu.cache import build_cache
from gymemu.data import Frames, Windows, read_episodes


def test_lossless_cache_preserves_pixels_windows_actions_and_padding(snapshot, tmp_path):
    cache = tmp_path / "cache"
    build_cache(snapshot, cache, workers=1)
    original = Frames(snapshot, compact=True)
    cached = Frames(snapshot, compact=True, cache=cache)
    for frame_id in original.ids:
        assert torch.equal(original.get(int(frame_id)), cached.get(int(frame_id)))
    episodes = read_episodes(snapshot, "train")
    left = Windows(original, episodes, 4, [0, 2])
    right = Windows(cached, episodes, 4, [0, 2])
    for index in range(len(left)):
        a, b = left[index], right[index]
        assert a[1] == b[1]
        assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])
    assert np.array_equal(original.ids, cached.ids)


def test_cache_rejects_changed_source_or_corrupt_shard(snapshot, tmp_path):
    cache = tmp_path / "cache"
    build_cache(snapshot, cache, workers=1)
    shard = next(cache.glob("*.arrow"))
    contents = shard.read_bytes()
    shard.write_bytes(contents[:-8] + b"corrupt!")
    with pytest.raises(ValueError, match="checksum"):
        Frames(snapshot, compact=True, cache=cache)
    shard.write_bytes(contents)
    source = next((snapshot / "frames/assets").glob("*.parquet"))
    table = pq.read_table(source)
    pq.write_table(table, source, compression="NONE")
    with pytest.raises(ValueError, match="source"):
        Frames(snapshot, compact=True, cache=cache)


def test_cached_frames_reopen_in_loader_workers(snapshot, tmp_path):
    import pickle

    from torch.utils.data import DataLoader

    cache = tmp_path / "cache"
    build_cache(snapshot, cache, workers=1)
    frames = Frames(snapshot, compact=True, cache=cache)
    assert len(pickle.dumps(frames.images)) < 4096
    windows = Windows(frames, read_episodes(snapshot, "train"), 4, [0, 2])
    loader = DataLoader(windows, batch_size=2, num_workers=1, multiprocessing_context="spawn")
    history, action, target = next(iter(loader))
    assert torch.equal(history[0], windows[0][0])
    assert torch.equal(target[1], windows[1][2])
    assert action.tolist() == [windows[0][1], windows[1][1]]


def test_direct_batches_match_standard_loader(snapshot, tmp_path):
    from torch.utils.data import DataLoader

    from gymemu.batches import CachedBatchLoader

    cache = tmp_path / "cache"
    build_cache(snapshot, cache, workers=1)
    frames = Frames(snapshot, compact=True, cache=cache)
    windows = Windows(frames, read_episodes(snapshot, "train"), 4, [0, 2])
    order = [4, 0, 2, 1, 3]
    expected = list(DataLoader(windows, batch_size=2, sampler=order))
    loader = CachedBatchLoader(windows, batch_size=2, workers=2, sampler=order)
    assert len(loader) == len(expected)
    for _ in range(2):
        for left, right in zip(expected, loader, strict=True):
            assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))
