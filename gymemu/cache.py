"""Verified lossless LZ4 frame cache; source Parquet files remain unchanged."""

import hashlib
import io
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_files(root):
    files = sorted((Path(root) / "frames/assets").glob("*.parquet"))
    if not files:
        raise ValueError("No source frame Parquet files")
    return files


def _convert(task):
    source, destination = map(Path, task)
    data = pq.read_table(source, columns=["frame_id", "image"])
    compressed = []
    shape = None
    for value in data["image"]:
        image = value.as_py()
        with Image.open(io.BytesIO(image["bytes"])) as decoded:
            if decoded.mode != "RGB":
                raise ValueError("Expected RGB images")
            pixels = np.asarray(decoded).transpose(2, 0, 1).copy()
        if shape is not None and tuple(pixels.shape) != shape:
            raise ValueError("All images must have the same dimensions")
        shape = tuple(pixels.shape)
        raw = pixels.tobytes()
        packed = pa.compress(raw, codec="lz4", asbytes=True)
        if pa.decompress(packed, len(raw), codec="lz4", asbytes=True) != raw:
            raise ValueError("Lossless cache roundtrip failed")
        compressed.append(packed)
    if shape is None:
        raise ValueError("Empty source frame shard")
    table = pa.table(
        {"frame_id": data["frame_id"], "image": pa.array(compressed, type=pa.binary())}
    )
    temporary = destination.with_suffix(".tmp")
    with pa.OSFile(str(temporary), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    temporary.replace(destination)
    return {
        "name": destination.name,
        "sha256": file_hash(destination),
        "rows": len(table),
        "shape": list(shape),
        "bytes": destination.stat().st_size,
    }


def build_cache(root, output, workers=6):
    root, output = Path(root), Path(output)
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be positive")
    files = source_files(root)
    signatures = [{"name": p.name, "sha256": file_hash(p)} for p in files]
    output.mkdir(parents=True, exist_ok=False)
    tasks = [(str(p), str(output / f"{i:05d}.arrow")) for i, p in enumerate(files)]
    started = time.monotonic()
    shards = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for index, result in enumerate(pool.map(_convert, tasks), 1):
            shards.append(result)
            if index % 10 == 0 or index == len(files):
                print(
                    f"cache shards={index}/{len(files)} frames={sum(s['rows'] for s in shards)} "
                    f"seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )
    shapes = {tuple(s["shape"]) for s in shards}
    if len(shapes) != 1:
        raise ValueError("All images must have the same dimensions")
    # A changing dataset cannot produce a valid cache receipt.
    if signatures != [{"name": p.name, "sha256": file_hash(p)} for p in source_files(root)]:
        raise ValueError("Frame source changed during cache construction")
    manifest = {
        "format_version": 1,
        "codec": "lz4",
        "layout": "CHW",
        "shape": list(shapes.pop()),
        "sources": signatures,
        "shards": shards,
        "seconds": time.monotonic() - started,
    }
    temporary = output / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(output / "manifest.json")
    return output


class CachedImages:
    """Reopen memory maps after worker spawn instead of pickling gigabytes of frame bytes."""

    def __init__(self, cache):
        self.cache = Path(cache)
        self._data = None
        self._maps = []

    def __getstate__(self):
        return {"cache": self.cache, "_data": None, "_maps": []}

    def _open(self):
        if self._data is None:
            manifest = json.loads((self.cache / "manifest.json").read_text())
            tables = []
            for shard in manifest["shards"]:
                mmap = pa.memory_map(str(self.cache / shard["name"]))
                self._maps.append(mmap)
                tables.append(pa.ipc.open_file(mmap).read_all())
            self._data = pa.concat_tables(tables)
        return self._data

    def __getitem__(self, index):
        return self._open()["image"][index]


def open_cache(root, cache, ids):
    cache = Path(cache)
    manifest = json.loads((cache / "manifest.json").read_text())
    if (
        manifest.get("format_version") != 1
        or manifest.get("codec") != "lz4"
        or manifest.get("layout") != "CHW"
    ):
        raise ValueError("Unsupported frame cache format")
    shape = manifest.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or shape[0] != 3
        or any(type(n) is not int or n < 1 for n in shape)
    ):
        raise ValueError("Unsupported frame cache format")
    signatures = [{"name": p.name, "sha256": file_hash(p)} for p in source_files(root)]
    if signatures != manifest["sources"]:
        raise ValueError("Frame cache does not match its source files")
    for shard in manifest["shards"]:
        if Path(shard["name"]).name != shard["name"]:
            raise ValueError("Invalid frame cache shard path")
        if file_hash(cache / shard["name"]) != shard["sha256"]:
            raise ValueError("Frame cache checksum mismatch")
    images = CachedImages(cache)
    if not np.array_equal(images._open()["frame_id"].to_numpy(), ids):
        raise ValueError("Frame cache IDs differ from source IDs")
    return images, tuple(manifest["shape"])
