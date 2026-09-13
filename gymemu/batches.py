"""Bounded in-process batch loading for losslessly cached frames."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler


class CachedBatchLoader:
    """Fill pinned batches directly, avoiding worker IPC and a second host copy."""

    def __init__(
        self, dataset, batch_size, workers=2, shuffle=False, pin_memory=False, sampler=None
    ):
        if not dataset.frames.cached or not dataset.frames.compact:
            raise ValueError("Cached batches require compact, cached frames")
        self.dataset = dataset
        self.batch_size = batch_size
        self.workers = max(1, workers)
        self.pin_memory = pin_memory
        self.sampler = (
            sampler
            if sampler is not None
            else (RandomSampler(dataset) if shuffle else SequentialSampler(dataset))
        )

    def __len__(self):
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def _batch(self, indices):
        data = self.dataset
        frames = data.frames
        history = torch.empty(
            (len(indices), data.input_history, *frames.shape),
            dtype=torch.uint8,
            pin_memory=self.pin_memory,
        )
        targets = torch.empty(
            (len(indices), *frames.shape), dtype=torch.uint8, pin_memory=self.pin_memory
        )
        action_shape = (len(indices), *data.action_shape)
        actions = torch.empty(action_shape, dtype=torch.long, pin_memory=self.pin_memory)
        # NumPy copies release the GIL and do not dispatch tiny PyTorch operations.
        history_np, targets_np, actions_np = history.numpy(), targets.numpy(), actions.numpy()
        frame_ids = []
        destinations = []
        for row, index in enumerate(indices):
            if index < 0 or index >= len(data):
                raise IndexError(index)
            number = int(np.searchsorted(data.ends, index, side="right"))
            episode = data.episodes[number]
            position = index - (int(data.ends[number - 1]) if number else 0)
            ids = episode.frames[max(0, position - data.input_history) : position]
            padding = data.input_history - len(ids)
            history_np[row, :padding] = 0
            for column, frame_id in enumerate(ids, start=padding):
                frame_ids.append(frame_id)
                destinations.append(history_np[row, column])
            frame_ids.append(episode.frames[position])
            destinations.append(targets_np[row])
            actions_np[row] = data.action_at(episode, position)
        # Resolve actual IDs in one vectorized lookup; IDs are never array offsets.
        positions = frames.order[np.searchsorted(frames.ids, frame_ids)]
        images = frames.images._open()["image"]
        size = int(np.prod(frames.shape))
        for position, destination in zip(positions, destinations):
            raw = pa.decompress(images[int(position)].as_buffer(), size, codec="lz4")
            destination[:] = np.frombuffer(raw, dtype=np.uint8).reshape(frames.shape)
        return history, actions, targets

    def __iter__(self):
        batches = iter(BatchSampler(self.sampler, self.batch_size, drop_last=False))
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            pending = deque(
                pool.submit(self._batch, ids) for ids in islice(batches, self.workers * 2)
            )
            while pending:
                batch = pending.popleft().result()
                ids = next(batches, None)
                if ids is not None:
                    pending.append(pool.submit(self._batch, ids))
                yield batch
