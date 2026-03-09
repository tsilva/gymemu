from __future__ import annotations

import re

STACKED_DATASET_PATTERN = re.compile(r"_stack(?P<history_length>\d+)(?:_|$)")


def infer_history_length(dataset_id: str | None, explicit_history_length: int | None = None) -> int:
    if explicit_history_length is not None:
        if explicit_history_length < 1:
            raise ValueError("history_length must be at least 1")
        return explicit_history_length

    if dataset_id:
        match = STACKED_DATASET_PATTERN.search(dataset_id)
        if match:
            return int(match.group("history_length"))

    return 1
