"""Retain audited version floors for dependencies used by the current baseline."""

from importlib.metadata import version

import pytest


@pytest.mark.parametrize(
    ("distribution", "minimum"),
    [
        ("click", (8, 3, 3)),
        ("idna", (3, 15)),
        ("pillow", (12, 3, 0)),
        ("pygments", (2, 20, 0)),
        ("torch", (2, 13, 0)),
    ],
)
def test_audited_dependency_floors(distribution: str, minimum: tuple[int, ...]) -> None:
    release = version(distribution).split("+", maxsplit=1)[0]
    numeric = tuple(int(part) for part in release.split(".") if part.isdigit())
    assert numeric >= minimum
