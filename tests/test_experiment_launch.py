"""dstack placement may not change the resolved training recipe."""

import pytest

from gymemu.commands.experiment import render_task


def test_queued_task_preserves_recipe_and_bounds_paid_compute():
    image = "ghcr.io/tsilva/gymemu/train@sha256:" + "a" * 64
    recipe = "name: tiny\nrun_id: abc\n"
    local = render_task(
        recipe, image=image, run_id="a" * 32, compute="local",
        max_duration="1h", max_price=None,
    )
    spot = render_task(
        recipe, image=image, run_id="a" * 32, compute="spot",
        max_duration="1h", max_price=2.0,
    )
    assert local["image"] == spot["image"] == image
    assert local["commands"] == spot["commands"]
    assert local["fleets"] == ["b3"]
    assert spot["spot_policy"] == "spot" and spot["max_price"] == 2.0
    assert spot["max_duration"] == "1h"
    with pytest.raises(ValueError, match="price"):
        render_task(
            recipe, image=image, run_id="a" * 32, compute="on-demand",
            max_duration="1h", max_price=None,
        )
