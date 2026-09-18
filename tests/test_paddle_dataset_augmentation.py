import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gymemu import augment_paddle_dataset as augmentation


def test_full_annotation_preserves_rows_and_aligns_both_sides(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    thresholds = [(0, 162), (2000, 100)]
    monkeypatch.setattr(augmentation, "load_thresholds", lambda _: thresholds)
    native = tmp_path / "native.rs"
    native.write_text("test source")
    manifest = {
        "contract": {
            "environment": {
                "env_provider": "env-breakoutatari2600-turbo-native",
                "frame_skip": 2,
                "sticky_action_prob": 0,
                "env_args": {"noop_reset_max": 30, "use_fire_reset": False},
            }
        }
    }
    (source / "manifest.json").write_text(json.dumps(manifest))
    expected = {}
    for split, eid in [("train", 11), ("heldout", 89)]:
        controller = augmentation.initial_controller(eid, thresholds)
        states = [controller.state()]
        rows = []
        for step, action in enumerate([1, 1, 2, 1, 2]):
            for _ in range(1 if step == 4 else 2):
                controller.step(action + 1)
            states.append(controller.state())
            labels = {
                "paddle_x": controller.x * 65536,
                "paddle_vx_normalized": controller.vx / 160,
                "lives": 2 if step < 2 else 1,
                "ball_y": 208 if step == 3 else 0,
            }
            record = json.dumps(
                {
                    "structure": json.dumps(
                        [
                            "dict",
                            [
                                [
                                    "labels",
                                    [
                                        "dict",
                                        [[key, ["scalar", value]] for key, value in labels.items()],
                                    ],
                                ]
                            ],
                        ]
                    )
                }
            )
            rows.append(
                {
                    "episode_id": eid,
                    "step": step,
                    "native_action_json": json.dumps(action),
                    "selected_action_json": "0",
                    "configured_frame_skip": 2,
                    "elapsed_native_frames": None,
                    "record_json": record,
                    "source_frame_id": 900 + step * 3,
                    "keep": "unchanged",
                }
            )
        expected[split] = states
        directory = source / "transitions" / split
        directory.mkdir(parents=True)
        # One episode spans two files; life losses must not reset its controller.
        pq.write_table(pa.Table.from_pylist(rows[:2]), directory / "00000.parquet")
        pq.write_table(pa.Table.from_pylist(rows[2:]), directory / "00001.parquet")
        directory = source / "episodes" / split
        directory.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist([{"episode_id": eid, "seed": eid, "length": 5}]),
            directory / "00000.parquet",
        )
    report = augmentation.augment(source, output, native)
    assert report["status"] == "validated"
    for split in ("train", "heldout"):
        assert report["summary"][split]["transitions"] == 5
        assert report["summary"][split]["last_life_one_frame_stops"] == 1
        rows = pq.read_table(output / "transitions" / split).to_pylist()
        for i, row in enumerate(rows):
            assert [row["source_" + field] for field in augmentation.FIELDS] == expected[split][i]
            assert [row[field] for field in augmentation.FIELDS] == expected[split][i + 1]
            assert row["keep"] == "unchanged"
            assert row["selected_action_json"] == "0"
        episode = pq.read_table(output / "episodes" / split).to_pylist()[0]
        assert [episode["initial_" + f] for f in augmentation.FIELDS] == expected[split][0]
    assert augmentation.VERSION not in json.loads((source / "manifest.json").read_text())
    with pytest.raises(ValueError, match="Output must be new"):
        augmentation.augment(source, output, native)


def test_replay_rejects_missing_rows_and_incomplete_episodes():
    replay = augmentation.EpisodeReplay({7: {"seed": 1, "length": 3}}, [(0, 162)])
    with pytest.raises(ValueError, match="out-of-order"):
        replay.transition({"episode_id": 7, "step": 1})
    with pytest.raises(ValueError, match="Incomplete"):
        replay.finish()


def test_replay_rejects_bad_successor_labels():
    replay = augmentation.EpisodeReplay({7: {"seed": 1, "length": 1}}, [(0, 162)])
    record = json.dumps(
        {
            "structure": json.dumps(
                [
                    "dict",
                    [
                        [
                            "labels",
                            [
                                "dict",
                                [
                                    ["paddle_x", ["scalar", -65536]],
                                    ["paddle_vx_normalized", ["scalar", 0]],
                                ],
                            ],
                        ]
                    ],
                ]
            )
        }
    )
    with pytest.raises(ValueError, match="replay mismatch"):
        replay.transition(
            {
                "episode_id": 7,
                "step": 0,
                "native_action_json": "1",
                "configured_frame_skip": 2,
                "elapsed_native_frames": None,
                "record_json": record,
            }
        )
