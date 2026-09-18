"""Breakout brick-label prototype. Writes diagnostics, never modifies source data.

Calibrate geometry on one training image, then freeze it for both splits. The
extractor sees RGB only. Recorded counts are independent checks, not corrections.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

from gymemu.data import Frames


def candidates(root, per_shard, heldout_offset=0):
    """Consecutive prefixes sample every shard and preserve transition identity."""
    result = []
    columns = ["episode_id", "step", "source_frame_id", "successor_frame_id", "record_json"]
    for split in ("train", "heldout"):
        for path in sorted((root / "transitions" / split).glob("*.parquet")):
            offset = heldout_offset if split == "heldout" else 0
            batch = next(
                pq.ParquetFile(path).iter_batches(batch_size=per_shard + offset, columns=columns)
            ).slice(offset, per_shard)
            for row in batch.to_pylist():
                tree = dict(json.loads(json.loads(row.pop("record_json"))["structure"])[1])
                row["labels"] = {key: value[1] for key, value in tree["labels"][1]}
                row.update(split=split, shard=path.name)
                result.append(row)
    return result


def calibrate(rgb):
    """Find six broad color bands in a mostly intact, HUD-masked training frame.

    Eighteen fixed-width brick columns are a declared Breakout-specific assumption.
    No held-out images or recorded counts are used to set extraction thresholds.
    """
    bands = []
    for y in range(rgb.shape[0] // 2):
        colors, counts = np.unique(rgb[y], axis=0, return_counts=True)
        chromatic = colors.max(1) != colors.min(1)
        counts = counts * chromatic
        best = counts.argmax()
        if counts[best] < rgb.shape[1] // 2:
            continue
        color = colors[best].tolist()
        xs = np.flatnonzero(np.all(rgb[y] == color, axis=1))
        if bands and bands[-1]["color"] == color and bands[-1]["y1"] == y:
            bands[-1]["y1"] = y + 1
            bands[-1]["x0"] = min(bands[-1]["x0"], int(xs.min()))
            bands[-1]["x1"] = max(bands[-1]["x1"], int(xs.max()) + 1)
        else:
            bands.append(dict(y0=y, y1=y + 1, x0=int(xs.min()), x1=int(xs.max()) + 1, color=color))
    assert len(bands) == 6, bands
    x0, x1 = min(b["x0"] for b in bands), max(b["x1"] for b in bands)
    assert (x1 - x0) % 18 == 0
    return dict(
        bands=bands,
        x0=x0,
        x1=x1,
        columns=18,
        present_fraction=0.75,
        absent_fraction=0.05,
        small_sprite_width=2,
        small_sprite_height=4,
    )


def extract(rgb, geometry):
    """Return visible occupancy {-1 unknown, 0 absent, 1 present} and RGB support.

    Support is a matching-pixel fraction, not a calibrated probability. This labels
    what is rendered, which need not equal internal state during wall animation.
    """
    width = (geometry["x1"] - geometry["x0"]) // geometry["columns"]
    support = []
    for band in geometry["bands"]:
        crop = rgb[band["y0"] : band["y1"], geometry["x0"] : geometry["x1"]]
        matching = np.all(crop == band["color"], axis=-1)
        support.append(matching.reshape(len(crop), geometry["columns"], width).mean((0, 2)))
    support = np.array(support, dtype=np.float32)
    grid = np.full(support.shape, -1, dtype=np.int8)
    grid[support <= geometry["absent_fraction"]] = 0
    grid[support >= geometry["present_fraction"]] = 1
    for r, c in zip(*np.where(grid == -1), strict=True):
        band = geometry["bands"][r]
        x0 = geometry["x0"] + c * width
        pixels = rgb[band["y0"] : band["y1"], x0 : x0 + width]
        ys, xs = np.where(np.all(pixels == band["color"], axis=-1))
        # The ball takes the scanline's color, including inside the brick wall.
        # A <=2x4 patch cannot be an 8x6 brick. Do not erase larger partial patches.
        if (
            xs.max() - xs.min() + 1 <= geometry["small_sprite_width"]
            and ys.max() - ys.min() + 1 <= geometry["small_sprite_height"]
        ):
            grid[r, c] = 0
    return grid, support


def panel(rgb, grid, geometry, title, subtitle):
    """Data diagnostic: untouched nearest-neighbor RGB alongside extracted cells."""
    canvas = Image.new("RGB", (730, 705), "#161a23")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill="white")
    draw.text((12, 28), subtitle, fill="#c2cbdb")
    canvas.paste(Image.fromarray(rgb).resize((480, 630), Image.Resampling.NEAREST), (10, 55))
    for r, band in enumerate(geometry["bands"]):
        for c in range(geometry["columns"]):
            value = grid[r, c]
            color = tuple(band["color"]) if value == 1 else "#080b11"
            if value == -1:
                color = "#ffffff"
            x, y = 505 + c * 12, 225 + r * 25
            draw.rectangle((x, y, x + 10, y + 21), fill=color, outline="#647080")
    draw.text((505, 390), "Colored: present", fill="white")
    draw.text((505, 408), "Dark: absent", fill="white")
    draw.text((505, 426), "White: uncertain", fill="white")
    return canvas


def write_report(summary, output):
    selected_titles = [
        "Almost full wall",
        "Sparse wall",
        "Ball-shaped patch excluded",
        "Brick destroyed",
        "Count disagreement",
        "Held-out damaged wall",
    ]
    cases = [
        e
        for title in selected_titles
        for e in summary["examples"]
        if e["title"] == title and e["available"]
    ]
    contact = Image.new("RGB", (1460, 705 * ((len(cases) + 1) // 2)), "#161a23")
    for i, case in enumerate(cases):
        contact.paste(Image.open(output / case["file"]), (730 * (i % 2), 705 * (i // 2)))
    contact.save(output / "contact-sheet.png")
    lines = [
        "# Deterministic brick-grid prototype",
        "",
        "The extractor recovered a 6 x 18 visible brick grid from existing RGB frames. "
        "Source datasets were not changed. This is a sampled feasibility test, not "
        "a full dataset labeling run or a per-cell ground-truth accuracy estimate.",
        "",
        "## Results",
        "",
        "| Split | Transitions | Episodes | Exact recorded-count agreement | "
        "Single-brick destruction checks |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "heldout"):
        s = summary[split]
        lines.append(
            f"| {split} | {s['frames']:,} | {s['episodes']} | "
            f"{s['exact']:,}/{s['frames']:,} ({s['exact'] / s['frames']:.2%}) | "
            f"{s['matched_destruction_pairs']}/{s['brick_destruction_pairs']} |"
        )
    lines += [
        "",
        "## Method and independence",
        "",
        f"Dataset: `{summary['dataset']}`.",
        "",
        f"Calibration used training episode {summary['geometry']['calibration']['episode_id']}, "
        f"step {summary['geometry']['calibration']['step']}. Geometry is x=[8,152), "
        "y=[57,93), six rows of 18 cells, each 8 x 6 pixels. Colors and row boundaries "
        "come from that RGB image; the 18-column layout is a game-specific assumption.",
        "",
        "A cell is present at >=75% matching row-color pixels. <=5% is absent. "
        "A remaining patch fitting within 2 x 4 pixels is treated as the ball, not a brick. "
        "Larger partial patches remain unknown. Pixel support is not a probability.",
        "",
        f"Training evaluation uses the first {summary['per_shard']} transitions of every shard. "
        f"Final held-out evaluation uses {summary['per_shard']} transitions starting at offset "
        f"{summary['heldout_offset']} of every held-out shard. Initial held-out prefixes were "
        "exploratory; the ball-shape rule was derived from training ambiguities and frozen "
        "before the fresh evaluation. Samples are clustered, not independent random frames.",
        "",
        "RGB is the extractor's only input. Counts, ball positions, and wall counters do not "
        "alter the extracted grid. Native counts are used afterward to accept or reject a "
        "state label. Labels align to successor frame IDs; temporal checks require adjacent "
        "steps in the same episode.",
        "",
        "## Failure modes and limits",
        "",
        "Training count mismatches occur during startup drawing, when internal brick count "
        "is 108 but the image shows an incomplete wall. The visible grid is retained, while "
        "the entire internal-state label is masked unknown. These frames require animation "
        "state if a renderer must reproduce startup exactly.",
        "",
        "Exact count agreement cannot rule out compensating cell errors. Visual examples, "
        "synthetic geometry/occlusion checks, and adjacent destruction checks provide "
        "additional evidence, but no independent per-brick native labels are available. "
        "Later-wall reset cases were absent from the selected sample and remain untested. "
        "The geometry and palette apply to this fixed-resolution, lossless dataset.",
        "",
        "## Artifacts",
        "",
        "- `labels.parquet`: episode/step/frame identity, row-major visible `grid`, "
        "count-gated `state_grid`, matching fractions, count and temporal checks.",
        "- `geometry.json`: fixed detector settings and calibration identity.",
        "- `results.json`: provenance, aggregate results, example identities.",
        "- `contact-sheet.png` and `case-*.png`: RGB alongside the inferred matrix.",
        "",
        f"![Selected diagnostic frames]({(output / 'contact-sheet.png').resolve()})",
        "",
        "## Selected cases",
        "",
    ]
    for e in summary["examples"]:
        if e["available"]:
            lines.append(
                f"- [{e['title']}]({(output / e['file']).resolve()}): {e['split']} "
                f"episode {e['episode_id']}, step {e['step']}, "
                f"frame {e['successor_frame_id']}; visible={e['visible_present']}, "
                f"recorded={e['recorded_bricks']}."
            )
        else:
            lines.append(f"- {e['title']}: no example in the selected sample.")
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("logs/brick-grid-prototype"))
    parser.add_argument("--per-shard", type=int, default=64)
    parser.add_argument(
        "--heldout-offset",
        type=int,
        default=64,
        help="Fresh final evaluation excludes the initial exploratory prefixes",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = candidates(args.dataset, args.per_shard, args.heldout_offset)
    frames = Frames(args.dataset, compact=True, cache=args.cache)

    def rgb(row):
        return frames.get(row["successor_frame_id"]).permute(1, 2, 0).numpy()

    seed = next(
        r
        for r in rows
        if r["split"] == "train" and r["step"] > 100 and r["labels"]["bricks_remaining"] >= 100
    )
    geometry = calibrate(rgb(seed))
    geometry["calibration"] = {
        k: seed[k] for k in ("split", "episode_id", "step", "successor_frame_id")
    }
    (args.output / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
    outputs, grids = [], []
    previous = {}
    for i, row in enumerate(rows):
        grid, support = extract(rgb(row), geometry)
        grids.append(grid)
        labels = row["labels"]
        present, unknown = int((grid == 1).sum()), int((grid == -1).sum())
        expected = labels["bricks_remaining"]
        key = (row["split"], row["episode_id"])
        last = previous.get(key)
        consecutive = last is not None and last[0]["step"] + 1 == row["step"]
        if consecutive:
            assert last[0]["successor_frame_id"] == row["source_frame_id"]
        gains = int(((last[1] == 0) & (grid == 1)).sum()) if consecutive else None
        losses = int(((last[1] == 1) & (grid == 0)).sum()) if consecutive else None
        delta = expected - last[0]["labels"]["bricks_remaining"] if consecutive else None
        output = {
            k: row[k]
            for k in (
                "split",
                "episode_id",
                "step",
                "source_frame_id",
                "successor_frame_id",
                "shard",
            )
        }
        output.update(
            recorded_bricks=expected,
            visible_present=present,
            unknown_cells=unknown,
            count_exact=unknown == 0 and present == expected,
            count_compatible=present <= expected <= present + unknown,
            count_error=present - expected,
            grid=grid.ravel().tolist(),
            state_grid=(
                grid.ravel().tolist() if unknown == 0 and present == expected else [-1] * grid.size
            ),
            state_label_accepted=unknown == 0 and present == expected,
            pixel_support=support.ravel().tolist(),
            small_sprite_cells=int(((grid == 0) & (support > geometry["absent_fraction"])).sum()),
            visible_gains=gains,
            visible_losses=losses,
            recorded_delta=delta,
            walls_cleared=labels["walls_cleared"],
            ball_in_wall=47 <= labels["ball_y"] <= 84,
            serve=labels["ball_y"] == 0,
        )
        outputs.append(output)
        previous[key] = (row, grid)
        if (i + 1) % 10000 == 0:
            print(f"extracted {i + 1}/{len(rows)}", flush=True)
    pq.write_table(pa.Table.from_pylist(outputs), args.output / "labels.parquet")
    summary = {}
    for split in ("train", "heldout"):
        values = [o for o in outputs if o["split"] == split]
        summary[split] = {
            "frames": len(values),
            "episodes": len({v["episode_id"] for v in values}),
            "exact": sum(v["count_exact"] for v in values),
            "compatible": sum(v["count_compatible"] for v in values),
            "unknown_frames": sum(v["unknown_cells"] > 0 for v in values),
            "small_sprite_frames": sum(v["small_sprite_cells"] > 0 for v in values),
            "brick_destruction_pairs": sum(v["recorded_delta"] == -1 for v in values),
            "matched_destruction_pairs": sum(
                v["recorded_delta"] == -1 and v["visible_losses"] == 1 for v in values
            ),
            "error_histogram": dict(sorted(Counter(v["count_error"] for v in values).items())),
            "mismatch_steps": [v["step"] for v in values if not v["count_compatible"]][:40],
        }
    selectors = [
        ("Full visible wall", lambda o: o["count_exact"] and o["recorded_bricks"] == 108),
        ("Almost full wall", lambda o: o["count_exact"] and 95 < o["recorded_bricks"] < 108),
        ("Half-cleared wall", lambda o: o["count_exact"] and 45 <= o["recorded_bricks"] <= 55),
        ("Sparse wall", lambda o: o["count_exact"] and o["recorded_bricks"] <= 10),
        ("Ball in wall", lambda o: o["count_exact"] and o["ball_in_wall"]),
        ("Brick destroyed", lambda o: o["recorded_delta"] == -1 and o["visible_losses"] == 1),
        ("Uncertain cell", lambda o: o["unknown_cells"] > 0),
        ("Ball-shaped patch excluded", lambda o: o["small_sprite_cells"] > 0),
        ("Count disagreement", lambda o: not o["count_compatible"]),
        ("Visible wall growth", lambda o: (o["visible_gains"] or 0) > 0),
        ("Serve state", lambda o: o["serve"]),
        ("Later wall", lambda o: o["walls_cleared"] > 0),
        (
            "Held-out damaged wall",
            lambda o: (
                o["split"] == "heldout" and o["count_exact"] and 20 < o["recorded_bricks"] < 80
            ),
        ),
    ]
    examples, selected = [], set()
    for title, predicate in selectors:
        choices = [i for i, o in enumerate(outputs) if predicate(o) and i not in selected]
        if not choices:
            examples.append(dict(title=title, available=False))
            continue
        i = choices[0]
        selected.add(i)
        o = outputs[i]
        subtitle = (
            f"{o['split']} episode {o['episode_id']} step {o['step']} | "
            f"visible {o['visible_present']} unknown {o['unknown_cells']} "
            f"recorded {o['recorded_bricks']}"
        )
        filename = f"case-{len(examples):02d}.png"
        panel(rgb(rows[i]), grids[i], geometry, title, subtitle).save(args.output / filename)
        examples.append(
            dict(
                title=title,
                available=True,
                file=filename,
                **{k: v for k, v in o.items() if k not in ("grid", "state_grid", "pixel_support")},
            )
        )
    summary.update(
        dataset=str(args.dataset),
        per_shard=args.per_shard,
        heldout_offset=args.heldout_offset,
        geometry=geometry,
        examples=examples,
        unique_frames=len({o["successor_frame_id"] for o in outputs}),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dataset_manifest_sha256=hashlib.sha256(
            (args.dataset / "manifest.json").read_bytes()
        ).hexdigest(),
    )
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary, args.output)
    print(json.dumps({k: v for k, v in summary.items() if k in ("train", "heldout")}, indent=2))


if __name__ == "__main__":
    main()
