"""Publish a completely validated additive brick-annotation build to its source Hub repo."""

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

from gymemu.brick_labels import VERSION, Issue
from gymemu.cache import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--repo", default="tsilva/gradlab-breakout-trajectories")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    build = args.build.resolve()
    progress = json.loads((build / "progress.json").read_text())
    if progress["stage"] != "validated":
        raise ValueError("Full build validation has not completed")
    identity = json.loads((build / "build-identity.json").read_text())
    source = Path(identity["source_dataset"])
    destination = build / "dataset"
    annotations = destination / "annotations" / VERSION
    for path in (
        Path(__file__).with_name("augment_brick_dataset.py"),
        Path(__file__).parent / "gymemu/brick_labels.py",
    ):
        if file_hash(path) != identity["code"][path.name]:
            raise ValueError(f"Build code has changed: {path.name}")
        shutil.copyfile(path, annotations / path.name)
    files = json.loads((annotations / "files.json").read_text())
    for receipt in files:
        if file_hash(destination / receipt["relative"]) != receipt["sha256"]:
            raise ValueError(f"Validated file changed: {receipt['relative']}")
    # Include episode files, which were separately roundtrip-checked by the build.
    for path in sorted(destination.glob("episodes/*/*.parquet")):
        relative = str(path.relative_to(destination))
        if not any(r["relative"] == relative for r in files):
            files.append(
                dict(
                    relative=relative,
                    sha256=file_hash(path),
                    original_sha256=file_hash(source / relative),
                )
            )
    (annotations / "files.json").write_text(json.dumps(files, indent=2) + "\n")
    # The public receipt has Hub provenance, not local machine paths.
    provenance = {k: v for k, v in identity.items() if k != "source_dataset"}
    provenance["source_dataset"] = args.repo
    (annotations / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    summary = progress["summary"]
    schema = dict(
        version=VERSION,
        grid_shape=[6, 18],
        grid_values={"absent": 0, "present": 1, "unknown": -1},
        quality_bits={issue.name: issue.value for issue in Issue},
        source_revision=identity["source_revision"],
        startup_definition="Initial prefix before first full visible wall or gameplay "
        "evidence, only when first native successor has 108 bricks, 0 walls, 0 score.",
    )
    (annotations / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    for name in ("manifest.json", "dataset-summary.json"):
        value = json.loads((source / name).read_text())
        value.setdefault("annotations", {})[VERSION] = dict(
            schema=schema, summary=summary, receipts=f"annotations/{VERSION}"
        )
        (destination / name).write_text(json.dumps(value, indent=2) + "\n")
    docs = (Path(__file__).parent / "docs/training.md").read_text()
    specification = docs.split("## Brick dataset annotations\n", 1)[1].split("\n## ", 1)[0]
    # Local invocation paths are examples; this paragraph explains the published tables.
    card = (source / "README.md").read_text()
    lines = [
        "",
        "## Brick annotations added September 17, 2026",
        "",
        f"This revision adds `{VERSION}` annotations to all transitions and episode-initial "
        f"states from source revision `{identity['source_revision']}`. Every original column "
        "was roundtrip-checked for equality. Image assets and split memberships are unchanged. "
        "Existing collection statistics above describe the source snapshot.",
        "",
        "| Split | Transitions | Suspect grids | Initial-animation targets |",
        "|---|---:|---:|---:|",
    ]
    for split in ("train", "heldout"):
        s = summary[split]
        lines.append(
            f"| {split} | {s['transitions']:,} | {s['suspect']:,} | {s['initial_layout']:,} |"
        )
    lines += [
        "",
        f"Additionally, {summary['initial_frames_layout']:,} episode-initial frames "
        "are flagged as initial animation. All initial-frame native counts are unavailable "
        "and explicitly flagged. Suspect matrices are retained for inspection, not silently "
        "corrected or removed.",
        "",
        specification,
        "The detector is game-specific and deterministic. It does not provide independent "
        "native per-cell ground truth. Quality flags are conservative checks, not proof "
        "that an unflagged grid is correct. See the annotation directory for source hashes, "
        "extractor code, complete summaries, and file receipts.",
    ]
    (destination / "README.md").write_text(card.rstrip() + "\n" + "\n".join(lines) + "\n")
    paths = sorted(p for p in destination.rglob("*") if p.is_file())
    api = HfApi()
    if api.dataset_info(args.repo).sha != identity["source_revision"]:
        raise ValueError("Hub head changed since the source snapshot; refusing to overwrite")
    print(
        f"Prepared {len(paths)} files, {sum(p.stat().st_size for p in paths):,} bytes", flush=True
    )
    if args.prepare_only:
        return
    result = api.create_commit(
        repo_id=args.repo,
        repo_type="dataset",
        parent_commit=identity["source_revision"],
        operations=[
            CommitOperationAdd(path_in_repo=str(p.relative_to(destination)), path_or_fileobj=p)
            for p in paths
        ],
        commit_message="Add brick grids, quality flags, and initial wall-animation labels",
        commit_description="Deterministic RGB extraction over every existing frame. All original "
        "transition/episode columns validated unchanged; images and split identities preserved.",
    )
    receipt = dict(
        repo=args.repo,
        source_revision=identity["source_revision"],
        revision=result.oid,
        url=result.commit_url,
        files=len(paths),
    )
    (build / "publication.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
