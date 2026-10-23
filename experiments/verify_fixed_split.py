"""Verify fixed-split integrity before starting any training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="experiments/manifests/fixed_split_v1.csv"
    )
    parser.add_argument("--project-root", default=".")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    manifest = (root / args.manifest).resolve()
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Manifest is empty")

    required = {
        "image_path",
        "mask_path",
        "group_id",
        "split",
        "is_augmented",
    }
    missing_columns = required - set(rows[0])
    if missing_columns:
        raise ValueError(f"Missing manifest columns: {sorted(missing_columns)}")

    image_paths: set[str] = set()
    groups_by_split: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    source_counts: dict[str, Counter] = defaultdict(Counter)
    problems: list[str] = []
    for row in rows:
        split = row["split"]
        if split not in {"train", "val", "test"}:
            problems.append(f"Invalid split: {split}")
            continue
        if row["image_path"] in image_paths:
            problems.append(f"Duplicate image: {row['image_path']}")
        image_paths.add(row["image_path"])
        image_path = root / row["image_path"]
        mask_path = root / row["mask_path"]
        if not image_path.is_file():
            problems.append(f"Missing image: {row['image_path']}")
        if not mask_path.is_file():
            problems.append(f"Missing mask: {row['mask_path']}")
        if split != "train" and int(row["is_augmented"]):
            problems.append(f"Offline augmentation in {split}: {row['image_path']}")
        groups_by_split[split].add(row["group_id"])
        counts[split] += 1
        source_counts[split][row.get("source", "UNKNOWN")] += 1

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = groups_by_split[left] & groups_by_split[right]
        if overlap:
            problems.append(
                f"{left}/{right} group overlap ({len(overlap)}): "
                f"{sorted(overlap)[:10]}"
            )
    if problems:
        raise ValueError("Split verification failed:\n- " + "\n- ".join(problems[:50]))

    report = {
        "status": "ok",
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "record_counts": dict(counts),
        "group_counts": {
            split: len(groups_by_split[split]) for split in ("train", "val", "test")
        },
        "source_counts": {
            split: dict(source_counts[split]) for split in ("train", "val", "test")
        },
        "group_overlap": 0,
        "offline_augmented_records_in_val_or_test": 0,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
