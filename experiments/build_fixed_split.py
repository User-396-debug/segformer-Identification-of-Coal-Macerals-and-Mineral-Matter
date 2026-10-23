"""Build a leakage-resistant fixed split manifest.

The requested 7.5:1.5:1.5 values are treated as relative weights and
normalised before assignment. Offline augmented samples (``_aug_``) never
enter validation or test; they are retained only when their original sample
is assigned to training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")
COLOR_TO_ID = {
    (0, 0, 0): 0,
    (255, 0, 0): 1,
    (0, 255, 0): 2,
    (0, 0, 255): 3,
    (255, 255, 0): 4,
}
CLASS_COLORS = np.asarray(list(COLOR_TO_ID), dtype=np.int32)


def canonical_group(stem: str) -> str:
    """Collapse offline augmentation variants into their source-image group."""
    return re.sub(r"_aug_.*$", "", stem, flags=re.IGNORECASE)


def source_name(stem: str) -> str:
    return stem.split("_", 1)[0].upper()


def mask_path_for(image_path: Path) -> Path:
    return image_path.parent.parent / "masks" / f"{image_path.stem}.png"


def mask_signature(mask_path: Path) -> tuple[str, list[int]]:
    mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if mask_bgr is None:
        raise ValueError(f"Cannot read mask: {mask_path}")
    mask = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    # Some PNG masks contain anti-aliased colours close to the five canonical
    # annotation colours. Map every pixel to its nearest canonical colour;
    # the training dataset uses the identical rule.
    pixels = mask.astype(np.int32)
    distances = ((pixels[:, :, None, :] - CLASS_COLORS[None, None, :, :]) ** 2).sum(
        axis=3
    )
    ids = np.argmin(distances, axis=2).astype(np.uint8)
    counts = np.bincount(ids.ravel(), minlength=len(COLOR_TO_ID)).astype(int).tolist()
    signature = "".join("1" if count else "0" for count in counts)
    return signature, counts


def collect_records(project_root: Path) -> list[dict]:
    by_path: dict[str, dict] = {}
    for original_split in SPLITS:
        image_dir = project_root / "data" / original_split / "images"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask_path = mask_path_for(image_path)
            if not mask_path.is_file():
                raise FileNotFoundError(f"Missing mask for {image_path}: {mask_path}")
            relative_image = image_path.relative_to(project_root).as_posix()
            if relative_image in by_path:
                raise ValueError(f"Duplicate image path: {relative_image}")
            signature, class_pixels = mask_signature(mask_path)
            stem = image_path.stem
            by_path[relative_image] = {
                "image_path": relative_image,
                "mask_path": mask_path.relative_to(project_root).as_posix(),
                "filename": image_path.name,
                "group_id": canonical_group(stem),
                "source": source_name(stem),
                "is_augmented": int(bool(re.search(r"_aug_", stem, re.IGNORECASE))),
                "original_split": original_split,
                "class_signature": signature,
                "class_pixels": class_pixels,
            }
    return list(by_path.values())


def assign_groups(
    records: list[dict], weights: list[float], split_seed: int
) -> dict[str, str]:
    ratios = np.asarray(weights, dtype=np.float64)
    if ratios.shape != (3,) or np.any(ratios <= 0):
        raise ValueError("Exactly three positive split weights are required")
    ratios /= ratios.sum()

    group_records: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        group_records[record["group_id"]].append(record)

    # Split only original images. Groups are stratified by source and by the
    # union of classes present in the group.
    strata: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for group_id, members in group_records.items():
        originals = [item for item in members if not item["is_augmented"]]
        if not originals:
            raise ValueError(f"Augmented group has no original sample: {group_id}")
        signature = "".join(
            "1" if any(item["class_pixels"][class_id] for item in originals) else "0"
            for class_id in range(len(COLOR_TO_ID))
        )
        source = originals[0]["source"]
        strata[(source, signature)].append((group_id, len(originals)))

    rng = random.Random(split_seed)
    assignments: dict[str, str] = {}
    for stratum in sorted(strata):
        groups = strata[stratum]
        rng.shuffle(groups)
        groups.sort(key=lambda item: item[1], reverse=True)
        total = sum(size for _, size in groups)
        target = ratios * total
        assigned = np.zeros(3, dtype=np.float64)
        for group_id, size in groups:
            # Prefer the split with the largest remaining proportional deficit.
            deficit = (target - assigned) / np.maximum(target, 1.0)
            split_index = int(np.argmax(deficit))
            assignments[group_id] = SPLITS[split_index]
            assigned[split_index] += size
    return assignments


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    records: list[dict],
    assignments: dict[str, str],
    output: Path,
    weights: list[float],
    split_seed: int,
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_path",
        "mask_path",
        "filename",
        "group_id",
        "source",
        "is_augmented",
        "original_split",
        "split",
        "class_signature",
        "class_0_pixels",
        "class_1_pixels",
        "class_2_pixels",
        "class_3_pixels",
        "class_4_pixels",
    ]
    written: list[dict] = []
    for record in records:
        split = assignments[record["group_id"]]
        # Offline augmentation is a training-only operation.
        if record["is_augmented"] and split != "train":
            continue
        row = {key: record[key] for key in fieldnames if key in record}
        row["split"] = split
        for class_id, count in enumerate(record["class_pixels"]):
            row[f"class_{class_id}_pixels"] = count
        written.append(row)
    written.sort(key=lambda row: (SPLITS.index(row["split"]), row["filename"]))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(written)

    original_counts = Counter(
        assignments[record["group_id"]]
        for record in records
        if not record["is_augmented"]
    )
    effective_counts = Counter(row["split"] for row in written)
    summary = {
        "split_id": output.stem,
        "split_seed": split_seed,
        "requested_weights": weights,
        "normalised_ratios": (np.asarray(weights) / sum(weights)).tolist(),
        "original_image_counts": dict(original_counts),
        "effective_training_counts": dict(effective_counts),
        "manifest_sha256": sha256_file(output),
        "group_count": len(assignments),
        "offline_augmented_records_in_train": sum(
            int(row["is_augmented"]) for row in written if row["split"] == "train"
        ),
        "offline_augmented_records_in_val_or_test": sum(
            int(row["is_augmented"]) for row in written if row["split"] != "train"
        ),
    }
    summary_path = output.with_name(f"{output.stem}_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--output", default="experiments/manifests/fixed_split_v1.csv"
    )
    parser.add_argument("--weights", type=float, nargs=3, default=[7.5, 1.5, 1.5])
    parser.add_argument("--split-seed", type=int, default=3407)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    output = (root / args.output).resolve()
    records = collect_records(root)
    assignments = assign_groups(records, args.weights, args.split_seed)
    summary = write_manifest(
        records, assignments, output, args.weights, args.split_seed
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
