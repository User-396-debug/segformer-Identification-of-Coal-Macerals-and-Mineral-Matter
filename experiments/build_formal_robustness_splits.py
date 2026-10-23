"""Build preregistered, subtype-aware manifests for split robustness.

The pilot split exposed that one extreme broad siderite image (KW_4_13)
can dominate the green-class pixel count and therefore the five-class
macro mIoU.  This builder treats that image as a fixed, independent
challenge case.  It is never used for training and is not included in the
main robustness test-set average.

The main manifests still contain siderite in every test set.  The remaining
siderite groups are assigned by a frozen rule:

* one broad siderite group in train and one in test;
* one small-inclusion siderite group in train and one in validation.

Candidates are ranked only by data characteristics (image counts and
per-class pixel proportions), before any model is trained.  Five candidates
with diverse test groups are selected and written with their hashes and
selection criteria.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.build_fixed_split import (  # noqa: E402
    COLOR_TO_ID,
    SPLITS,
    collect_records,
    sha256_file,
)


CHALLENGE_GROUP = "KW_4_13"
BROAD_SID_GROUPS = ("KW_4_14", "KW_4_15")
SMALL_SID_GROUPS = ("KW_14_1", "KW_14_2")
CM_GROUPS = ("LZT_10_8", "LZT_10_8_1")
TRACE_GROUPS = ("LZT_40_164_1", "LZT_6_4_1")
CUSTOM_GROUPS = set(
    BROAD_SID_GROUPS + SMALL_SID_GROUPS + CM_GROUPS + TRACE_GROUPS
)


def read_subtypes(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["group_id"]: row for row in rows}
    required = {CHALLENGE_GROUP, *CUSTOM_GROUPS}
    missing = required - set(result)
    if missing:
        raise ValueError(f"Subtype inventory is missing groups: {sorted(missing)}")
    return result


def group_metadata(records: list[dict]) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        grouped[record["group_id"]].append(record)

    metadata: dict[str, dict] = {}
    for group_id, members in grouped.items():
        originals = [item for item in members if not item["is_augmented"]]
        if not originals:
            raise ValueError(f"Augmented group has no original: {group_id}")
        pixels = np.asarray(
            [item["class_pixels"] for item in originals], dtype=np.int64
        ).sum(axis=0)
        signature = "".join("1" if value else "0" for value in pixels)
        metadata[group_id] = {
            "group_id": group_id,
            "members": members,
            "originals": originals,
            "original_count": len(originals),
            "augmented_count": sum(item["is_augmented"] for item in members),
            "source": originals[0]["source"],
            "signature": signature,
            "class_pixels": pixels,
        }
    return metadata


def assign_pair(
    assignments: dict[str, str],
    groups: tuple[str, str],
    destinations: tuple[str, str],
    rng: random.Random,
) -> None:
    shuffled = list(groups)
    rng.shuffle(shuffled)
    for group_id, split in zip(shuffled, destinations):
        assignments[group_id] = split


def build_assignment(
    metadata: dict[str, dict],
    ratios: np.ndarray,
    split_seed: int,
) -> dict[str, str]:
    rng = random.Random(split_seed)
    assignments: dict[str, str] = {}

    # Frozen subtype constraints.
    assign_pair(assignments, BROAD_SID_GROUPS, ("train", "test"), rng)
    assign_pair(assignments, SMALL_SID_GROUPS, ("train", "val"), rng)
    assign_pair(assignments, CM_GROUPS, ("train", "test"), rng)
    assign_pair(assignments, TRACE_GROUPS, ("train", "val"), rng)

    strata: dict[tuple[str, str], list[tuple[str, int]]] = defaultdict(list)
    for group_id, item in metadata.items():
        if group_id == CHALLENGE_GROUP or group_id in CUSTOM_GROUPS:
            continue
        strata[(item["source"], item["signature"])].append(
            (group_id, item["original_count"])
        )

    for stratum in sorted(strata):
        groups = strata[stratum]
        rng.shuffle(groups)
        groups.sort(key=lambda item: item[1], reverse=True)
        total = sum(size for _, size in groups)
        target = ratios * total
        assigned = np.zeros(3, dtype=np.float64)
        for group_id, size in groups:
            deficit = (target - assigned) / np.maximum(target, 1.0)
            index = int(np.argmax(deficit))
            assignments[group_id] = SPLITS[index]
            assigned[index] += size

    eligible = set(metadata) - {CHALLENGE_GROUP}
    if set(assignments) != eligible:
        missing = eligible - set(assignments)
        extra = set(assignments) - eligible
        raise ValueError(
            f"Incomplete assignment: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return assignments


def candidate_statistics(
    metadata: dict[str, dict],
    assignments: dict[str, str],
    ratios: np.ndarray,
    subtypes: dict[str, dict[str, str]],
) -> dict:
    original_counts = np.zeros(3, dtype=np.int64)
    effective_counts = np.zeros(3, dtype=np.int64)
    pixels = np.zeros((3, len(COLOR_TO_ID)), dtype=np.int64)
    group_counts = np.zeros(3, dtype=np.int64)
    subtype_groups: dict[str, Counter] = defaultdict(Counter)
    test_groups: list[str] = []

    for group_id, split in assignments.items():
        index = SPLITS.index(split)
        item = metadata[group_id]
        original_counts[index] += item["original_count"]
        effective_counts[index] += item["original_count"]
        if split == "train":
            effective_counts[index] += item["augmented_count"]
        pixels[index] += item["class_pixels"]
        group_counts[index] += 1
        subtype = subtypes.get(group_id, {}).get("mineral_subtype", "non_green")
        subtype_groups[subtype][split] += 1
        if split == "test":
            test_groups.append(group_id)

    image_fractions = original_counts / original_counts.sum()
    class_fractions = pixels / np.maximum(pixels.sum(axis=0), 1)
    # Equal class weighting prevents abundant background from masking a poor
    # rare-class distribution. Image-count balance is a smaller tie-breaker.
    class_error = float(np.mean((class_fractions - ratios[:, None]) ** 2))
    image_error = float(np.mean((image_fractions - ratios) ** 2))
    effective_train_target = ratios[0] * original_counts.sum()
    effective_penalty = float(
        ((effective_counts[0] - effective_train_target) / original_counts.sum()) ** 2
    )
    score = class_error + 0.25 * image_error + 0.01 * effective_penalty

    return {
        "score": score,
        "original_image_counts": {
            split: int(original_counts[index])
            for index, split in enumerate(SPLITS)
        },
        "effective_record_counts": {
            split: int(effective_counts[index])
            for index, split in enumerate(SPLITS)
        },
        "group_counts": {
            split: int(group_counts[index]) for index, split in enumerate(SPLITS)
        },
        "original_image_fractions": {
            split: float(image_fractions[index])
            for index, split in enumerate(SPLITS)
        },
        "class_pixel_counts": {
            split: pixels[index].astype(int).tolist()
            for index, split in enumerate(SPLITS)
        },
        "class_pixel_fractions_by_split": {
            split: class_fractions[index].astype(float).tolist()
            for index, split in enumerate(SPLITS)
        },
        "subtype_group_counts": {
            subtype: dict(counter) for subtype, counter in subtype_groups.items()
        },
        "test_groups": sorted(test_groups),
    }


def jaccard(left: list[str], right: list[str]) -> float:
    left_set, right_set = set(left), set(right)
    return len(left_set & right_set) / max(len(left_set | right_set), 1)


def select_candidates(candidates: list[dict], count: int) -> list[dict]:
    candidates.sort(key=lambda item: (item["statistics"]["score"], item["seed"]))
    selected: list[dict] = []
    # Prefer clearly different test sets, then relax only if required.
    for threshold in (0.35, 0.45, 0.60, 1.0):
        for candidate in candidates:
            if candidate in selected:
                continue
            tests = candidate["statistics"]["test_groups"]
            if all(
                jaccard(tests, item["statistics"]["test_groups"]) <= threshold
                for item in selected
            ):
                selected.append(candidate)
                if len(selected) == count:
                    return selected
    raise RuntimeError(f"Could select only {len(selected)} of {count} candidates")


def validate_constraints(
    assignments: dict[str, str],
    subtypes: dict[str, dict[str, str]],
) -> None:
    if CHALLENGE_GROUP in assignments:
        raise ValueError("Challenge group leaked into a formal manifest")
    if {assignments[group] for group in BROAD_SID_GROUPS} != {"train", "test"}:
        raise ValueError("Broad Sid constraint failed")
    if {assignments[group] for group in SMALL_SID_GROUPS} != {"train", "val"}:
        raise ValueError("Small Sid constraint failed")
    sid_by_split = Counter(
        split
        for group, split in assignments.items()
        if subtypes.get(group, {}).get("mineral_subtype") == "Sid"
    )
    if sid_by_split != Counter({"train": 2, "val": 1, "test": 1}):
        raise ValueError(f"Unexpected formal Sid distribution: {sid_by_split}")


def write_manifest(
    output: Path,
    records: list[dict],
    assignments: dict[str, str],
    subtypes: dict[str, dict[str, str]],
) -> None:
    fieldnames = [
        "image_path",
        "mask_path",
        "filename",
        "group_id",
        "source",
        "is_augmented",
        "original_split",
        "split",
        "mineral_subtype",
        "class_signature",
        "class_0_pixels",
        "class_1_pixels",
        "class_2_pixels",
        "class_3_pixels",
        "class_4_pixels",
    ]
    rows: list[dict] = []
    for record in records:
        group_id = record["group_id"]
        if group_id == CHALLENGE_GROUP:
            continue
        split = assignments[group_id]
        if record["is_augmented"] and split != "train":
            continue
        row = {
            key: record[key]
            for key in fieldnames
            if key in record and key != "class_pixels"
        }
        row["split"] = split
        row["mineral_subtype"] = subtypes.get(group_id, {}).get(
            "mineral_subtype", "non_green"
        )
        for class_id, value in enumerate(record["class_pixels"]):
            row[f"class_{class_id}_pixels"] = value
        rows.append(row)

    rows.sort(key=lambda row: (SPLITS.index(row["split"]), row["filename"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--inventory",
        default=(
            "outputs/split_robustness/robustness_split_02/"
            "green_subtype_inventory/green_subtype_inventory.csv"
        ),
    )
    parser.add_argument("--output-dir", default="experiments/manifests")
    parser.add_argument("--prefix", default="formal_robustness")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--candidate-start", type=int, default=5000)
    parser.add_argument("--candidate-count", type=int, default=3000)
    parser.add_argument("--weights", type=float, nargs=3, default=[7.5, 1.5, 1.5])
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    inventory = (root / args.inventory).resolve()
    output_dir = (root / args.output_dir).resolve()
    subtypes = read_subtypes(inventory)
    records = collect_records(root)
    metadata = group_metadata(records)
    if CHALLENGE_GROUP not in metadata:
        raise ValueError(f"Challenge group not found in dataset: {CHALLENGE_GROUP}")

    ratios = np.asarray(args.weights, dtype=np.float64)
    if ratios.shape != (3,) or np.any(ratios <= 0):
        raise ValueError("Exactly three positive split weights are required")
    ratios /= ratios.sum()

    candidates: list[dict] = []
    for seed in range(args.candidate_start, args.candidate_start + args.candidate_count):
        assignments = build_assignment(metadata, ratios, seed)
        validate_constraints(assignments, subtypes)
        statistics = candidate_statistics(metadata, assignments, ratios, subtypes)
        candidates.append(
            {"seed": seed, "assignments": assignments, "statistics": statistics}
        )

    selected = select_candidates(candidates, args.count)
    preregistration = {
        "protocol": "formal_subtype_balanced_split_robustness_v1",
        "status": "frozen_before_training",
        "requested_weights": args.weights,
        "normalised_ratios": ratios.tolist(),
        "candidate_seed_range": [
            args.candidate_start,
            args.candidate_start + args.candidate_count - 1,
        ],
        "selection_criterion": (
            "minimum equal-class pixel-fraction error plus image-count tie-breaker; "
            "test-set Jaccard diversity preferred; no model scores used"
        ),
        "challenge_policy": {
            "group_id": CHALLENGE_GROUP,
            "role": "fixed independent Sid challenge",
            "in_training": False,
            "included_in_main_miou": False,
            "reported_separately": True,
        },
        "formal_sid_rule": {
            "train": ["one broad Sid", "one small-inclusion Sid"],
            "validation": ["one small-inclusion Sid"],
            "test": ["one non-extreme broad Sid"],
        },
        "splits": [],
    }

    for index, candidate in enumerate(selected, start=1):
        split_id = f"{args.prefix}_{index:02d}"
        manifest = output_dir / f"{split_id}.csv"
        write_manifest(
            manifest, records, candidate["assignments"], subtypes
        )
        summary = {
            "split_id": split_id,
            "split_seed": candidate["seed"],
            "manifest_sha256": sha256_file(manifest),
            "challenge_group_excluded": CHALLENGE_GROUP,
            **candidate["statistics"],
        }
        summary_path = output_dir / f"{split_id}_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        preregistration["splits"].append(
            {
                "split_id": split_id,
                "split_seed": candidate["seed"],
                "manifest": manifest.relative_to(root).as_posix(),
                "manifest_sha256": summary["manifest_sha256"],
                "score": summary["score"],
                "test_groups": summary["test_groups"],
            }
        )

    preregistration_path = output_dir / f"{args.prefix}_preregistration.json"
    preregistration_path.write_text(
        json.dumps(preregistration, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(preregistration, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
