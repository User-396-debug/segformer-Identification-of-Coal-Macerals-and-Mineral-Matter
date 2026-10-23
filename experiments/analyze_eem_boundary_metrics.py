#!/usr/bin/env python3
"""Summarize paired BF1 and Boundary IoU results for the EEM/CAM ablation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scipy import stats


VARIANTS = ("base", "eem", "cam", "eem_cam")
SEEDS = tuple(range(2026, 2031))
CLASS_NAMES = ("background", "red", "green", "blue", "yellow")
METRICS = {
    "boundary_f1_2px": (
        "boundary",
        "macro_boundary_f1_foreground",
    ),
    "boundary_iou_standard": (
        "boundary_iou",
        "standard",
        "macro_boundary_iou_foreground",
    ),
    "boundary_iou_2px": (
        "boundary_iou",
        "fixed_width",
        "macro_boundary_iou_foreground",
    ),
}
PER_CLASS_PATHS = {
    "boundary_f1_2px": ("boundary", "per_class", "f1"),
    "boundary_iou_standard": (
        "boundary_iou",
        "standard",
        "per_class",
        "boundary_iou",
    ),
    "boundary_iou_2px": (
        "boundary_iou",
        "fixed_width",
        "per_class",
        "boundary_iou",
    ),
}


def nested(payload: dict, keys: tuple[str, ...]) -> float:
    value = payload
    for key in keys:
        value = value[key]
    return float(value)


def class_metric(
    payload: dict, metric_name: str, class_index: int
) -> float:
    keys = PER_CLASS_PATHS[metric_name]
    value = payload
    per_class_position = keys.index("per_class")
    for key in keys[:per_class_position]:
        value = value[key]
    value = value["per_class"]
    value = value[class_index]
    for key in keys[per_class_position + 1 :]:
        value = value[key]
    return float(value)


def paired_summary(
    left: dict[int, float], right: dict[int, float]
) -> dict:
    paired = [left[seed] - right[seed] for seed in SEEDS]
    mean = statistics.mean(paired)
    sd = statistics.stdev(paired)
    standard_error = sd / math.sqrt(len(paired))
    t_critical = float(stats.t.ppf(0.975, df=len(paired) - 1))
    test = stats.ttest_1samp(paired, popmean=0.0)
    return {
        "paired_differences": dict(zip(SEEDS, paired)),
        "mean_paired_difference": mean,
        "sample_sd": sd,
        "mean_difference_95ci": [
            mean - t_critical * standard_error,
            mean + t_critical * standard_error,
        ],
        "paired_t_statistic": float(test.statistic),
        "paired_t_pvalue_two_sided": float(test.pvalue),
        "positive_seed_count": sum(value > 0 for value in paired),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw: dict[str, dict[str, dict[int, float]]] = {}
    metadata = None
    for metric_name, metric_path in METRICS.items():
        raw[metric_name] = {}
        for variant in VARIANTS:
            values = {}
            for seed in SEEDS:
                path = args.root / variant / f"seed_{seed}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                values[seed] = nested(payload, metric_path)
                if metadata is None:
                    metadata = {
                        "boundary_f1_tolerance_pixels": payload[
                            "boundary"
                        ]["tolerance_pixels"],
                        "boundary_iou_standard_width_pixels": payload[
                            "boundary_iou"
                        ]["standard"]["width_pixels"],
                        "boundary_iou_standard_dilation_ratio": payload[
                            "boundary_iou"
                        ]["standard"]["dilation_ratio"],
                        "boundary_iou_fixed_width_pixels": payload[
                            "boundary_iou"
                        ]["fixed_width"]["width_pixels"],
                    }
            raw[metric_name][variant] = values

    result = {"seeds": list(SEEDS), "metadata": metadata, "metrics": {}}
    for metric_name, variants in raw.items():
        variant_summary = {}
        for variant, values in variants.items():
            sample = list(values.values())
            variant_summary[variant] = {
                "values_by_seed": values,
                "mean": statistics.mean(sample),
                "sample_sd": statistics.stdev(sample),
            }
        contrasts = {}
        for name, left, right in (
            ("eem_minus_base", "eem", "base"),
            ("cam_minus_base", "cam", "base"),
            ("eem_cam_minus_cam", "eem_cam", "cam"),
            ("eem_cam_minus_eem", "eem_cam", "eem"),
        ):
            contrasts[name] = paired_summary(
                variants[left], variants[right]
            )
        result["metrics"][metric_name] = {
            "variants": variant_summary,
            "paired_contrasts": contrasts,
        }
        per_class = {}
        for class_index, class_name in enumerate(CLASS_NAMES):
            class_variants = {}
            for variant in VARIANTS:
                values = {}
                for seed in SEEDS:
                    path = args.root / variant / f"seed_{seed}.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    values[seed] = class_metric(
                        payload, metric_name, class_index
                    )
                sample = list(values.values())
                class_variants[variant] = {
                    "values_by_seed": values,
                    "mean": statistics.mean(sample),
                    "sample_sd": statistics.stdev(sample),
                }
            per_class[class_name] = {
                "variants": class_variants,
                "eem_minus_base": paired_summary(
                    class_variants["eem"]["values_by_seed"],
                    class_variants["base"]["values_by_seed"],
                ),
            }
        result["metrics"][metric_name]["per_class"] = per_class

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
