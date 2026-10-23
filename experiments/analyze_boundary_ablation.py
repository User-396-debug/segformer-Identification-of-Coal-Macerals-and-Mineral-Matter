#!/usr/bin/env python3
"""Summarize five-seed boundary metrics for the completed EEM/CAM ablation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scipy import stats


VARIANTS = ("base", "eem", "cam", "eem_cam")
SEEDS = tuple(range(2026, 2031))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: dict[str, dict[str, object]] = {}
    raw: dict[str, dict[int, float]] = {}
    for variant in VARIANTS:
        values: dict[int, float] = {}
        for seed in SEEDS:
            path = args.root / variant / f"seed_{seed}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            values[seed] = float(
                payload["boundary"]["macro_boundary_f1_foreground"]
            )
        raw[variant] = values
        sample = list(values.values())
        rows[variant] = {
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
        paired = [raw[left][seed] - raw[right][seed] for seed in SEEDS]
        paired_mean = statistics.mean(paired)
        paired_sd = statistics.stdev(paired)
        standard_error = paired_sd / math.sqrt(len(paired))
        t_critical = float(
            stats.t.ppf(0.975, df=len(paired) - 1)
        )
        t_test = stats.ttest_1samp(paired, popmean=0.0)
        contrasts[name] = {
            "paired_differences": dict(zip(SEEDS, paired)),
            "mean_paired_difference": paired_mean,
            "sample_sd": paired_sd,
            "mean_difference_95ci": [
                paired_mean - t_critical * standard_error,
                paired_mean + t_critical * standard_error,
            ],
            "paired_t_statistic": float(t_test.statistic),
            "paired_t_pvalue_two_sided": float(t_test.pvalue),
            "positive_seed_count": sum(value > 0 for value in paired),
        }

    result = {
        "metric": "foreground_macro_boundary_f1",
        "tolerance_pixels": 2,
        "seeds": list(SEEDS),
        "variants": rows,
        "paired_contrasts": contrasts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
