#!/usr/bin/env python3
"""Validate and summarize the formal 2x2 EEM/CAM ablation."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from scipy import stats


VARIANTS = ("base", "eem", "cam", "eem_cam")
SEEDS = (2026, 2027, 2028, 2029, 2030)
CLASS_NAMES = ("background", "red", "green", "blue", "yellow")
EXPECTED_MANIFEST_SHA256 = (
    "3992b99fc6b659627996059dd2e558c55b9174b612a76fb5d97c726b5accc2ea"
)
EXPECTED_SWITCHES = {
    "base": (False, False),
    "eem": (True, False),
    "cam": (False, True),
    "eem_cam": (True, True),
}
METRICS = (
    "mean_iou",
    "mean_dice",
    "pixel_accuracy",
    *(f"iou_{name}" for name in CLASS_NAMES),
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_sd_ci(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray(list(values), dtype=float)
    n = data.size
    mean = float(np.mean(data))
    sd = float(np.std(data, ddof=1)) if n > 1 else 0.0
    if n > 1:
        critical = float(stats.t.ppf(0.975, n - 1))
        half_width = critical * sd / math.sqrt(n)
    else:
        half_width = 0.0
    return {
        "n": int(n),
        "mean": mean,
        "sample_sd": sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def one_sample_test(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray(list(values), dtype=float)
    summary = mean_sd_ci(data)
    t_result = stats.ttest_1samp(data, popmean=0.0)
    try:
        w_result = stats.wilcoxon(
            data, alternative="two-sided", zero_method="wilcox", method="auto"
        )
        w_stat = float(w_result.statistic)
        w_p = float(w_result.pvalue)
    except ValueError:
        w_stat = 0.0
        w_p = 1.0
    sd = summary["sample_sd"]
    return {
        **summary,
        "t_statistic": float(t_result.statistic),
        "t_p": float(t_result.pvalue),
        "wilcoxon_statistic": w_stat,
        "wilcoxon_p": w_p,
        "cohen_dz": summary["mean"] / sd if sd else 0.0,
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running_max = 0.0
    for rank, raw_index in enumerate(order):
        index = int(raw_index)
        running_max = max(running_max, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted.tolist()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized_control_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove the two intended factors and variant label before comparison."""
    normalized = copy.deepcopy(config)
    normalized["experiment"]["name"] = "ablation_eem_cam_v1_VARIANT"
    normalized["model"].pop("use_edge_enhancement", None)
    normalized["model"].pop("use_color_attention", None)
    return normalized


def load_runs(root: Path) -> tuple[dict[str, dict[int, dict]], dict[str, Any]]:
    errors: list[str] = []
    runs: dict[str, dict[int, dict]] = {variant: {} for variant in VARIANTS}
    normalized_configs: list[dict[str, Any]] = []
    manifest_hashes: set[str] = set()
    pretrained_hashes: set[str] = set()

    for variant in VARIANTS:
        expected_eem, expected_cam = EXPECTED_SWITCHES[variant]
        for seed in SEEDS:
            run_dir = root / variant / f"seed_{seed}"
            paths = {
                "metadata": run_dir / "run_metadata.json",
                "metrics": run_dir / "test_metrics.json",
                "history": run_dir / "history.csv",
                "config": run_dir / "config_resolved.yaml",
            }
            for path in paths.values():
                if not path.is_file():
                    errors.append(f"Missing file: {path}")
            if any(not path.is_file() for path in paths.values()):
                continue

            metadata = read_json(paths["metadata"])
            metrics = read_json(paths["metrics"])
            config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
            with paths["history"].open(
                "r", encoding="utf-8-sig", newline=""
            ) as handle:
                history = list(csv.DictReader(handle))

            prefix = f"{variant} seed={seed}"
            checks = {
                "status=completed": metadata.get("status") == "completed",
                "seed matches": metadata.get("seed") == seed,
                "model=ecsegformer": metadata.get("model") == "ecsegformer",
                "smoke_test=false": metadata.get("smoke_test") is False,
                "split=fixed_split_v1": metadata.get("split_id")
                == "fixed_split_v1",
                "manifest hash expected": metadata.get("manifest_sha256")
                == EXPECTED_MANIFEST_SHA256,
                "170/170 encoder mapping": metadata.get(
                    "pretraining_load", {}
                ).get("matched_encoder_tensors")
                == 170
                and metadata.get("pretraining_load", {}).get(
                    "encoder_tensor_count"
                )
                == 170,
                "dataset sizes": metadata.get("dataset_sizes")
                == {"train": 975, "val": 148, "test": 145},
                "30 epochs": len(history) == 30
                and bool(history)
                and int(history[-1]["epoch"]) == 30,
                "test split": metrics.get("dataset_split") == "test",
                "37 test batches": metrics.get("processed_batches") == 37,
                "metrics seed matches": metrics.get("seed") == seed,
                "metrics smoke_test=false": metrics.get("smoke_test") is False,
                "EEM switch": config["model"].get("use_edge_enhancement")
                is expected_eem,
                "CAM switch": config["model"].get("use_color_attention")
                is expected_cam,
                "auxiliary head controlled": config["model"].get(
                    "use_green_aux_head"
                )
                is True,
            }
            for label, passed in checks.items():
                if not passed:
                    errors.append(f"{prefix}: failed {label}")

            per_class = metrics.get("per_class", [])
            if len(per_class) != len(CLASS_NAMES):
                errors.append(
                    f"{prefix}: expected 5 class metrics, got {len(per_class)}"
                )
                continue

            record = {
                "variant": variant,
                "seed": seed,
                "mean_iou": float(metrics["mean_iou"]),
                "mean_dice": float(metrics["mean_dice"]),
                "pixel_accuracy": float(metrics["pixel_accuracy"]),
                "best_val_mean_iou": float(metadata["best_val_mean_iou"]),
                "runtime_seconds": float(metadata["completed_at_unix"])
                - float(metadata["started_at_unix"]),
                "best_epoch": int(
                    max(history, key=lambda row: float(row["val_mean_iou"]))[
                        "epoch"
                    ]
                ),
            }
            for index, class_name in enumerate(CLASS_NAMES):
                record[f"iou_{class_name}"] = float(per_class[index]["iou"])
            runs[variant][seed] = record
            normalized_configs.append(normalized_control_config(config))
            manifest_hashes.add(str(metadata.get("manifest_sha256")))
            pretrained_hashes.add(str(metadata.get("pretrained_sha256")))

    if normalized_configs:
        canonical = yaml.safe_dump(normalized_configs[0], sort_keys=True)
        for index, config in enumerate(normalized_configs[1:], start=1):
            if yaml.safe_dump(config, sort_keys=True) != canonical:
                errors.append(
                    f"Controlled configuration differs in loaded run index {index}"
                )
    if manifest_hashes != {EXPECTED_MANIFEST_SHA256}:
        errors.append(f"Unexpected manifest hashes: {sorted(manifest_hashes)}")
    if len(pretrained_hashes) != 1:
        errors.append(f"Pretrained hashes differ: {sorted(pretrained_hashes)}")

    validation = {
        "valid": not errors,
        "expected_runs": len(VARIANTS) * len(SEEDS),
        "loaded_runs": sum(len(value) for value in runs.values()),
        "errors": errors,
        "manifest_hashes": sorted(manifest_hashes),
        "pretrained_hashes": sorted(pretrained_hashes),
    }
    return runs, validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/ablation_eem_cam_v1")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/ablation_eem_cam_v1/statistics"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs, validation = load_runs(args.root)
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not validation["valid"]:
        raise SystemExit(
            f"Validation failed; see {args.output_dir / 'validation_report.json'}"
        )

    per_seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        row: dict[str, Any] = {"seed": seed}
        for variant in VARIANTS:
            for metric in METRICS:
                row[f"{variant}_{metric}"] = runs[variant][seed][metric]
        for variant in VARIANTS[1:]:
            for metric in METRICS:
                row[f"{variant}_minus_base_{metric}"] = (
                    runs[variant][seed][metric] - runs["base"][seed][metric]
                )
        for metric in METRICS:
            base = runs["base"][seed][metric]
            eem = runs["eem"][seed][metric]
            cam = runs["cam"][seed][metric]
            both = runs["eem_cam"][seed][metric]
            row[f"eem_main_effect_{metric}"] = 0.5 * (
                (eem - base) + (both - cam)
            )
            row[f"cam_main_effect_{metric}"] = 0.5 * (
                (cam - base) + (both - eem)
            )
            row[f"interaction_{metric}"] = both - eem - cam + base
        per_seed_rows.append(row)
    write_csv(args.output_dir / "per_seed_results.csv", per_seed_rows)

    summary_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for variant in VARIANTS:
        summaries[variant] = {}
        for metric in (*METRICS, "best_val_mean_iou", "best_epoch", "runtime_seconds"):
            summary = mean_sd_ci(runs[variant][seed][metric] for seed in SEEDS)
            summaries[variant][metric] = summary
            summary_rows.append({"variant": variant, "metric": metric, **summary})
    write_csv(args.output_dir / "variant_summary.csv", summary_rows)

    contrasts: dict[str, dict[str, float]] = {}
    contrast_rows: list[dict[str, Any]] = []
    primary_contrast_names: list[str] = []
    for variant in VARIANTS[1:]:
        name = f"{variant}_minus_base"
        values = [
            runs[variant][seed]["mean_iou"]
            - runs["base"][seed]["mean_iou"]
            for seed in SEEDS
        ]
        result = one_sample_test(values)
        contrasts[name] = result
        primary_contrast_names.append(name)
        contrast_rows.append({"contrast": name, "metric": "mean_iou", **result})

    factorial_values = {
        "eem_main_effect": [
            0.5
            * (
                runs["eem"][seed]["mean_iou"]
                - runs["base"][seed]["mean_iou"]
                + runs["eem_cam"][seed]["mean_iou"]
                - runs["cam"][seed]["mean_iou"]
            )
            for seed in SEEDS
        ],
        "cam_main_effect": [
            0.5
            * (
                runs["cam"][seed]["mean_iou"]
                - runs["base"][seed]["mean_iou"]
                + runs["eem_cam"][seed]["mean_iou"]
                - runs["eem"][seed]["mean_iou"]
            )
            for seed in SEEDS
        ],
        "eem_x_cam_interaction": [
            runs["eem_cam"][seed]["mean_iou"]
            - runs["eem"][seed]["mean_iou"]
            - runs["cam"][seed]["mean_iou"]
            + runs["base"][seed]["mean_iou"]
            for seed in SEEDS
        ],
    }
    for name, values in factorial_values.items():
        result = one_sample_test(values)
        contrasts[name] = result
        primary_contrast_names.append(name)
        contrast_rows.append({"contrast": name, "metric": "mean_iou", **result})

    adjusted_t = holm_adjust([contrasts[name]["t_p"] for name in primary_contrast_names])
    adjusted_w = holm_adjust(
        [contrasts[name]["wilcoxon_p"] for name in primary_contrast_names]
    )
    for row, t_p, w_p in zip(contrast_rows, adjusted_t, adjusted_w, strict=True):
        row["t_p_holm"] = t_p
        row["wilcoxon_p_holm"] = w_p
        contrasts[row["contrast"]]["t_p_holm"] = t_p
        contrasts[row["contrast"]]["wilcoxon_p_holm"] = w_p
    write_csv(args.output_dir / "primary_contrasts.csv", contrast_rows)

    payload = {
        "validation": validation,
        "variant_summaries": summaries,
        "primary_mean_iou_contrasts": contrasts,
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = [
        "# EEM/CAM ablation: 4 variants × 5 paired seeds",
        "",
        "All variants use the fixed split, seeds 2026–2030, 30 epochs, and the "
        "same auxiliary head and optimization protocol. Only EEM and CAM vary.",
        "",
        "## Test mIoU",
        "",
        "| Variant | Mean | Sample SD | 95% t CI |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        item = summaries[variant]["mean_iou"]
        report.append(
            f"| {variant} | {item['mean']:.6f} | {item['sample_sd']:.6f} | "
            f"[{item['ci95_low']:.6f}, {item['ci95_high']:.6f}] |"
        )
    report.extend(
        [
            "",
            "## Paired primary contrasts",
            "",
            "| Contrast | Mean effect | 95% CI | Paired t p | Holm p | Wilcoxon p | Holm p |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in contrast_rows:
        report.append(
            f"| {row['contrast']} | {row['mean']:.6f} | "
            f"[{row['ci95_low']:.6f}, {row['ci95_high']:.6f}] | "
            f"{row['t_p']:.8g} | {row['t_p_holm']:.8g} | "
            f"{row['wilcoxon_p']:.8g} | {row['wilcoxon_p_holm']:.8g} |"
        )
    report.extend(
        [
            "",
            "With five seeds, confidence intervals and effect consistency are "
            "more informative than relying on p-values alone.",
            "",
        ]
    )
    (args.output_dir / "analysis_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
