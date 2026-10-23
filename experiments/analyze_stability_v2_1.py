#!/usr/bin/env python3
"""Validate and compare ECSegFormer stable-v2.1 over 10 paired seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from analyze_main_comparison import (
    holm_adjust,
    mean_sd_ci,
    paired_test,
    read_json,
    write_csv,
)


SEEDS = tuple(range(2026, 2036))
CLASS_NAMES = ("background", "red", "green", "blue", "yellow")
METRICS = (
    "mean_iou",
    "mean_dice",
    "pixel_accuracy",
    *(f"iou_{name}" for name in CLASS_NAMES),
)
EXPECTED_MANIFEST = (
    "3992b99fc6b659627996059dd2e558c55b9174b612a76fb5d97c726b5accc2ea"
)
EXPECTED_STABLE_CONFIG = (
    "688a08c7eccadaa15da8153b55fd3ac180532ce7fe99b2c66e5f7a2d88ed9342"
)


def load_history(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stable-root",
        type=Path,
        default=Path("outputs/stability_v2_1/ecsegformer"),
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=Path("outputs/main_comparison"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/stability_v2_1/statistics"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    roots = {
        "stable_v2_1": args.stable_root,
        "original_ecsegformer": args.baseline_root / "ecsegformer",
        "segformer": args.baseline_root / "segformer",
    }
    runs: dict[str, dict[int, dict[str, Any]]] = {
        name: {} for name in roots
    }
    errors: list[str] = []

    for name, root in roots.items():
        for seed in SEEDS:
            run_dir = root / f"seed_{seed}"
            metadata_path = run_dir / "run_metadata.json"
            metrics_path = run_dir / "test_metrics.json"
            history_path = run_dir / "history.csv"
            config_path = run_dir / "config_resolved.yaml"
            required = (metadata_path, metrics_path, history_path, config_path)
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                errors.extend(f"Missing file: {path}" for path in missing)
                continue

            metadata = read_json(metadata_path)
            test = read_json(metrics_path)
            history = load_history(history_path)
            prefix = f"{name} seed={seed}"
            expected_epochs = 30 if name == "stable_v2_1" else 50
            checks = {
                "status=completed": metadata.get("status") == "completed",
                "seed matches": metadata.get("seed") == seed,
                "smoke_test=false": metadata.get("smoke_test") is False,
                "matched encoder=170": metadata.get(
                    "pretraining_load", {}
                ).get("matched_encoder_tensors")
                == 170,
                "encoder count=170": metadata.get(
                    "pretraining_load", {}
                ).get("encoder_tensor_count")
                == 170,
                "test size=145": metadata.get("dataset_sizes", {}).get("test")
                == 145,
                "processed_batches=37": test.get("processed_batches") == 37,
                "test split": test.get("dataset_split") == "test",
                "metrics smoke_test=false": test.get("smoke_test") is False,
                "manifest expected": metadata.get("manifest_sha256")
                == EXPECTED_MANIFEST,
                "metrics manifest matches": test.get("manifest_sha256")
                == metadata.get("manifest_sha256"),
                "history length": len(history) == expected_epochs,
                "final epoch": bool(history)
                and int(history[-1]["epoch"]) == expected_epochs,
            }
            if name == "stable_v2_1":
                checks["stable config hash"] = (
                    metadata.get("config_sha256") == EXPECTED_STABLE_CONFIG
                )
                checks["model=ecsegformer"] = (
                    metadata.get("model") == "ecsegformer"
                )
                held_lrs = [
                    float(row["learning_rate"])
                    for row in history
                    if int(row["epoch"]) >= 22
                ]
                checks["epoch22-30 LR=5e-6"] = (
                    len(held_lrs) == 9
                    and all(abs(value - 5e-6) < 1e-12 for value in held_lrs)
                )
            for label, passed in checks.items():
                if not passed:
                    errors.append(f"{prefix}: failed {label}")

            per_class = test.get("per_class", [])
            if len(per_class) != 5:
                errors.append(
                    f"{prefix}: expected 5 classes, got {len(per_class)}"
                )
                continue

            best_row = max(
                history, key=lambda row: float(row["val_mean_iou"])
            )
            tail = history[-max(1, round(0.2 * len(history))) :]
            record: dict[str, Any] = {
                "seed": seed,
                "mean_iou": float(test["mean_iou"]),
                "mean_dice": float(test["mean_dice"]),
                "pixel_accuracy": float(test["pixel_accuracy"]),
                "best_val_mean_iou": float(metadata["best_val_mean_iou"]),
                "best_epoch": int(best_row["epoch"]),
                "best_epoch_fraction": int(best_row["epoch"]) / len(history),
                "final_val_mean_iou": float(history[-1]["val_mean_iou"]),
                "tail_val_sd": float(
                    np.std(
                        [float(row["val_mean_iou"]) for row in tail],
                        ddof=1,
                    )
                ),
                "runtime_seconds": float(metadata["completed_at_unix"])
                - float(metadata["started_at_unix"]),
            }
            record["best_to_final_val_gap"] = (
                record["best_val_mean_iou"]
                - record["final_val_mean_iou"]
            )
            for index, class_name in enumerate(CLASS_NAMES):
                record[f"iou_{class_name}"] = float(
                    per_class[index]["iou"]
                )
            runs[name][seed] = record

    validation = {
        "valid": not errors,
        "expected_runs": len(roots) * len(SEEDS),
        "loaded_runs": sum(len(values) for values in runs.values()),
        "errors": errors,
    }
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if errors:
        raise SystemExit(
            f"Validation failed; see {args.output_dir / 'validation_report.json'}"
        )

    per_seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        row: dict[str, Any] = {"seed": seed}
        for name in roots:
            for metric in METRICS:
                row[f"{name}_{metric}"] = runs[name][seed][metric]
            row[f"{name}_runtime_seconds"] = runs[name][seed][
                "runtime_seconds"
            ]
            row[f"{name}_best_val_mean_iou"] = runs[name][seed][
                "best_val_mean_iou"
            ]
            row[f"{name}_best_epoch"] = runs[name][seed]["best_epoch"]
        row["stable_minus_original_mean_iou"] = (
            runs["stable_v2_1"][seed]["mean_iou"]
            - runs["original_ecsegformer"][seed]["mean_iou"]
        )
        row["stable_minus_segformer_mean_iou"] = (
            runs["stable_v2_1"][seed]["mean_iou"]
            - runs["segformer"][seed]["mean_iou"]
        )
        per_seed_rows.append(row)
    write_csv(args.output_dir / "per_seed_results.csv", per_seed_rows)

    summary_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    curve_metrics = (
        "best_val_mean_iou",
        "best_epoch",
        "best_epoch_fraction",
        "final_val_mean_iou",
        "best_to_final_val_gap",
        "tail_val_sd",
        "runtime_seconds",
    )
    for name in roots:
        summaries[name] = {}
        for metric in (*METRICS, *curve_metrics):
            summary = mean_sd_ci(
                [runs[name][seed][metric] for seed in SEEDS]
            )
            summaries[name][metric] = summary
            summary_rows.append(
                {"model": name, "metric": metric, **summary}
            )
    write_csv(args.output_dir / "model_summary.csv", summary_rows)

    paired_rows: list[dict[str, Any]] = []
    paired_payload: dict[str, dict[str, dict[str, float]]] = {}
    wins: dict[str, dict[str, int]] = {}
    for comparator in ("original_ecsegformer", "segformer"):
        comparison = f"stable_v2_1_minus_{comparator}"
        paired_payload[comparison] = {}
        comparison_rows: list[dict[str, Any]] = []
        for metric in METRICS:
            result = paired_test(
                [runs["stable_v2_1"][seed][metric] for seed in SEEDS],
                [runs[comparator][seed][metric] for seed in SEEDS],
            )
            paired_payload[comparison][metric] = result
            comparison_rows.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    **result,
                    "paired_t_p_holm": result["paired_t_p"],
                    "wilcoxon_p_holm": result["wilcoxon_p"],
                }
            )
        secondary = [
            row for row in comparison_rows if row["metric"] != "mean_iou"
        ]
        adjusted_t = holm_adjust(
            [float(row["paired_t_p"]) for row in secondary]
        )
        adjusted_w = holm_adjust(
            [float(row["wilcoxon_p"]) for row in secondary]
        )
        for row, t_value, w_value in zip(
            secondary, adjusted_t, adjusted_w, strict=True
        ):
            row["paired_t_p_holm"] = t_value
            row["wilcoxon_p_holm"] = w_value
        paired_rows.extend(comparison_rows)

        differences = [
            runs["stable_v2_1"][seed]["mean_iou"]
            - runs[comparator][seed]["mean_iou"]
            for seed in SEEDS
        ]
        wins[comparison] = {
            "stable_wins": sum(value > 0 for value in differences),
            "comparator_wins": sum(value < 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
        }
    write_csv(args.output_dir / "paired_tests.csv", paired_rows)

    payload = {
        "validation": validation,
        "summaries": summaries,
        "paired": paired_payload,
        "wins": wins,
    }
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "# ECSegFormer stable-v2.1: 10-seed paired analysis",
        "",
        "All 30 runs passed validation. stable-v2.1 used 30 epochs; both "
        "original baselines used 50 epochs.",
        "",
        "## Test mIoU",
        "",
        "| Model | Mean | Sample SD | 95% t CI |",
        "|---|---:|---:|---:|",
    ]
    for name in roots:
        summary = summaries[name]["mean_iou"]
        report.append(
            f"| {name} | {fmt(summary['mean'])} | "
            f"{fmt(summary['sample_sd'])} | "
            f"[{fmt(summary['ci95_low'])}, {fmt(summary['ci95_high'])}] |"
        )
    report.extend(["", "## Paired mIoU tests", ""])
    for comparator in ("original_ecsegformer", "segformer"):
        key = f"stable_v2_1_minus_{comparator}"
        result = paired_payload[key]["mean_iou"]
        count = wins[key]
        report.append(
            f"- Versus {comparator}: mean difference={fmt(result['mean'])}, "
            f"SD={fmt(result['sample_sd'])}, "
            f"95% CI=[{fmt(result['ci95_low'])}, "
            f"{fmt(result['ci95_high'])}], "
            f"paired t p={result['paired_t_p']:.8g}, "
            f"Wilcoxon p={result['wilcoxon_p']:.8g}, "
            f"Cohen dz={fmt(result['cohen_dz'])}; "
            f"wins/losses/ties={count['stable_wins']}/"
            f"{count['comparator_wins']}/{count['ties']}."
        )
    report.extend(
        [
            "",
            "## Secondary metrics: stable-v2.1 versus SegFormer",
            "",
            "| Metric | Stable mean | SegFormer mean | Paired difference | "
            "Holm t p | Holm Wilcoxon p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    seg_secondary = [
        row
        for row in paired_rows
        if row["comparison"] == "stable_v2_1_minus_segformer"
        and row["metric"] != "mean_iou"
    ]
    for row in seg_secondary:
        metric = str(row["metric"])
        report.append(
            f"| {metric} | {fmt(summaries['stable_v2_1'][metric]['mean'])} | "
            f"{fmt(summaries['segformer'][metric]['mean'])} | "
            f"{fmt(float(row['mean']))} | "
            f"{float(row['paired_t_p_holm']):.8g} | "
            f"{float(row['wilcoxon_p_holm']):.8g} |"
        )
    report.extend(
        [
            "",
            "## Runtime and curve diagnostics",
            "",
            "| Model | Runtime min (mean +/- SD) | Tail val SD | Best-final gap |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in roots:
        runtime = summaries[name]["runtime_seconds"]
        tail_sd = summaries[name]["tail_val_sd"]
        gap = summaries[name]["best_to_final_val_gap"]
        report.append(
            f"| {name} | {runtime['mean'] / 60:.2f} +/- "
            f"{runtime['sample_sd'] / 60:.2f} | "
            f"{fmt(tail_sd['mean'])} | {fmt(gap['mean'])} |"
        )
    stable_sd = summaries["stable_v2_1"]["mean_iou"]["sample_sd"]
    original_sd = summaries["original_ecsegformer"]["mean_iou"]["sample_sd"]
    seg_sd = summaries["segformer"]["mean_iou"]["sample_sd"]
    stable_runtime = summaries["stable_v2_1"]["runtime_seconds"]["mean"]
    original_runtime = summaries["original_ecsegformer"]["runtime_seconds"][
        "mean"
    ]
    worst_seed = min(
        SEEDS, key=lambda seed: runs["stable_v2_1"][seed]["mean_iou"]
    )
    worst_delta = (
        runs["stable_v2_1"][worst_seed]["mean_iou"]
        - runs["segformer"][worst_seed]["mean_iou"]
    )
    report.extend(
        [
            "",
            "## Verdict",
            "",
            (
                f"- stable-v2.1 reduced test-mIoU SD by "
                f"{100 * (1 - stable_sd / original_sd):.1f}% versus the "
                "original ECSegFormer, but its SD remained "
                f"{stable_sd / seg_sd:.2f}x SegFormer's."
            ),
            (
                f"- Mean runtime fell by "
                f"{100 * (1 - stable_runtime / original_runtime):.1f}% "
                "versus the original ECSegFormer."
            ),
            (
                f"- The worst stable-v2.1 run was seed={worst_seed}: "
                f"mIoU={fmt(runs['stable_v2_1'][worst_seed]['mean_iou'])}, "
                f"{fmt(worst_delta)} versus the same-seed SegFormer."
            ),
            (
                "- The optimization significantly outperformed SegFormer on "
                "mean mIoU, but did not improve the original ECSegFormer mean, "
                "did not become as stable as SegFormer, and did not eliminate "
                "all same-seed losses."
            ),
            "",
            "See `per_seed_results.csv`, `model_summary.csv`, "
            "`paired_tests.csv`, and `analysis_summary.json` for full details.",
            "",
        ]
    )
    (args.output_dir / "analysis_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
