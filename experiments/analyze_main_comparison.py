#!/usr/bin/env python3
"""Validate and summarize the 2-model x 10-seed main comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats


MODELS = ("segformer", "ecsegformer")
CLASS_NAMES = ("background", "red", "green", "blue", "yellow")
EXPECTED_MANIFEST_SHA256 = (
    "3992b99fc6b659627996059dd2e558c55b9174b612a76fb5d97c726b5accc2ea"
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_sd_ci(values: Iterable[float], confidence: float = 0.95) -> dict[str, float]:
    data = np.asarray(list(values), dtype=float)
    n = data.size
    mean = float(np.mean(data))
    sd = float(np.std(data, ddof=1)) if n > 1 else 0.0
    if n > 1:
        critical = float(stats.t.ppf((1.0 + confidence) / 2.0, n - 1))
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


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment, returned in original order."""
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        candidate = (count - rank) * p_values[int(index)]
        running_max = max(running_max, candidate)
        adjusted[int(index)] = min(1.0, running_max)
    return adjusted.tolist()


def paired_test(ec_values: list[float], seg_values: list[float]) -> dict[str, float]:
    ec = np.asarray(ec_values, dtype=float)
    seg = np.asarray(seg_values, dtype=float)
    differences = ec - seg
    summary = mean_sd_ci(differences)
    t_result = stats.ttest_rel(ec, seg, alternative="two-sided")
    try:
        wilcoxon_result = stats.wilcoxon(
            differences,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        wilcoxon_stat = float(wilcoxon_result.statistic)
        wilcoxon_p = float(wilcoxon_result.pvalue)
    except ValueError:
        wilcoxon_stat = 0.0
        wilcoxon_p = 1.0
    sd = summary["sample_sd"]
    return {
        **summary,
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_p": float(t_result.pvalue),
        "wilcoxon_statistic": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
        "cohen_dz": summary["mean"] / sd if sd else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/main_comparison"),
        help="Directory containing model/seed_* result folders.",
    )
    parser.add_argument("--seed-start", type=int, default=2026)
    parser.add_argument("--seed-end", type=int, default=2035)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/main_comparison/statistics"),
    )
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_end + 1))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_errors: list[str] = []
    runs: dict[str, dict[int, dict[str, Any]]] = {model: {} for model in MODELS}
    manifest_hashes: set[str] = set()
    config_hashes: set[str] = set()
    pretrained_hashes: set[str] = set()

    for model in MODELS:
        for seed in seeds:
            run_dir = args.root / model / f"seed_{seed}"
            required = (
                run_dir / "run_metadata.json",
                run_dir / "test_metrics.json",
                run_dir / "history.csv",
                run_dir / "config_resolved.yaml",
            )
            for path in required:
                if not path.is_file():
                    validation_errors.append(f"Missing file: {path}")
            if any(not path.is_file() for path in required):
                continue

            metadata = read_json(required[0])
            metrics = read_json(required[1])
            with required[2].open("r", encoding="utf-8-sig", newline="") as handle:
                history = list(csv.DictReader(handle))

            prefix = f"{model} seed={seed}"
            checks = {
                "status=completed": metadata.get("status") == "completed",
                "model matches": metadata.get("model") == model,
                "seed matches": metadata.get("seed") == seed,
                "smoke_test=false": metadata.get("smoke_test") is False,
                "matched encoder=170": metadata.get("pretraining_load", {}).get(
                    "matched_encoder_tensors"
                )
                == 170,
                "encoder count=170": metadata.get("pretraining_load", {}).get(
                    "encoder_tensor_count"
                )
                == 170,
                "test size=145": metadata.get("dataset_sizes", {}).get("test") == 145,
                "processed_batches=37": metrics.get("processed_batches") == 37,
                "test split": metrics.get("dataset_split") == "test",
                "metrics smoke_test=false": metrics.get("smoke_test") is False,
                "history has 50 epochs": len(history) == 50,
                "final epoch=50": bool(history) and int(history[-1]["epoch"]) == 50,
                "manifest hash matches expected": metadata.get("manifest_sha256")
                == EXPECTED_MANIFEST_SHA256,
                "metrics manifest matches metadata": metrics.get("manifest_sha256")
                == metadata.get("manifest_sha256"),
            }
            for label, passed in checks.items():
                if not passed:
                    validation_errors.append(f"{prefix}: failed {label}")

            manifest_hashes.add(str(metadata.get("manifest_sha256")))
            config_hashes.add(str(metadata.get("config_sha256")))
            pretrained_hashes.add(str(metadata.get("pretrained_sha256")))

            per_class = metrics.get("per_class", [])
            if len(per_class) != len(CLASS_NAMES):
                validation_errors.append(
                    f"{prefix}: expected 5 class metrics, got {len(per_class)}"
                )

            record: dict[str, Any] = {
                "model": model,
                "seed": seed,
                "mean_iou": float(metrics["mean_iou"]),
                "mean_dice": float(metrics["mean_dice"]),
                "pixel_accuracy": float(metrics["pixel_accuracy"]),
                "best_val_mean_iou": float(metadata["best_val_mean_iou"]),
                "best_epoch": max(
                    history, key=lambda row: float(row["val_mean_iou"])
                )["epoch"],
                "final_val_mean_iou": float(history[-1]["val_mean_iou"]),
                "last10_val_mean_iou": float(
                    np.mean(
                        [float(row["val_mean_iou"]) for row in history[-10:]]
                    )
                ),
                "runtime_seconds": float(metadata["completed_at_unix"])
                - float(metadata["started_at_unix"]),
            }
            record["best_to_final_val_gap"] = (
                record["best_val_mean_iou"] - record["final_val_mean_iou"]
            )
            best_threshold = 0.95 * record["best_val_mean_iou"]
            record["first_epoch_at_95pct_best"] = next(
                int(row["epoch"])
                for row in history
                if float(row["val_mean_iou"]) >= best_threshold
            )
            for class_index, class_name in enumerate(CLASS_NAMES):
                record[f"iou_{class_name}"] = float(per_class[class_index]["iou"])
            runs[model][seed] = record

    if len(manifest_hashes) != 1:
        validation_errors.append(f"Manifest hashes differ: {sorted(manifest_hashes)}")
    if len(config_hashes) != 1:
        validation_errors.append(f"Config hashes differ: {sorted(config_hashes)}")
    if len(pretrained_hashes) != 1:
        validation_errors.append(
            f"Pretrained hashes differ: {sorted(pretrained_hashes)}"
        )

    validation_report = {
        "valid": not validation_errors,
        "expected_runs": len(MODELS) * len(seeds),
        "loaded_runs": sum(len(model_runs) for model_runs in runs.values()),
        "errors": validation_errors,
        "manifest_hashes": sorted(manifest_hashes),
        "config_hashes": sorted(config_hashes),
        "pretrained_hashes": sorted(pretrained_hashes),
    }
    with (output_dir / "validation_report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(validation_report, handle, ensure_ascii=False, indent=2)

    if validation_errors:
        raise SystemExit(
            "Validation failed; see "
            f"{output_dir / 'validation_report.json'} for details."
        )

    metric_names = [
        "mean_iou",
        "mean_dice",
        "pixel_accuracy",
        *(f"iou_{name}" for name in CLASS_NAMES),
    ]
    secondary_metrics = [name for name in metric_names if name != "mean_iou"]

    per_seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        seg = runs["segformer"][seed]
        ec = runs["ecsegformer"][seed]
        row: dict[str, Any] = {"seed": seed}
        for metric in metric_names:
            row[f"segformer_{metric}"] = seg[metric]
            row[f"ecsegformer_{metric}"] = ec[metric]
            row[f"difference_{metric}"] = ec[metric] - seg[metric]
        row.update(
            {
                "segformer_best_val_mean_iou": seg["best_val_mean_iou"],
                "ecsegformer_best_val_mean_iou": ec["best_val_mean_iou"],
                "segformer_best_epoch": int(seg["best_epoch"]),
                "ecsegformer_best_epoch": int(ec["best_epoch"]),
                "segformer_final_val_mean_iou": seg["final_val_mean_iou"],
                "ecsegformer_final_val_mean_iou": ec["final_val_mean_iou"],
                "segformer_last10_val_mean_iou": seg["last10_val_mean_iou"],
                "ecsegformer_last10_val_mean_iou": ec["last10_val_mean_iou"],
                "segformer_best_to_final_val_gap": seg[
                    "best_to_final_val_gap"
                ],
                "ecsegformer_best_to_final_val_gap": ec[
                    "best_to_final_val_gap"
                ],
                "segformer_first_epoch_at_95pct_best": seg[
                    "first_epoch_at_95pct_best"
                ],
                "ecsegformer_first_epoch_at_95pct_best": ec[
                    "first_epoch_at_95pct_best"
                ],
                "segformer_runtime_seconds": seg["runtime_seconds"],
                "ecsegformer_runtime_seconds": ec["runtime_seconds"],
                "miou_winner": (
                    "ecsegformer"
                    if ec["mean_iou"] > seg["mean_iou"]
                    else "segformer"
                    if ec["mean_iou"] < seg["mean_iou"]
                    else "tie"
                ),
            }
        )
        per_seed_rows.append(row)
    write_csv(output_dir / "per_seed_results.csv", per_seed_rows)

    model_summary_rows: list[dict[str, Any]] = []
    model_summaries: dict[str, dict[str, dict[str, float]]] = {}
    for model in MODELS:
        model_summaries[model] = {}
        for metric in [
            *metric_names,
            "best_val_mean_iou",
            "final_val_mean_iou",
            "last10_val_mean_iou",
            "best_to_final_val_gap",
            "best_epoch",
            "first_epoch_at_95pct_best",
            "runtime_seconds",
        ]:
            summary = mean_sd_ci([runs[model][seed][metric] for seed in seeds])
            model_summaries[model][metric] = summary
            model_summary_rows.append({"model": model, "metric": metric, **summary})
    write_csv(output_dir / "model_summary.csv", model_summary_rows)

    paired_results: dict[str, dict[str, float]] = {}
    paired_rows: list[dict[str, Any]] = []
    for metric in metric_names:
        result = paired_test(
            [runs["ecsegformer"][seed][metric] for seed in seeds],
            [runs["segformer"][seed][metric] for seed in seeds],
        )
        paired_results[metric] = result
        paired_rows.append(
            {
                "metric": metric,
                **result,
                "paired_t_p_holm": result["paired_t_p"],
                "wilcoxon_p_holm": result["wilcoxon_p"],
            }
        )

    secondary_rows = [
        row for row in paired_rows if row["metric"] in secondary_metrics
    ]
    adjusted_t = holm_adjust([float(row["paired_t_p"]) for row in secondary_rows])
    adjusted_w = holm_adjust(
        [float(row["wilcoxon_p"]) for row in secondary_rows]
    )
    for row, t_adjusted, w_adjusted in zip(
        secondary_rows, adjusted_t, adjusted_w, strict=True
    ):
        row["paired_t_p_holm"] = t_adjusted
        row["wilcoxon_p_holm"] = w_adjusted
    write_csv(output_dir / "paired_tests.csv", paired_rows)

    ec_wins = sum(row["miou_winner"] == "ecsegformer" for row in per_seed_rows)
    seg_wins = sum(row["miou_winner"] == "segformer" for row in per_seed_rows)
    ties = len(seeds) - ec_wins - seg_wins
    primary = paired_results["mean_iou"]
    summary_payload = {
        "validation": validation_report,
        "model_summaries": model_summaries,
        "paired_results": paired_results,
        "secondary_holm_adjusted": {
            row["metric"]: {
                "paired_t_p_holm": row["paired_t_p_holm"],
                "wilcoxon_p_holm": row["wilcoxon_p_holm"],
            }
            for row in secondary_rows
        },
        "wins": {
            "ecsegformer": ec_wins,
            "segformer": seg_wins,
            "ties": ties,
        },
    }
    with (output_dir / "analysis_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary_payload, handle, ensure_ascii=False, indent=2)

    seg_primary = model_summaries["segformer"]["mean_iou"]
    ec_primary = model_summaries["ecsegformer"]["mean_iou"]
    report_lines = [
        "# Main comparison: 2 models × 10 paired seeds",
        "",
        "## Validation",
        "",
        (
            f"All {validation_report['loaded_runs']}/"
            f"{validation_report['expected_runs']} runs passed validation: "
            "completed, non-smoke-test, 170/170 encoder mapping, 145 test "
            "images, 37 test batches, 50 epochs, and identical manifest/config/"
            "pretrained hashes."
        ),
        "",
        "## Primary metric: test mIoU",
        "",
        "| Model | Mean | Sample SD | 95% t CI |",
        "|---|---:|---:|---:|",
        (
            f"| SegFormer | {fmt(seg_primary['mean'])} | "
            f"{fmt(seg_primary['sample_sd'])} | "
            f"[{fmt(seg_primary['ci95_low'])}, "
            f"{fmt(seg_primary['ci95_high'])}] |"
        ),
        (
            f"| ECSegFormer | {fmt(ec_primary['mean'])} | "
            f"{fmt(ec_primary['sample_sd'])} | "
            f"[{fmt(ec_primary['ci95_low'])}, "
            f"{fmt(ec_primary['ci95_high'])}] |"
        ),
        "",
        (
            "Paired difference (ECSegFormer − SegFormer): "
            f"{fmt(primary['mean'])}, SD={fmt(primary['sample_sd'])}, "
            f"95% CI=[{fmt(primary['ci95_low'])}, "
            f"{fmt(primary['ci95_high'])}]."
        ),
        (
            f"Paired t-test: t={fmt(primary['paired_t_statistic'])}, "
            f"p={primary['paired_t_p']:.8g}. Wilcoxon: "
            f"W={fmt(primary['wilcoxon_statistic'])}, "
            f"p={primary['wilcoxon_p']:.8g}. "
            f"Cohen dz={fmt(primary['cohen_dz'])}."
        ),
        (
            f"Per-seed mIoU wins: ECSegFormer {ec_wins}, "
            f"SegFormer {seg_wins}, ties {ties}."
        ),
        "",
        "## Secondary metrics",
        "",
        "| Metric | Mean paired difference | 95% CI | Paired t p | Holm p | Wilcoxon p | Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in secondary_rows:
        report_lines.append(
            f"| {row['metric']} | {fmt(float(row['mean']))} | "
            f"[{fmt(float(row['ci95_low']))}, "
            f"{fmt(float(row['ci95_high']))}] | "
            f"{float(row['paired_t_p']):.8g} | "
            f"{float(row['paired_t_p_holm']):.8g} | "
            f"{float(row['wilcoxon_p']):.8g} | "
            f"{float(row['wilcoxon_p_holm']):.8g} |"
        )

    report_lines.extend(
        [
            "",
            "## Runtime and training stability",
            "",
            "| Model | Runtime mean ± SD (min) | Best val mIoU mean ± SD | Final val mIoU mean ± SD |",
            "|---|---:|---:|---:|",
        ]
    )
    for model in MODELS:
        runtime = model_summaries[model]["runtime_seconds"]
        best_val = model_summaries[model]["best_val_mean_iou"]
        final_val = model_summaries[model]["final_val_mean_iou"]
        report_lines.append(
            f"| {model} | {runtime['mean'] / 60:.2f} ± "
            f"{runtime['sample_sd'] / 60:.2f} | "
            f"{fmt(best_val['mean'])} ± {fmt(best_val['sample_sd'])} | "
            f"{fmt(final_val['mean'])} ± {fmt(final_val['sample_sd'])} |"
        )

    runtime_ratio = (
        model_summaries["ecsegformer"]["runtime_seconds"]["mean"]
        / model_summaries["segformer"]["runtime_seconds"]["mean"]
    )
    report_lines.extend(
        [
            "",
            (
                f"ECSegFormer took {runtime_ratio:.2f}× the mean runtime of "
                "SegFormer."
            ),
            (
                "SegFormer best epoch: "
                f"{model_summaries['segformer']['best_epoch']['mean']:.1f} ± "
                f"{model_summaries['segformer']['best_epoch']['sample_sd']:.1f}; "
                "ECSegFormer: "
                f"{model_summaries['ecsegformer']['best_epoch']['mean']:.1f} ± "
                f"{model_summaries['ecsegformer']['best_epoch']['sample_sd']:.1f}."
            ),
            (
                "Mean best-to-final validation mIoU gap: SegFormer "
                f"{fmt(model_summaries['segformer']['best_to_final_val_gap']['mean'])}; "
                "ECSegFormer "
                f"{fmt(model_summaries['ecsegformer']['best_to_final_val_gap']['mean'])}. "
                "Best-checkpoint testing therefore remains important for both models."
            ),
            (
                "The test mIoU sample SD was "
                f"{fmt(ec_primary['sample_sd'])} for ECSegFormer versus "
                f"{fmt(seg_primary['sample_sd'])} for SegFormer, so the improved "
                "ECSegFormer mean is accompanied by greater seed sensitivity."
            ),
        ]
    )

    report_lines.extend(
        [
            "",
            "Detailed seed-level results and best epochs are in "
            "`per_seed_results.csv`; machine-readable statistics are in "
            "`analysis_summary.json`.",
            "",
        ]
    )
    (output_dir / "analysis_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
