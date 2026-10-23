"""Summarise the preregistered formal split-robustness experiment."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
FORMAL_ROOT = ROOT / "outputs" / "formal_split_robustness"
OUTPUT = FORMAL_ROOT / "statistics"
SPLITS = [f"formal_robustness_{index:02d}" for index in range(1, 6)]
SEEDS = [2026, 2027, 2028]
MODELS = ["segformer", "ecsegformer"]
CLASS_NAMES = ["background", "red", "green", "blue", "yellow"]


def descriptive(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    n = len(array)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if n > 1 else 0.0
    sem = sd / math.sqrt(n) if n else math.nan
    critical = float(stats.t.ppf(0.975, n - 1)) if n > 1 else math.nan
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "sem": sem,
        "ci95_low": mean - critical * sem if n > 1 else mean,
        "ci95_high": mean + critical * sem if n > 1 else mean,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def paired_test(seg: list[float], ecs: list[float]) -> dict:
    left = np.asarray(seg, dtype=float)
    right = np.asarray(ecs, dtype=float)
    delta = right - left
    t_result = stats.ttest_rel(right, left)
    try:
        w_result = stats.wilcoxon(
            right, left, alternative="two-sided", method="auto"
        )
        w_stat, w_p = float(w_result.statistic), float(w_result.pvalue)
    except ValueError:
        w_stat, w_p = math.nan, math.nan
    return {
        "segformer": descriptive(left.tolist()),
        "ecsegformer": descriptive(right.tolist()),
        "delta_ecs_minus_seg": descriptive(delta.tolist()),
        "paired_t_statistic": float(t_result.statistic),
        "paired_t_p_two_sided": float(t_result.pvalue),
        "wilcoxon_statistic": w_stat,
        "wilcoxon_p_two_sided": w_p,
        "ecs_wins": int((delta > 0).sum()),
        "ties": int((delta == 0).sum()),
        "pairs": len(delta),
    }


def read_metric(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_formal_rows() -> list[dict]:
    rows: list[dict] = []
    for split_id in SPLITS:
        for seed in SEEDS:
            for model in MODELS:
                run_dir = FORMAL_ROOT / split_id / model / f"seed_{seed}"
                metrics = read_metric(run_dir / "test_metrics.json")
                metadata = read_metric(run_dir / "run_metadata.json")
                per_class = metrics["per_class"]
                row = {
                    "split_id": split_id,
                    "seed": seed,
                    "model": model,
                    "mean_iou": float(metrics["mean_iou"]),
                    "pixel_accuracy": float(metrics["pixel_accuracy"]),
                    "mean_dice": float(metrics["mean_dice"]),
                    "runtime_minutes": (
                        float(metadata["completed_at_unix"])
                        - float(metadata["started_at_unix"])
                    )
                    / 60.0,
                }
                for index, name in enumerate(CLASS_NAMES):
                    row[f"{name}_iou"] = float(per_class[index]["iou"])
                    row[f"{name}_precision"] = float(per_class[index]["precision"])
                    row[f"{name}_recall"] = float(per_class[index]["recall"])
                rows.append(row)
    if len(rows) != 30:
        raise ValueError(f"Expected 30 formal runs, found {len(rows)}")
    return rows


def load_challenge_rows() -> list[dict]:
    rows: list[dict] = []
    for split_id in SPLITS:
        path = (
            FORMAL_ROOT
            / split_id
            / "challenge_diagnostics"
            / "per_image_green_diagnostics.json"
        )
        payload = read_metric(path)
        for row in payload["records"]:
            if row["group_id"] == "KW_4_13" and row["stage"] == "final":
                rows.append(
                    {
                        "split_id": split_id,
                        "seed": int(row["seed"]),
                        "model": row["model"],
                        "green_iou": float(row["green_iou"]),
                        # The metric implementation returns None when the
                        # model predicts no green pixels; for aggregate
                        # challenge reporting this is a zero-precision case.
                        "green_precision": float(row["green_precision"] or 0.0),
                        "green_recall": float(row["green_recall"] or 0.0),
                        **{
                            key: row[key]
                            for key in row
                            if key.startswith("true_green_pred_")
                        },
                    }
                )
    if len(rows) != 30:
        raise ValueError(f"Expected 30 KW_4_13 records, found {len(rows)}")
    return rows


def metric_pairs(rows: list[dict], metric: str) -> tuple[list[float], list[float]]:
    indexed = {
        (row["split_id"], row["seed"], row["model"]): float(row[metric])
        for row in rows
    }
    seg, ecs = [], []
    for split_id in SPLITS:
        for seed in SEEDS:
            seg.append(indexed[(split_id, seed, "segformer")])
            ecs.append(indexed[(split_id, seed, "ecsegformer")])
    return seg, ecs


def comparison_from_directory(root: Path, seeds: list[int]) -> dict:
    seg, ecs = [], []
    for seed in seeds:
        seg.append(
            read_metric(root / "segformer" / f"seed_{seed}" / "test_metrics.json")[
                "mean_iou"
            ]
        )
        ecs.append(
            read_metric(root / "ecsegformer" / f"seed_{seed}" / "test_metrics.json")[
                "mean_iou"
            ]
        )
    return paired_test(seg, ecs)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = load_formal_rows()
    challenge_rows = load_challenge_rows()
    write_csv(OUTPUT / "per_run_results.csv", rows)
    write_csv(OUTPUT / "kw413_challenge_results.csv", challenge_rows)

    indexed = {
        (row["split_id"], row["seed"], row["model"]): row for row in rows
    }
    paired_rows: list[dict] = []
    for split_id in SPLITS:
        for seed in SEEDS:
            seg = indexed[(split_id, seed, "segformer")]
            ecs = indexed[(split_id, seed, "ecsegformer")]
            paired_rows.append(
                {
                    "split_id": split_id,
                    "seed": seed,
                    "segformer_miou": seg["mean_iou"],
                    "ecsegformer_miou": ecs["mean_iou"],
                    "delta_miou": ecs["mean_iou"] - seg["mean_iou"],
                    "segformer_green_iou": seg["green_iou"],
                    "ecsegformer_green_iou": ecs["green_iou"],
                    "delta_green_iou": ecs["green_iou"] - seg["green_iou"],
                    "segformer_green_precision": seg["green_precision"],
                    "ecsegformer_green_precision": ecs["green_precision"],
                    "segformer_green_recall": seg["green_recall"],
                    "ecsegformer_green_recall": ecs["green_recall"],
                }
            )
    write_csv(OUTPUT / "paired_results.csv", paired_rows)

    split_rows: list[dict] = []
    for split_id in SPLITS:
        result = {"split_id": split_id}
        for model in MODELS:
            subset = [
                row for row in rows if row["split_id"] == split_id and row["model"] == model
            ]
            for metric in ("mean_iou", "green_iou", "green_precision", "green_recall"):
                summary = descriptive([row[metric] for row in subset])
                result[f"{model}_{metric}_mean"] = summary["mean"]
                result[f"{model}_{metric}_sd"] = summary["sd"]
        result["delta_miou_mean"] = (
            result["ecsegformer_mean_iou_mean"] - result["segformer_mean_iou_mean"]
        )
        result["delta_green_iou_mean"] = (
            result["ecsegformer_green_iou_mean"] - result["segformer_green_iou_mean"]
        )
        split_rows.append(result)
    write_csv(OUTPUT / "per_split_summary.csv", split_rows)

    run_level_miou = paired_test(*metric_pairs(rows, "mean_iou"))
    split_level_miou = paired_test(
        [row["segformer_mean_iou_mean"] for row in split_rows],
        [row["ecsegformer_mean_iou_mean"] for row in split_rows],
    )
    class_statistics = {}
    for name in CLASS_NAMES:
        class_statistics[name] = paired_test(*metric_pairs(rows, f"{name}_iou"))
    green_statistics = {
        metric: paired_test(*metric_pairs(rows, metric))
        for metric in ("green_iou", "green_precision", "green_recall")
    }

    challenge_index = {
        (row["split_id"], row["seed"], row["model"]): row
        for row in challenge_rows
    }
    challenge_statistics = {}
    for metric in ("green_iou", "green_precision", "green_recall"):
        seg, ecs = [], []
        for split_id in SPLITS:
            for seed in SEEDS:
                seg.append(challenge_index[(split_id, seed, "segformer")][metric])
                ecs.append(challenge_index[(split_id, seed, "ecsegformer")][metric])
        challenge_statistics[metric] = paired_test(seg, ecs)
    challenge_by_split = []
    for split_id in SPLITS:
        with (
            ROOT / "experiments" / "manifests" / f"{split_id}.csv"
        ).open(newline="", encoding="utf-8") as handle:
            manifest_rows = list(csv.DictReader(handle))
        broad_sid_train = sorted(
            {
                row["group_id"]
                for row in manifest_rows
                if row["split"] == "train"
                and row["group_id"] in {"KW_4_14", "KW_4_15"}
            }
        )
        item = {
            "split_id": split_id,
            "broad_sid_in_train": broad_sid_train,
        }
        for model in MODELS:
            subset = [
                row
                for row in challenge_rows
                if row["split_id"] == split_id and row["model"] == model
            ]
            item[f"{model}_green_iou"] = descriptive(
                [row["green_iou"] for row in subset]
            )
        challenge_by_split.append(item)

    model_split_variation = {}
    for model in MODELS:
        split_means = [
            row[f"{model}_mean_iou_mean"] for row in split_rows
        ]
        within_split_sds = [
            row[f"{model}_mean_iou_sd"] for row in split_rows
        ]
        model_split_variation[model] = {
            "sd_of_five_split_means": float(np.std(split_means, ddof=1)),
            "mean_within_split_seed_sd": float(np.mean(within_split_sds)),
            "range_of_split_means": [
                float(min(split_means)),
                float(max(split_means)),
            ],
        }

    fixed_significance = comparison_from_directory(
        ROOT / "outputs" / "main_comparison", list(range(2026, 2036))
    )
    pilot_split02 = comparison_from_directory(
        ROOT / "outputs" / "split_robustness" / "robustness_split_02", SEEDS
    )

    starts, completes = [], []
    for split_id in SPLITS:
        for seed in SEEDS:
            for model in MODELS:
                metadata = read_metric(
                    FORMAL_ROOT
                    / split_id
                    / model
                    / f"seed_{seed}"
                    / "run_metadata.json"
                )
                starts.append(float(metadata["started_at_unix"]))
                completes.append(float(metadata["completed_at_unix"]))

    summary = {
        "design": {
            "formal_splits": 5,
            "seeds_per_split": 3,
            "models": MODELS,
            "paired_runs": 15,
            "total_runs": 30,
            "main_test_policy": (
                "representative Sid retained; KW_4_13 excluded from main mIoU "
                "and reported as a fixed independent challenge"
            ),
        },
        "formal_run_level_miou": run_level_miou,
        "formal_split_level_miou": split_level_miou,
        "per_class_iou": class_statistics,
        "green_metrics": green_statistics,
        "kw413_challenge": challenge_statistics,
        "kw413_challenge_by_split": challenge_by_split,
        "model_split_variation": model_split_variation,
        "fixed_split_significance_reference": fixed_significance,
        "pilot_split02_reference": pilot_split02,
        "total_wall_clock_hours": (max(completes) - min(starts)) / 3600.0,
        "total_gpu_run_hours": sum(row["runtime_minutes"] for row in rows) / 60.0,
    }
    (OUTPUT / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    r = run_level_miou
    s = split_level_miou
    g = green_statistics
    c = challenge_statistics
    lines = [
        "# Formal split-robustness analysis",
        "",
        "## Design",
        "",
        "- Five frozen, subtype/pixel-balanced group splits x three seeds x two models (30 runs; 15 paired comparisons).",
        "- Every main test split retains one representative broad Sid group. `KW_4_13` is a fixed independent extreme-Sid challenge and is not included in main mIoU.",
        "- All manifests were frozen before these 30 trainings; all model/training settings match the main significance experiment.",
        "",
        "## Main result",
        "",
        f"- Across 15 paired runs, SegFormer mIoU = {fmt(r['segformer']['mean'])} +/- {fmt(r['segformer']['sd'])}; "
        f"ECSegFormer = {fmt(r['ecsegformer']['mean'])} +/- {fmt(r['ecsegformer']['sd'])}.",
        f"- Mean paired improvement = {fmt(r['delta_ecs_minus_seg']['mean'])} "
        f"(95% CI {fmt(r['delta_ecs_minus_seg']['ci95_low'])} to {fmt(r['delta_ecs_minus_seg']['ci95_high'])}); "
        f"ECSegFormer wins {r['ecs_wins']}/{r['pairs']} pairs.",
        f"- Paired t-test over 15 runs: p={r['paired_t_p_two_sided']:.3g}; "
        f"Wilcoxon p={r['wilcoxon_p_two_sided']:.3g}.",
        f"- Conservative split-level test (three seeds averaged within each split, n=5): "
        f"mean improvement={fmt(s['delta_ecs_minus_seg']['mean'])}, "
        f"95% CI {fmt(s['delta_ecs_minus_seg']['ci95_low'])} to {fmt(s['delta_ecs_minus_seg']['ci95_high'])}, "
        f"paired t p={s['paired_t_p_two_sided']:.3g}, Wilcoxon p={s['wilcoxon_p_two_sided']:.3g}.",
        "",
        "## Green class",
        "",
        f"- Green IoU: SegFormer {fmt(g['green_iou']['segformer']['mean'])}, "
        f"ECSegFormer {fmt(g['green_iou']['ecsegformer']['mean'])}, "
        f"paired Delta={fmt(g['green_iou']['delta_ecs_minus_seg']['mean'])}.",
        f"- Green precision: {fmt(g['green_precision']['segformer']['mean'])} -> "
        f"{fmt(g['green_precision']['ecsegformer']['mean'])}; "
        f"green recall: {fmt(g['green_recall']['segformer']['mean'])} -> "
        f"{fmt(g['green_recall']['ecsegformer']['mean'])}.",
        "",
        "## KW_4_13 independent challenge",
        "",
        f"- Across all formal checkpoints, green IoU is SegFormer {fmt(c['green_iou']['segformer']['mean'])} "
        f"and ECSegFormer {fmt(c['green_iou']['ecsegformer']['mean'])}, but dispersion is extreme "
        f"(SD {fmt(c['green_iou']['segformer']['sd'])} and {fmt(c['green_iou']['ecsegformer']['sd'])}).",
        f"- The paired challenge difference is not conclusive: Delta={fmt(c['green_iou']['delta_ecs_minus_seg']['mean'])}, "
        f"95% CI {fmt(c['green_iou']['delta_ecs_minus_seg']['ci95_low'])} to "
        f"{fmt(c['green_iou']['delta_ecs_minus_seg']['ci95_high'])}, p={c['green_iou']['paired_t_p_two_sided']:.3g}.",
        "- In formal_robustness_04, where disseminated `KW_4_14` rather than broad pale `KW_4_15` is in training, both models are essentially zero on `KW_4_13` for all three seeds. Other splits often, but not always, generalise strongly.",
        "",
        "## Interpretation",
        "",
        "- The enhancement advantage is not an accidental benefit of one fixed train/test allocation: it is positive in every formal paired run and remains positive when the split, rather than each run, is treated as the independent unit.",
        "- The adjusted split protocol produces a stable estimate for the ordinary target distribution while still testing representative Sid.",
        "- Because the decision to isolate `KW_4_13` followed the pilot diagnosis, it must be disclosed as a pilot-informed challenge-set policy. The formal mIoU must not be described as performance on all possible Sid morphologies.",
        "- Report the main formal result together with the highly unstable `KW_4_13` challenge. This supports ordinary-distribution robustness while exposing that broad-Sid generalisation still depends strongly on morphology coverage and random seed.",
        "",
        f"Wall-clock duration: {summary['total_wall_clock_hours']:.2f} h; summed GPU run time: {summary['total_gpu_run_hours']:.2f} h.",
    ]
    (OUTPUT / "analysis_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
