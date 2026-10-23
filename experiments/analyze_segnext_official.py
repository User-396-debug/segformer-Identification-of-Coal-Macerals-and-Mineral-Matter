#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev


DEFAULT_SEEDS = (2026, 2027, 2028, 2029, 2030)
T_95_DF4 = 2.7764451051977987


def summarize(values):
    n = len(values)
    avg = mean(values)
    sd = stdev(values) if n > 1 else 0.0
    margin = T_95_DF4 * sd / math.sqrt(n) if n == 5 else None
    return {
        "n": n,
        "mean": avg,
        "sample_sd": sd,
        "ci95_low": avg - margin if margin is not None else None,
        "ci95_high": avg + margin if margin is not None else None,
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/segnext_official/fixed_split_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/segnext_official/statistics"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()

    rows = []
    raw = {}
    manifest_hashes = set()
    for seed in args.seeds:
        metrics_path = args.root / f"seed_{seed}" / "test_metrics.json"
        metadata_path = args.root / f"seed_{seed}" / "run_metadata.json"
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing completed seed result: {metrics_path}")
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing run metadata: {metadata_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metrics.get("smoke_test"):
            raise RuntimeError(f"Refusing to aggregate smoke-test result: {metrics_path}")
        if metadata.get("status") != "completed" or metadata.get("smoke_test"):
            raise RuntimeError(f"Run is not a completed formal run: {metadata_path}")
        if metadata.get("validation_precision") != "fp32":
            raise RuntimeError(f"Validation was not FP32: {metadata_path}")
        if metadata.get("test_precision") != "fp32":
            raise RuntimeError(f"Test was not FP32: {metadata_path}")
        manifest_hashes.add(metrics["manifest_sha256"])
        raw[str(seed)] = metrics
        row = {
            "seed": seed,
            "mean_iou": float(metrics["mean_iou"]),
            "mean_dice": float(metrics["mean_dice"]),
            "pixel_accuracy": float(metrics["pixel_accuracy"]),
            "test_loss": float(metrics.get("test_loss", metrics["loss"])),
        }
        for item in metrics.get("per_class", []):
            idx = int(item["class_id"])
            row[f"class_{idx}_iou"] = float(item["iou"])
            row[f"class_{idx}_dice"] = float(item["dice"])
        rows.append(row)

    if len(manifest_hashes) != 1:
        raise RuntimeError(f"Seeds used different manifests: {sorted(manifest_hashes)}")
    metric_names = [key for key in rows[0] if key != "seed"]
    summary = {
        "model": "SegNeXt-T (official MSCAN-T + LightHamHead)",
        "split": "fixed_split_v1",
        "manifest_sha256": next(iter(manifest_hashes)),
        "train_precision": "amp_fp16",
        "validation_precision": "fp32",
        "test_precision": "fp32",
        "seeds": args.seeds,
        "metrics": {
            name: summarize([row[name] for row in rows]) for name in metric_names
        },
        "per_seed": raw,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (args.output / "per_seed.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "sample_sd", "ci95_low", "ci95_high"])
        for name in metric_names:
            item = summary["metrics"][name]
            writer.writerow(
                [
                    name,
                    item["mean"],
                    item["sample_sd"],
                    item["ci95_low"],
                    item["ci95_high"],
                ]
            )
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
