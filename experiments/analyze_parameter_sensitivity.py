"""Validate and summarize the reviewer-requested EEM/CAM/loss sensitivity study."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

SEEDS = (2041, 2042, 2043)
CONFIGS = {
    "sensitivity_eem_d123": ("EEM dilation", "{1,2,3}"),
    "sensitivity_eem_d124": ("EEM dilation", "{1,2,4} (default)"),
    "sensitivity_eem_d135": ("EEM dilation", "{1,3,5}"),
    "sensitivity_cam_s075": ("CAM strength", "0.75×"),
    "sensitivity_cam_s100": ("CAM strength", "1.00× (default)"),
    "sensitivity_cam_s125": ("CAM strength", "1.25×"),
    "sensitivity_loss_focal": ("Loss weights", "Focal/Dice/IoU = 0.60/0.25/0.15"),
    "sensitivity_loss_default": ("Loss weights", "Focal/Dice/IoU = 0.50/0.30/0.20 (default)"),
    "sensitivity_loss_region": ("Loss weights", "Focal/Dice/IoU = 0.40/0.35/0.25"),
}
DEFAULT = {
    "EEM dilation": "sensitivity_eem_d124",
    "CAM strength": "sensitivity_cam_s100",
    "Loss weights": "sensitivity_loss_default",
}
T975_DF2 = 4.302652729


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def summary(values: list[float]) -> dict[str, float]:
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    margin = T975_DF2 * sd / math.sqrt(len(values))
    return {"mean": mean, "sample_sd": sd, "ci95_low": mean - margin, "ci95_high": mean + margin}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/parameter_sensitivity_v1"))
    args = parser.parse_args()
    rows: list[dict] = []
    errors: list[str] = []
    manifest_hashes: set[str] = set()
    for config_name, (factor, setting) in CONFIGS.items():
        values: list[float] = []
        for seed in SEEDS:
            run_dir = args.root / config_name / f"seed_{seed}"
            metadata_path = run_dir / "run_metadata.json"
            metrics_path = run_dir / "test_metrics.json"
            if not metadata_path.is_file() or not metrics_path.is_file():
                errors.append(f"missing completed result: {config_name}, seed {seed}")
                continue
            metadata, metrics = read_json(metadata_path), read_json(metrics_path)
            if metadata.get("status") != "completed" or metrics.get("smoke_test"):
                errors.append(f"invalid run: {config_name}, seed {seed}")
                continue
            if metadata.get("seed") != seed or metrics.get("seed") != seed:
                errors.append(f"seed mismatch: {config_name}, seed {seed}")
                continue
            manifest_hashes.add(str(metadata.get("manifest_sha256")))
            values.append(float(metrics["mean_iou"]))
        if len(values) == len(SEEDS):
            rows.append({"factor": factor, "configuration": config_name, "setting": setting, **summary(values)})
    if errors or len(manifest_hashes) != 1:
        payload = {"valid": False, "errors": errors, "manifest_hashes": sorted(manifest_hashes)}
        (args.root / "sensitivity_validation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise SystemExit("Sensitivity validation failed; see sensitivity_validation.json")
    for row in rows:
        default = next(item for item in rows if item["configuration"] == DEFAULT[row["factor"]])
        row["delta_vs_default"] = row["mean"] - default["mean"]
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "sensitivity_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    lines = ["# ECSegFormer parameter sensitivity (3 seeds per unique configuration)", "", "Primary metric: test mIoU. Values are mean ± sample SD; 95% t intervals use df=2.", ""]
    for factor in ("EEM dilation", "CAM strength", "Loss weights"):
        lines.extend([f"## {factor}", "", "| Setting | mIoU mean ± SD | 95% CI | Δ vs default |", "|---|---:|---:|---:|"])
        for row in (item for item in rows if item["factor"] == factor):
            lines.append(f"| {row['setting']} | {row['mean']:.4f} ± {row['sample_sd']:.4f} | [{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | {row['delta_vs_default']:+.4f} |")
        lines.append("")
    lines.append("Each factor has its own explicitly named default configuration, independently trained with the same three seeds.")
    (args.root / "sensitivity_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"valid": True, "runs": len(CONFIGS) * len(SEEDS), "summary": str(args.root / "sensitivity_summary.csv")}, indent=2))


if __name__ == "__main__":
    main()
