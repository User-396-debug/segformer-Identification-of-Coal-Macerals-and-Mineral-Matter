"""Aggregate reviewer baseline reruns and measure held-out inference cost."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy import stats
from torch.utils.data import DataLoader

from experiments.train_main_comparison import (
    ManifestDataset,
    create_model,
    create_transforms,
    seed_worker,
    set_reproducibility,
)

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("unet", "deeplabv3plus", "pspnet")
SEEDS = (2026, 2027, 2028, 2029, 2030)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ci95(values: np.ndarray) -> list[float]:
    if len(values) < 2:
        return [float(values[0]), float(values[0])]
    sem = stats.sem(values)
    delta = stats.t.ppf(0.975, len(values) - 1) * sem
    return [float(values.mean() - delta), float(values.mean() + delta)]


def describe(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(len(array)),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)),
        "ci95": ci95(array),
    }


def inference_seconds_per_image(config: dict, model_name: str, checkpoint: Path, seed: int) -> float:
    device = torch.device("cuda")
    set_reproducibility(seed, bool(config["training"].get("deterministic", True)))
    _, eval_transform = create_transforms(config["data"]["image_size"])
    dataset = ManifestDataset(ROOT, ROOT / config["data"]["manifest"], "test", eval_transform)
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0, pin_memory=True)
    model, _ = create_model(config, model_name, device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"], strict=True)
    model.eval()
    with torch.inference_mode():
        warmup = torch.randn(2, 3, config["data"]["image_size"], config["data"]["image_size"], device=device)
        for _ in range(5):
            model(warmup)
        torch.cuda.synchronize()
        start = time.perf_counter()
        count = 0
        for images, _, _ in loader:
            model(images.to(device, non_blocking=True))
            count += len(images)
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / count


def paired_test(reference: list[float], candidate: list[float]) -> dict:
    delta = np.asarray(reference) - np.asarray(candidate)
    t = stats.ttest_rel(reference, candidate)
    w = stats.wilcoxon(reference, candidate, zero_method="wilcox")
    return {
        "mean_delta_ecsegformer_minus_baseline": float(delta.mean()),
        "delta_ci95": ci95(delta),
        "paired_t_p": float(t.pvalue),
        "wilcoxon_p": float(w.pvalue),
        "cohen_dz": float(delta.mean() / delta.std(ddof=1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", default="outputs/reviewer_baselines_fixed_split_v1")
    parser.add_argument("--ecsegformer-root", default="outputs/main_comparison/ecsegformer")
    parser.add_argument("--config", default="config/experiments/main_comparison.yaml")
    args = parser.parse_args()
    baseline_root = ROOT / args.baseline_root
    ec_root = ROOT / args.ecsegformer_root
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    summaries, comparisons, validation = {}, {}, {"valid": True, "errors": []}
    ec_metrics = {seed: load_json(ec_root / f"seed_{seed}" / "test_metrics.json") for seed in SEEDS}
    for model in MODELS:
        runs = []
        for seed in SEEDS:
            run_dir = baseline_root / model / f"seed_{seed}"
            metadata = load_json(run_dir / "run_metadata.json")
            metrics = load_json(run_dir / "test_metrics.json")
            if metadata.get("status") != "completed" or metrics.get("smoke_test"):
                validation["valid"] = False
                validation["errors"].append(f"{model}/seed_{seed}: incomplete or smoke test")
            if metadata.get("manifest_sha256") != ec_metrics[seed].get("manifest_sha256"):
                validation["valid"] = False
                validation["errors"].append(f"{model}/seed_{seed}: manifest mismatch")
            inference = inference_seconds_per_image(config, model, run_dir / "best.pth", seed)
            runs.append({"seed": seed, "metrics": metrics, "runtime_seconds": metadata["completed_at_unix"] - metadata["started_at_unix"], "inference_seconds_per_image": inference})
        summaries[model] = {
            "mean_iou": describe([run["metrics"]["mean_iou"] for run in runs]),
            "mean_dice": describe([run["metrics"]["mean_dice"] for run in runs]),
            "pixel_accuracy": describe([run["metrics"]["pixel_accuracy"] for run in runs]),
            "runtime_minutes": describe([run["runtime_seconds"] / 60 for run in runs]),
            "inference_milliseconds_per_image": describe([run["inference_seconds_per_image"] * 1000 for run in runs]),
            "runs": runs,
        }
        comparisons[model] = {
            "mean_iou": paired_test([ec_metrics[s]["mean_iou"] for s in SEEDS], [r["metrics"]["mean_iou"] for r in runs]),
            "mean_dice": paired_test([ec_metrics[s]["mean_dice"] for s in SEEDS], [r["metrics"]["mean_dice"] for r in runs]),
        }
    result = {"validation": validation, "models": summaries, "ecsegformer_vs_baselines": comparisons, "matched_seeds": list(SEEDS)}
    output = baseline_root / "statistics"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Reviewer baseline reruns", "", f"Validation: {'PASS' if validation['valid'] else 'FAIL'}.", "", "| Model | test mIoU (mean ± SD) | Dice (mean ± SD) | Training min (mean ± SD) | Inference ms/image (mean ± SD) |", "|---|---:|---:|---:|---:|"]
    for model, values in summaries.items():
        lines.append("| {model} | {miou:.4f} ± {miou_sd:.4f} | {dice:.4f} ± {dice_sd:.4f} | {runtime:.1f} ± {runtime_sd:.1f} | {infer:.1f} ± {infer_sd:.1f} |".format(model=model, miou=values['mean_iou']['mean'], miou_sd=values['mean_iou']['sample_sd'], dice=values['mean_dice']['mean'], dice_sd=values['mean_dice']['sample_sd'], runtime=values['runtime_minutes']['mean'], runtime_sd=values['runtime_minutes']['sample_sd'], infer=values['inference_milliseconds_per_image']['mean'], infer_sd=values['inference_milliseconds_per_image']['sample_sd']))
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
