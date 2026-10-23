#!/usr/bin/env python3
"""Compare original EEM with the lightweight EEM candidates."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.segformer_enhancements import (  # noqa: E402
    EdgeEnhancementModule,
    DirectSupervisedEdgeEnhancementModule,
    EfficientEdgeEnhancementModule,
)


def benchmark(module, logits, repeats=30, warmup=10):
    module.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            module(logits)
        torch.cuda.synchronize()
        timings = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            module(logits)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
        peak_memory = torch.cuda.max_memory_allocated()
    return {
        "latency_ms_mean": statistics.mean(timings),
        "latency_ms_sd": statistics.stdev(timings),
        "peak_allocated_mib": peak_memory / (1024**2),
    }


def main() -> None:
    device = torch.device("cuda")
    torch.manual_seed(2026)
    logits = torch.randn(1, 5, 512, 512, device=device)
    original = EdgeEnhancementModule(num_classes=5).to(device)
    improved = EfficientEdgeEnhancementModule(num_classes=5).to(device)
    improved_v2_1 = DirectSupervisedEdgeEnhancementModule(
        num_classes=5
    ).to(device)
    with torch.inference_mode():
        improved_output, boundary_logits = improved(
            logits, return_boundary=True
        )
        improved_v2_1_output, boundary_logits_v2_1 = improved_v2_1(
            logits, return_boundary=True
        )
    result = {
        "device": torch.cuda.get_device_name(0),
        "input_shape": list(logits.shape),
        "original": {
            "parameters": sum(p.numel() for p in original.parameters()),
            **benchmark(original, logits),
        },
        "improved": {
            "parameters": sum(p.numel() for p in improved.parameters()),
            "identity_max_abs_error_at_initialization": float(
                (improved_output - logits).abs().max()
            ),
            "boundary_shape": list(boundary_logits.shape),
            **benchmark(improved, logits),
        },
        "improved_v2_1": {
            "parameters": sum(
                p.numel() for p in improved_v2_1.parameters()
            ),
            "identity_max_abs_error_at_initialization": float(
                (improved_v2_1_output - logits).abs().max()
            ),
            "boundary_shape": list(boundary_logits_v2_1.shape),
            **benchmark(improved_v2_1, logits),
        },
    }
    result["parameter_reduction_fraction"] = 1.0 - (
        result["improved"]["parameters"]
        / result["original"]["parameters"]
    )
    result["latency_reduction_fraction"] = 1.0 - (
        result["improved"]["latency_ms_mean"]
        / result["original"]["latency_ms_mean"]
    )
    result["v2_1_parameter_reduction_fraction"] = 1.0 - (
        result["improved_v2_1"]["parameters"]
        / result["original"]["parameters"]
    )
    result["v2_1_latency_reduction_fraction"] = 1.0 - (
        result["improved_v2_1"]["latency_ms_mean"]
        / result["original"]["latency_ms_mean"]
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
