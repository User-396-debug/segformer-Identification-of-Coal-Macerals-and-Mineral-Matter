#!/usr/bin/env python3
"""Minimal definition checks for BoundaryIoUMeter."""

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.evaluate_boundary_metrics import BoundaryIoUMeter


identical = torch.zeros((1, 16, 16), dtype=torch.long)
identical[:, 4:12, 4:12] = 1
meter = BoundaryIoUMeter(5, width_pixels=2)
meter.update(identical, identical)
identical_result = meter.compute(16, 16)
identical_iou = identical_result["per_class"][1]["boundary_iou"]
assert abs(identical_iou - 1.0) < 1e-12

disjoint = torch.zeros_like(identical)
disjoint[:, 1:3, 1:3] = 1
meter = BoundaryIoUMeter(5, width_pixels=2)
meter.update(identical, disjoint)
disjoint_result = meter.compute(16, 16)
disjoint_iou = disjoint_result["per_class"][1]["boundary_iou"]
assert disjoint_iou == 0.0

ratio_meter = BoundaryIoUMeter(5, dilation_ratio=0.02)
assert ratio_meter._resolve_width(512, 512) == 14

print(
    {
        "identical_class1_iou": identical_iou,
        "disjoint_class1_iou": disjoint_iou,
        "canonical_width_at_512": 14,
    }
)
