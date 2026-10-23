#!/usr/bin/env python3
"""Evaluate segmentation, Boundary-F1, and class-wise Boundary IoU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import math

import torch
import yaml
from torch.utils.data import DataLoader

from train_main_comparison import (
    PROJECT_ROOT,
    ManifestDataset,
    create_model,
    create_transforms,
    seed_worker,
    set_reproducibility,
    unwrap_outputs,
)
from utils.segmentation_metrics import ConfusionMatrixMeter


CLASS_NAMES = ("background", "red", "green", "blue", "yellow")


def class_boundaries(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    one_hot = torch.nn.functional.one_hot(
        labels, num_classes=num_classes
    ).permute(0, 3, 1, 2).float()
    dilated = torch.nn.functional.max_pool2d(
        one_hot, kernel_size=3, stride=1, padding=1
    )
    eroded = -torch.nn.functional.max_pool2d(
        -one_hot, kernel_size=3, stride=1, padding=1
    )
    boundaries = (dilated - eroded) > 0
    boundaries[:, :, 0, :] = False
    boundaries[:, :, -1, :] = False
    boundaries[:, :, :, 0] = False
    boundaries[:, :, :, -1] = False
    return boundaries


class BoundaryF1Meter:
    def __init__(self, num_classes: int, tolerance: int) -> None:
        self.num_classes = num_classes
        self.tolerance = tolerance
        self.predicted = torch.zeros(num_classes, dtype=torch.float64)
        self.target = torch.zeros(num_classes, dtype=torch.float64)
        self.matched_predicted = torch.zeros(
            num_classes, dtype=torch.float64
        )
        self.matched_target = torch.zeros(num_classes, dtype=torch.float64)

    def update(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> None:
        predicted_boundary = class_boundaries(
            predictions, self.num_classes
        )
        target_boundary = class_boundaries(targets, self.num_classes)
        kernel = 2 * self.tolerance + 1
        dilated_target = torch.nn.functional.max_pool2d(
            target_boundary.float(),
            kernel_size=kernel,
            stride=1,
            padding=self.tolerance,
        ).bool()
        dilated_predicted = torch.nn.functional.max_pool2d(
            predicted_boundary.float(),
            kernel_size=kernel,
            stride=1,
            padding=self.tolerance,
        ).bool()

        reduce_dims = (0, 2, 3)
        self.predicted += predicted_boundary.sum(
            dim=reduce_dims
        ).double().cpu()
        self.target += target_boundary.sum(dim=reduce_dims).double().cpu()
        self.matched_predicted += (
            predicted_boundary & dilated_target
        ).sum(dim=reduce_dims).double().cpu()
        self.matched_target += (
            target_boundary & dilated_predicted
        ).sum(dim=reduce_dims).double().cpu()

    def compute(self) -> dict:
        eps = 1e-12
        precision = self.matched_predicted / (self.predicted + eps)
        recall = self.matched_target / (self.target + eps)
        f1 = 2.0 * precision * recall / (precision + recall + eps)
        per_class = [
            {
                "class_id": index,
                "class_name": CLASS_NAMES[index],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "predicted_boundary_pixels": int(self.predicted[index]),
                "target_boundary_pixels": int(self.target[index]),
            }
            for index in range(self.num_classes)
        ]
        return {
            "tolerance_pixels": self.tolerance,
            "macro_boundary_f1_all": float(f1.mean()),
            "macro_boundary_f1_foreground": float(f1[1:].mean()),
            "per_class": per_class,
        }


def inner_class_boundary_bands(
    labels: torch.Tensor,
    num_classes: int,
    width_pixels: int,
) -> torch.Tensor:
    """Return class-wise inner boundary bands as defined by Boundary IoU.

    This implements ``mask - erode(mask)`` with zero-valued padding outside
    the image, matching the reference implementation from Cheng et al. (2021).
    """
    if width_pixels < 1:
        raise ValueError("Boundary width must be at least one pixel")
    masks = torch.nn.functional.one_hot(
        labels, num_classes=num_classes
    ).permute(0, 3, 1, 2).float()
    padded = torch.nn.functional.pad(
        masks,
        (width_pixels, width_pixels, width_pixels, width_pixels),
        mode="constant",
        value=0.0,
    )
    kernel = 2 * width_pixels + 1
    eroded = -torch.nn.functional.max_pool2d(
        -padded, kernel_size=kernel, stride=1
    )
    return masks.bool() & ~eroded.bool()


class BoundaryIoUMeter:
    """Dataset-level class Boundary IoU over inner boundary bands."""

    def __init__(
        self,
        num_classes: int,
        width_pixels: int | None = None,
        dilation_ratio: float | None = None,
    ) -> None:
        if (width_pixels is None) == (dilation_ratio is None):
            raise ValueError(
                "Specify exactly one of width_pixels or dilation_ratio"
            )
        self.num_classes = num_classes
        self.width_pixels = width_pixels
        self.dilation_ratio = dilation_ratio
        self.intersection = torch.zeros(
            num_classes, dtype=torch.float64
        )
        self.union = torch.zeros(num_classes, dtype=torch.float64)

    def _resolve_width(self, height: int, width: int) -> int:
        if self.width_pixels is not None:
            return int(self.width_pixels)
        resolved = int(
            round(float(self.dilation_ratio) * math.sqrt(height**2 + width**2))
        )
        return max(1, resolved)

    def update(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> None:
        height, width = predictions.shape[-2:]
        resolved_width = self._resolve_width(height, width)
        predicted_boundary = inner_class_boundary_bands(
            predictions, self.num_classes, resolved_width
        )
        target_boundary = inner_class_boundary_bands(
            targets, self.num_classes, resolved_width
        )
        reduce_dims = (0, 2, 3)
        self.intersection += (
            predicted_boundary & target_boundary
        ).sum(dim=reduce_dims).double().cpu()
        self.union += (
            predicted_boundary | target_boundary
        ).sum(dim=reduce_dims).double().cpu()

    def compute(self, image_height: int, image_width: int) -> dict:
        eps = 1e-12
        iou = self.intersection / (self.union + eps)
        per_class = [
            {
                "class_id": index,
                "class_name": CLASS_NAMES[index],
                "boundary_iou": float(iou[index]),
                "intersection_pixels": int(self.intersection[index]),
                "union_pixels": int(self.union[index]),
            }
            for index in range(self.num_classes)
        ]
        return {
            "width_pixels": self._resolve_width(
                image_height, image_width
            ),
            "dilation_ratio": self.dilation_ratio,
            "macro_boundary_iou_all": float(iou.mean()),
            "macro_boundary_iou_foreground": float(iou[1:].mean()),
            "per_class": per_class,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tolerance", type=int, default=2)
    parser.add_argument(
        "--boundary-iou-ratio",
        type=float,
        default=0.02,
        help="Canonical Boundary IoU band width as an image-diagonal ratio.",
    )
    parser.add_argument(
        "--boundary-iou-fixed-width",
        type=int,
        default=2,
        help="Additional fixed-width Boundary IoU for BF1 comparability.",
    )
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    checkpoint_path = (PROJECT_ROOT / args.checkpoint).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    set_reproducibility(
        args.seed, bool(config["training"].get("deterministic", True))
    )
    device = torch.device(config.get("device", "cuda"))
    _, eval_transform = create_transforms(config["data"]["image_size"])
    manifest = (PROJECT_ROOT / config["data"]["manifest"]).resolve()
    dataset = ManifestDataset(
        PROJECT_ROOT, manifest, "test", eval_transform
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=config.get("num_workers", 0) > 0,
    )

    model, pretraining = create_model(config, "ecsegformer", device)
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    confusion = ConfusionMatrixMeter(len(CLASS_NAMES))
    boundary = BoundaryF1Meter(len(CLASS_NAMES), args.tolerance)
    boundary_iou_standard = BoundaryIoUMeter(
        len(CLASS_NAMES), dilation_ratio=args.boundary_iou_ratio
    )
    boundary_iou_fixed = BoundaryIoUMeter(
        len(CLASS_NAMES), width_pixels=args.boundary_iou_fixed_width
    )
    evaluated_height = None
    evaluated_width = None
    with torch.inference_mode():
        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits, _, _ = unwrap_outputs(model(images))
            predictions = torch.argmax(logits, dim=1)
            evaluated_height, evaluated_width = predictions.shape[-2:]
            confusion.update(
                predictions.cpu().numpy(), masks.cpu().numpy()
            )
            boundary.update(predictions, masks)
            boundary_iou_standard.update(predictions, masks)
            boundary_iou_fixed.update(predictions, masks)

    if evaluated_height is None or evaluated_width is None:
        raise ValueError("No test images were evaluated")

    result = {
        "model": "ecsegformer",
        "seed": args.seed,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "num_images": len(dataset),
        "pretraining_load": pretraining,
        "segmentation": confusion.compute(),
        "boundary": boundary.compute(),
        "boundary_iou": {
            "definition": (
                "Class-wise inner boundary-band IoU "
                "(Cheng et al., CVPR 2021)"
            ),
            "standard": boundary_iou_standard.compute(
                evaluated_height, evaluated_width
            ),
            "fixed_width": boundary_iou_fixed.compute(
                evaluated_height, evaluated_width
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
