"""Canonical pixel-level metrics for ECSegFormer experiments."""
from __future__ import annotations

import numpy as np


class ConfusionMatrixMeter:
    """Accumulate one dataset-level confusion matrix; never average per-image IoU."""

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, prediction, target) -> None:
        prediction = np.asarray(prediction, dtype=np.int64).ravel()
        target = np.asarray(target, dtype=np.int64).ravel()
        valid = ((target >= 0) & (target < self.num_classes) &
                 (prediction >= 0) & (prediction < self.num_classes))
        indices = self.num_classes * target[valid] + prediction[valid]
        self.matrix += np.bincount(indices, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self) -> dict:
        cm = self.matrix.astype(np.float64)
        tp = np.diag(cm)
        true_count = cm.sum(axis=1)
        pred_count = cm.sum(axis=0)
        fp = pred_count - tp
        fn = true_count - tp
        union = tp + fp + fn
        support = true_count.astype(np.int64)
        present = support > 0

        def ratio(numerator, denominator):
            out = np.full(self.num_classes, np.nan, dtype=np.float64)
            np.divide(numerator, denominator, out=out, where=denominator > 0)
            return out

        iou = ratio(tp, union)
        dice = ratio(2 * tp, 2 * tp + fp + fn)
        precision = ratio(tp, tp + fp)
        recall = ratio(tp, tp + fn)
        per_class = []
        for i in range(self.num_classes):
            per_class.append({
                "class_id": i, "support_pixels": int(support[i]),
                "iou": None if np.isnan(iou[i]) else float(iou[i]),
                "dice": None if np.isnan(dice[i]) else float(dice[i]),
                "precision": None if np.isnan(precision[i]) else float(precision[i]),
                "recall": None if np.isnan(recall[i]) else float(recall[i]),
            })
        total = cm.sum()
        return {
            "pixel_accuracy": float(tp.sum() / total) if total else None,
            "mean_iou": float(np.nanmean(iou[present])) if present.any() else None,
            "mean_dice": float(np.nanmean(dice[present])) if present.any() else None,
            "confusion_matrix": self.matrix.tolist(),
            "per_class": per_class,
        }
