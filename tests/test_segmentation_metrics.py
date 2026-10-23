import unittest

import numpy as np

from utils.segmentation_metrics import ConfusionMatrixMeter


class ConfusionMatrixMeterTest(unittest.TestCase):
    def test_macro_metrics_exclude_background(self):
        meter = ConfusionMatrixMeter(num_classes=3)
        target = np.array([0, 0, 0, 0, 1, 1, 2, 2])
        prediction = np.array([0, 0, 0, 0, 1, 2, 2, 2])

        meter.update(prediction, target)
        metrics = meter.compute()

        # Class 1: IoU=1/2, Dice=2/3; class 2: IoU=2/3, Dice=4/5.
        self.assertAlmostEqual(metrics["mean_iou"], (0.5 + 2.0 / 3.0) / 2.0)
        self.assertAlmostEqual(metrics["mean_dice"], (2.0 / 3.0 + 0.8) / 2.0)

        # A background-inclusive mean would be larger because background is perfect.
        self.assertLess(metrics["mean_iou"], (1.0 + 0.5 + 2.0 / 3.0) / 3.0)
        self.assertLess(metrics["mean_dice"], (1.0 + 2.0 / 3.0 + 0.8) / 3.0)


if __name__ == "__main__":
    unittest.main()
