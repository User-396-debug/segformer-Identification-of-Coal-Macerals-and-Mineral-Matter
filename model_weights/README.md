# Verified Checkpoints

This directory contains verified state-only inference checkpoints from the
controlled main-comparison experiment.

## ECSegFormer-B1

- file: `ecsegformer_b1_seed2026_weights.pth`
- model: ECSegFormer-B1 with EEM, CAM, and the minerals auxiliary head
- run: seed 2026, fixed split `fixed_split_v1`
- input size: 512 x 512 pixels
- classes: background, vitrinite, minerals, liptinite, inertinite

## SegFormer-B1 baseline

- file: `segformer_b1_seed2026_weights.pth`
- model: SegFormer-B1 without EEM, CAM, or an auxiliary head
- run: seed 2026, fixed split `fixed_split_v1`
- training protocol: 50 epochs, batch size 4
- input size: 512 x 512 pixels
- classes: background, vitrinite, minerals, liptinite, inertinite
- foreground-only test result: 70.09% mIoU and 81.52% mDice

The public files contain model tensors only. Optimizer states, training
histories, and private filesystem paths are not included. Each exported state
dictionary was reloaded into its current repository model definition with
`strict=True` before release.

The corresponding release configurations are provided as
`ecsegformer_b1_seed2026_config.yaml` and
`segformer_b1_seed2026_config.yaml`. The fixed-split manifest cannot recreate
the copyrighted image corpus by itself; source images and derived masks are
not redistributed in this repository.

Load either checkpoint as follows:

```python
import torch

checkpoint = torch.load(
    "model_weights/<checkpoint_name>.pth",
    map_location="cpu",
    weights_only=False,
)
state_dict = checkpoint["model_state_dict"]
```

Instantiate the corresponding model with the accompanying YAML file, then load
`state_dict` with `strict=True`.
