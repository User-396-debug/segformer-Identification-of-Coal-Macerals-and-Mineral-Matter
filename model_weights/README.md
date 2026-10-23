# Verified ECSegFormer Checkpoint

This directory contains one verified inference checkpoint from the controlled
main-comparison experiment:

- file: `ecsegformer_b1_seed2026_weights.pth`
- model: ECSegFormer-B1 with EEM, CAM, and the mineral auxiliary head
- run: seed 2026, fixed split `fixed_split_v1`
- input size: 512 x 512 pixels
- classes: background, vitrinite, mineral matter, liptinite, inertinite

The public file contains model tensors only. Optimizer state, training history,
run metadata, and evaluation results are not included. The exported state
dictionary was reloaded into the current repository model definition with
`strict=True` before release.

The corresponding resolved configuration is provided in
`ecsegformer_b1_seed2026_config.yaml`. The fixed-split manifest cannot recreate
the copyrighted image corpus by itself; source images and derived masks are not
redistributed in this repository.

Load the checkpoint as follows:

```python
import torch

checkpoint = torch.load(
    "model_weights/ecsegformer_b1_seed2026_weights.pth",
    map_location="cpu",
    weights_only=False,
)
state_dict = checkpoint["model_state_dict"]
```

Instantiate the model with the accompanying YAML file, then load `state_dict`
with `strict=True`.
