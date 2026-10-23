# Original ECSegFormer full-training path

This path reproduces the original full ECSegFormer training configuration.
It uses 300 epochs and a batch size of 8. It is separate from the controlled
50-epoch, batch-size-4 repeated-run comparison in `experiments/`.

Use the following files for this path:

- `config/config_ecsegformer.yaml`
- `train_enhancement_modules.py`
- `evaluate_ecsegformer.py`
- `models/segformer_model.py`
- `models/segformer_enhancements.py`
- `utils/improved_data_loader.py`
- `utils/class_balanced_sampler.py`
- `utils/advanced_loss_functions.py`
- `utils/segmentation_metrics.py`

Train from the repository root:

```powershell
python train_enhancement_modules.py --config config/config_ecsegformer.yaml --checkpoint checkpoints/segformer/best_model.pth
```

Evaluate only the held-out test split:

```powershell
python evaluate_ecsegformer.py --config config/config_ecsegformer.yaml --checkpoint checkpoints/segformer_enhanced/best_enhanced_model.pth
```

`evaluation_results/ecsegformer_test_metrics.json` is the only official metric artifact.  It is calculated from one confusion matrix accumulated over all test pixels; it must replace historical per-image-average result JSON files in figures and tables.

The remaining scripts and model variants are historical experiments or baseline
implementations. They are not part of this original full-training path and must
not write to the official result artifact.
