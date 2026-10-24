# ECSegFormer Coal Petrography Segmentation

This repository contains the research code for semantic segmentation of coal
petrographic images. The implementation extends a SegFormer baseline with an
Edge Enhancement Module (EEM), a Color Attention Module (CAM), and a composite
segmentation objective for five classes:

1. background
2. vitrinite
3. mineral matter
4. liptinite
5. inertinite

The repository is intended for code review and reproducibility. The source
photomicrographs, derived masks, and manuscript files are not included because
their redistribution requires separate permission. One verified inference
checkpoint from the controlled main-comparison experiment is available in
`model_weights/`.

## Repository scope

Tracked files should normally be limited to:

- `models/`: model definitions, including EEM and CAM
- `utils/`: data loading, augmentation, losses, metrics, and utilities
- `config/`: training and experiment configurations
- `experiments/`: training, evaluation, split, analysis, and verification scripts
- `scripts/`: repository and document utilities
- top-level Python entry points and dependency files

The local `.gitignore` excludes data, image files, checkpoints, pretrained
weights, experiment outputs, logs, manuscript documents, and temporary files.

## Environment

The experiments were developed with PyTorch and CUDA on an NVIDIA GPU. The
exact environment should be recorded before public release. The current
`requirements.txt` is a local `pip freeze` export and contains machine-specific
package URLs; it is retained as provenance but is not a portable installation
file. The older `requirements_autodl_fixed.txt` targets a legacy PyTorch/CUDA
stack and should only be used when reproducing that historical environment.
A CUDA-neutral starting-point dependency list is provided in
`requirements-public.txt`; validate its versions together with the selected
PyTorch release before publication.

Typical setup:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
# Install a CUDA-compatible PyTorch/torchvision pair for the target machine.
python -m pip install -r requirements-public.txt
# Do not install the ignored historical requirements exports blindly.
```

## Data layout

The training code expects a local dataset with paired images and masks:

```text
data/
  train/images/
  train/masks/
  val/images/
  val/masks/
  test/images/
  test/masks/
```

The exact image-group partition used for the reported experiments is provided
in `experiments/manifests/fixed_split_v1.csv`. It contains 725 original training
images, 148 validation images, and 145 reserved test images. An additional 250
offline-augmented derivatives occur only in the training split, giving 975,
148, and 145 effective entries, respectively. Offline augmented files use names
containing `_aug_`; their source group is kept in the same split as the original
image. The manifest contains identifiers, relative paths, split assignments,
class-presence signatures, and class pixel counts, but does not contain image or
mask data. Do not upload the source images or masks to this repository without
confirming the applicable copyright and data-use permissions.

## Training protocols

### Original ECSegFormer training

The original full ECSegFormer training path uses
`config/config_ecsegformer.yaml` or the equivalent
`config/config_enhancement_training.yaml`. These configurations use 300 epochs
and a batch size of 8. This path is separate from the controlled repeated-run
comparison below, which uses 50 epochs and a batch size of 4 for every compared
model.

```bash
python train_enhancement_modules.py \
  --config config/config_ecsegformer.yaml \
  --checkpoint checkpoints/segformer/best_model.pth
```

The data and checkpoint paths are local inputs and are not included in this
repository.

### Controlled main comparison

The main comparison configuration is:

```text
config/experiments/main_comparison.yaml
```

It specifies the fixed split, 512 x 512 input images, a pretrained MiT-B1
encoder, the ECSegFormer modules, the 50-epoch repeated-run protocol, and the
loss settings used by the corresponding experiment.

The primary training and analysis entry points are:

```bash
python experiments/train_main_comparison.py --model ecsegformer --config config/experiments/main_comparison.yaml --seed 2026 --run-dir outputs/main_comparison/ecsegformer/seed_2026
python experiments/analyze_main_comparison.py
python experiments/verify_fixed_split.py
```

Check each script's `--help` output before running because output paths and
optional arguments may differ between experiment families.

For the static complexity comparison at 512 x 512, after installing `fvcore`
and placing any optional checkpoint locally, run:

```bash
python scripts/remote_compute_flops.py
```

The script reports trainable parameters and single-forward-pass FLOPs for the
SegFormer baseline and ECSegFormer. Paths can be overridden with `ECS_CONFIG`,
`ECS_CHECKPOINT`, and `ECS_FLOPS_OUTPUT`.

## Experiment families

The `config/experiments/` directory contains configurations for:

- the primary SegFormer versus ECSegFormer comparison
- EEM/CAM ablation experiments
- dilation, CAM-strength, and loss-weight sensitivity analyses
- additional baseline comparisons
- split-robustness and diagnostic experiments

The controlled protocols are intentionally distinct: the formal repeated-run
comparison and the parameter and loss sensitivity configurations use 50 epochs
and batch size 4. These settings must not be confused with the separate original
full ECSegFormer path above, which uses 300 epochs and batch size 8.

Results from different experiment families should not be pooled unless their
data manifest, training budget, optimizer, scheduler, loss settings, and seed
protocol are identical.

## Reproducibility notes

- The split manifest must be frozen before training.
- The validation subset is used for checkpoint selection; the reserved test
  subset must not be used for early stopping or model selection.
- Random seeds and the resolved configuration should be saved with every run.
- The canonical evaluator reports mIoU and mDice as macro-averages over the
  present foreground classes (class IDs 1-4); background (class ID 0) is
  excluded from both means. Per-class metrics and pixel accuracy remain
  available separately.
- Static FLOPs are not hardware-dependent inference latency. Latency,
  throughput, and memory should be measured separately if deployment claims are
  made.

## Data and model availability

The source image archive and derived annotations are not redistributed here.
A verified ECSegFormer-B1 checkpoint for seed 2026 is provided in
`model_weights/` together with its resolved configuration. It corresponds to
the controlled 50-epoch, batch-size-4 main-comparison protocol and should not
be confused with the separate original 300-epoch training path.

## License

No open-source license is declared yet. Add a license after the authors confirm
the ownership and redistribution conditions for the code and any bundled
materials.
