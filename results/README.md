# Public run-level metrics

These CSV files contain the final run-level numerical results used in the
revised manuscript. They do not contain source images, reference masks,
predictions, or private filesystem paths.

## Metric convention

- Foreground mIoU and mDice are unweighted macro averages over
  vitrinite, minerals, liptinite, and inertinite; background is excluded.
- `boundary_f1_2px` is the foreground macro Boundary F1 score evaluated with a
  two-pixel tolerance.
- The foreground class names in `main_comparison_class_metrics.csv` are
  vitrinite, inertinite, liptinite, and minerals.
- The main and additional-model CSVs use percentages because they are the
  manuscript-aligned values transcribed from Supplementary Tables S1 and S2.
  The aligned reruns and sensitivity CSVs retain the original proportions in
  the range 0 to 1. Units are stated explicitly in the column names.
- Values transcribed from Supplementary Tables S1 and S2 are displayed to two
  decimal places. Manuscript summary means and sample standard deviations were
  calculated from the retained unrounded metrics, so recomputation from the
  displayed values can differ by 0.01 percentage point because of rounding.

## Files

- `main_comparison_per_seed.csv`: ten matched seeds for ECSegFormer and its
  SegFormer-B1 backbone.
- `main_comparison_class_metrics.csv`: per-class results for the same ten
  matched runs.
- `additional_models_per_seed.csv`: five-seed results for ECSegFormer, U-Net,
  DeepLabv3+, PSPNet, and SegNeXt-T.
- `aligned_ablation_per_seed.csv`: five-seed EEM/CAM ablation under the aligned
  50-epoch protocol.
- `sensitivity_per_seed.csv`: five-seed parameter-sensitivity results.
- `reviewer_controls_per_seed.csv`: five-seed results for the aligned full
  model, the SegFormer-B2 higher-capacity control, and CAM without RGB input.

The `series` column identifies the experimental series. Results from different
series should not be treated as paired runs unless the manuscript explicitly
defines the corresponding matched-seed comparison.

The public CSV files are released as the numerical records corresponding to the
reported tables. Source images, masks, and prediction arrays are not included.
