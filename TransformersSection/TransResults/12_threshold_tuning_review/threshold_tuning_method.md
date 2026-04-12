# Threshold Tuning Method

## Inputs
- Validation probability CSVs were treated as the only admissible source for threshold selection.
- Labeled test probability CSVs were used only for final thresholded evaluation.
- Unlabeled `test_predictions.csv` files were updated only by applying already selected validation thresholds.

## Rule used
- Threshold grid: `np.linspace(0.05, 0.95, 91)`
- Objective: `maximize_validation_f1_tie_break_to_0_5`
- Tie-break: closest threshold to `0.5`

## Why this matches TF-IDF closely enough
- The TF-IDF helper in `TF-IDF Section/utils/metrics_utils.py` uses the same probability grid and the same label-level F1 objective with a tie-break toward the default threshold.
- That means the transformer and TF-IDF pipelines are now aligned at the post-training threshold-selection stage, while AP remains the threshold-free model-selection metric.

## Explicit non-changes
- No retraining
- No checkpoint edits
- No AP redefinition
- No deletion of the original fixed-`0.5` artifacts
