# 01_multilabel_distilbert

I use this folder for the shared DistilBERT model that predicts all six toxicity labels together.

What I would read first
- `test_labeled_metrics.json` for the main held-out test metrics.
- `cv_validation_metrics_oof.json` for the out-of-fold validation summary across the three CV folds.
- `cv_fold_summary.csv` for the per-fold rollup.
- `test_average_precision_by_label.png` and `cv_validation_map_by_fold.png` for the quickest visual checks.

The `fold_1` through `fold_3` subfolders keep the per-fold curves and metrics. I leave the checkpoint files and row-level prediction exports out of Git because they are huge.
