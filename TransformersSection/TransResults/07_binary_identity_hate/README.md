# 07_binary_identity_hate

I use this folder for the one-label DistilBERT run on `identity_hate`.

What I would read first
- `test_labeled_metrics.json` for the held-out test metrics.
- `validation_average_precision_curve.png` for the checkpoint-selection curve.
- `training_loss_curve.png` for a quick sanity check on training behavior.
- `run_stats.json` for timing, throughput, parameter count, and GPU memory.

I keep the lightweight summaries here so I can compare this run against the shared multilabel model and the other binary runs in the bigger project.
