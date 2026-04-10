# TransResults

I use this folder for saved outputs from the transformer experiments.

How I laid it out
- `01_multilabel_distilbert` is the shared DistilBERT run with 3-fold cross-validation.
- `02_binary_toxic` through `07_binary_identity_hate` are one-label binary DistilBERT runs.
- `08_experiment_summary` through `11_test_results` are the rollup summaries, cost plots, and direct test comparisons.

I kept the small metrics files, configs, and plots here because they explain the transformer work quickly on GitHub. I leave checkpoints and row-level prediction dumps local because those are the biggest files by far.
