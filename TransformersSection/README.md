# Transformer Section

I use this folder for the transformer part of a larger fairness-aware toxic comment classification project built around the Jigsaw toxic comment data.

What is here
- `TransformerFinishedAP.ipynb` is the main notebook for the DistilBERT runs, evaluation, and rollup summaries.
- `data/` holds the local parquet inputs I staged for this workflow.
- `TransResults/` holds the saved metrics, plots, and comparison summaries.

Context that matters
- This is only the transformer section. The full repo also includes the classical ML, engineered-feature, and broader fairness analysis sections.
- The heavy transformer training and fine-tuning were done in Google Colab or another GPU-backed setup. I do not treat full local CPU retraining here as the normal path.
- I keep the lightweight summaries and plots tracked. I leave checkpoints, caches, and row-level prediction exports local because they are the files that make this section heavy.

If I only wanted the quick read, I would start in `TransResults/10_overall_summary/`, then `TransResults/11_test_results/`, then `TransResults/08_experiment_summary/`.
