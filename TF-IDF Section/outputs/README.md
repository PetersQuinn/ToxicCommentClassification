# Outputs

This folder holds generated artifacts from the TF-IDF pipeline.

For GitHub, we keep this folder small on purpose. We leave only a few lightweight reference files visible and keep the heavier run outputs local:

- logs
- plots
- vectorizer dumps
- model artifacts
- scored predictions
- bulk pretuning tables

The main kept summary is `final_identity_hate_prediction_exports_summary.csv`, which shows the final export comparison for the retained model families.

We also keep `feature_build/tfidf_build_metadata.json` so the chosen text field and reference TF-IDF profile stay easy to inspect.
