# GPTFIDF

This folder is the TF-IDF section of our broader toxic comment classification project on the Jigsaw dataset. We use it for the sparse lexical baseline and comparison work, while other parts of the repo cover dense features, transformer models, hybrids, and the project-level fairness analysis.

## What We Keep In Scope

This section now focuses on the three model families we actually kept and tuned:

- `logistic_regression`
- `linear_svm`
- `complement_nb`

Each family supports a shared `multilabel` path plus six label-specific `binary` runs. We still tune around `macro_pr_auc`, then set thresholds afterward for metrics like F1, precision, and recall.

## Pipeline

The workflow is a simple step sequence:

1. `step_0_initializations.py`
   Validates the local parquet inputs and writes run metadata.
2. `step_1_data_audit_and_validation.py`
   Checks shapes, schema alignment, label prevalence, nulls, duplicates, and text length.
3. `step_2_build_tfidf_features.py`
   Compares a small TF-IDF candidate set and saves the reference vectorizer metadata.
4. `step_3_train_baseline_models.py`
   Produces untuned holdout baselines for the retained model families.
5. `step_4_cross_validation_and_error_analysis.py`
   Runs compact baseline CV and pretuning diagnostics.
6. `step_5_model_tuning.py`
   Tunes the retained model families.
7. `step_6_final_model_evaluation.py`
   Fits the tuned models on the full train split and exports the final evaluation artifacts.

## Data

We expect cleaned parquet inputs under `data/parquets/` or `~/data/parquets/`:

- `train_cleaned.parquet`
- `test_cleaned.parquet`
- `test_labeled_cleaned.parquet`

For TF-IDF text input, we use this priority:

1. `comment_text_tfidf`
2. `comment_text_clean`
3. `comment_text`

The data files stay local and are intentionally ignored in this section.

## Running It

We have been using the existing Anaconda environment on this machine:

```powershell
C:\Users\quint\anaconda3\python.exe step_0_initializations.py
C:\Users\quint\anaconda3\python.exe step_1_data_audit_and_validation.py
C:\Users\quint\anaconda3\python.exe step_2_build_tfidf_features.py
C:\Users\quint\anaconda3\python.exe step_3_train_baseline_models.py
C:\Users\quint\anaconda3\python.exe step_4_cross_validation_and_error_analysis.py --cleanup-large-pretuning
C:\Users\quint\anaconda3\python.exe step_5_model_tuning.py
C:\Users\quint\anaconda3\python.exe step_6_final_model_evaluation.py
```

Useful overrides:

- `--models logistic_regression linear_svm`
- `--train-fraction 0.5`
- `--save-full-scored`
- `--overwrite`
- `--identity-hate-export-only`

## Outputs

We keep the GitHub-facing version of this folder intentionally lean. Most generated artifacts stay local:

- parquet inputs
- logs
- plots
- saved models
- prediction dumps
- bulk pretuning files
- generated markdown summaries

The small files we still keep visible are there to show the retained TF-IDF setup and the final `identity_hate` export comparison without pushing the whole artifact tree.

- `outputs/feature_build/tfidf_build_metadata.json`
- `outputs/final_identity_hate_prediction_exports_summary.csv`
