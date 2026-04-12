# Transformer Threshold Tuning Fix
## Why this change was necessary
Transformer thresholded metrics in the tracked repo were computed at a fixed `0.5`, while the TF-IDF pipeline already tuned thresholds after model selection. This postprocessing fix makes the transformer decision-threshold stage scientifically closer to the TF-IDF workflow without retraining any model.

## Inputs used
- Saved validation probability CSVs from `01_multilabel_distilbert` and binary folders `03` through `07`.
- Saved test probability CSVs from `01_multilabel_distilbert` and binary folders `02` through `07`.
- Existing fixed-`0.5` metric JSONs in the same model folders for before-vs-after comparison.
- Existing summary artifacts in `08_experiment_summary`, `10_overall_summary`, and `11_test_results`.
- TF-IDF threshold logic from `TF-IDF Section/utils/metrics_utils.py`.

## Method
- Thresholds were tuned only from saved validation probabilities.
- The tuning rule mirrors the TF-IDF helper: search `0.05` to `0.95` in steps of `0.01`, maximize validation F1 label by label, and break ties toward `0.5`.
- The selected thresholds were then applied to the saved labeled-test and unlabeled-test probability CSVs.
- Average precision was recomputed from the same saved probabilities and checked against the fixed-`0.5` artifacts to confirm it stayed unchanged.
- No training notebook, checkpoint, or model weight file was edited.

## Files created
- `TransformersSection\postprocess_transformer_thresholds.py`
- `TransformersSection\TransResults\01_multilabel_distilbert\selected_thresholds_threshold_tuned.json`
- `TransformersSection\TransResults\01_multilabel_distilbert\test_labeled_metrics_threshold_tuned.json`
- `TransformersSection\TransResults\01_multilabel_distilbert\test_labeled_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\01_multilabel_distilbert\test_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\01_multilabel_distilbert\validation_metrics_threshold_tuned.json`
- `TransformersSection\TransResults\01_multilabel_distilbert\validation_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\01_multilabel_distilbert\validation_threshold_search_threshold_tuned.csv`
- `TransformersSection\TransResults\02_binary_toxic\selected_thresholds_threshold_tuned.json`
- `TransformersSection\TransResults\02_binary_toxic\test_labeled_metrics_threshold_tuned.json`
- `TransformersSection\TransResults\02_binary_toxic\test_labeled_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\02_binary_toxic\test_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\03_binary_severe_toxic\selected_thresholds_threshold_tuned.json`
- `TransformersSection\TransResults\03_binary_severe_toxic\test_labeled_metrics_threshold_tuned.json`
- `TransformersSection\TransResults\03_binary_severe_toxic\test_labeled_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\03_binary_severe_toxic\test_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\03_binary_severe_toxic\validation_metrics_threshold_tuned.json`
- `TransformersSection\TransResults\03_binary_severe_toxic\validation_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\03_binary_severe_toxic\validation_threshold_search_threshold_tuned.csv`
- `TransformersSection\TransResults\04_binary_obscene\selected_thresholds_threshold_tuned.json`
- `TransformersSection\TransResults\04_binary_obscene\test_labeled_metrics_threshold_tuned.json`
- `TransformersSection\TransResults\04_binary_obscene\test_labeled_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\04_binary_obscene\test_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\04_binary_obscene\validation_metrics_threshold_tuned.json`
- `TransformersSection\TransResults\04_binary_obscene\validation_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\04_binary_obscene\validation_threshold_search_threshold_tuned.csv`
- `TransformersSection\TransResults\05_binary_threat\selected_thresholds_threshold_tuned.json`
- `TransformersSection\TransResults\05_binary_threat\test_labeled_metrics_threshold_tuned.json`
- `TransformersSection\TransResults\05_binary_threat\test_labeled_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\05_binary_threat\test_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\05_binary_threat\validation_metrics_threshold_tuned.json`
- `TransformersSection\TransResults\05_binary_threat\validation_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\05_binary_threat\validation_threshold_search_threshold_tuned.csv`
- `TransformersSection\TransResults\06_binary_insult\selected_thresholds_threshold_tuned.json`
- `TransformersSection\TransResults\06_binary_insult\test_labeled_metrics_threshold_tuned.json`
- `TransformersSection\TransResults\06_binary_insult\test_labeled_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\06_binary_insult\validation_metrics_threshold_tuned.json`
- `TransformersSection\TransResults\06_binary_insult\validation_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\06_binary_insult\validation_threshold_search_threshold_tuned.csv`
- `TransformersSection\TransResults\07_binary_identity_hate\selected_thresholds_threshold_tuned.json`
- `TransformersSection\TransResults\07_binary_identity_hate\test_labeled_metrics_threshold_tuned.json`
- `TransformersSection\TransResults\07_binary_identity_hate\test_labeled_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\07_binary_identity_hate\test_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\07_binary_identity_hate\validation_metrics_threshold_tuned.json`
- `TransformersSection\TransResults\07_binary_identity_hate\validation_predictions_threshold_tuned.csv`
- `TransformersSection\TransResults\07_binary_identity_hate\validation_threshold_search_threshold_tuned.csv`
- `TransformersSection\TransResults\08_experiment_summary\model_comparison_summary_threshold_tuned.csv`
- `TransformersSection\TransResults\08_experiment_summary\model_comparison_summary_threshold_tuned.json`
- `TransformersSection\TransResults\10_overall_summary\overall_handoff_summary_threshold_tuned.json`
- `TransformersSection\TransResults\11_test_results\binary_test_metrics_by_label_threshold_tuned.csv`
- `TransformersSection\TransResults\11_test_results\binary_test_summary_threshold_tuned.json`
- `TransformersSection\TransResults\11_test_results\f1_by_label_multilabel_vs_binary_threshold_tuned.png`
- `TransformersSection\TransResults\11_test_results\multilabel_test_metrics_by_label_threshold_tuned.csv`
- `TransformersSection\TransResults\11_test_results\multilabel_test_summary_threshold_tuned.json`
- `TransformersSection\TransResults\11_test_results\multilabel_vs_binary_test_comparison_threshold_tuned.csv`
- `TransformersSection\TransResults\11_test_results\overall_test_results_summary_threshold_tuned.json`
- `TransformersSection\TransResults\12_threshold_tuning_review\before_vs_after_metrics.csv`
- `TransformersSection\TransResults\12_threshold_tuning_review\before_vs_after_summary.md`
- `TransformersSection\TransResults\12_threshold_tuning_review\README.md`
- `TransformersSection\TransResults\12_threshold_tuning_review\threshold_tuning_method.md`
- `TransformersSection\TransResults\12_threshold_tuning_review\updated_artifacts_inventory.md`
- `TransformersSection\TransResults\12_threshold_tuning_review\what_changed.md`

## Files modified
- `text_data_results_summary.md`
- `text_data_results_quick_hits.md`

## Metrics unchanged by design
- Average precision for every transformer model and label, because AP is threshold-free and was recomputed from the original saved probabilities.
- Training time, throughput, parameter count, GPU memory, and any other training-side cost artifact.
- The fixed-`0.5` legacy JSON/CSV artifacts, which were preserved beside the new threshold-tuned versions.

## Metrics changed after threshold tuning
- Precision, recall, F1, and predicted positive rate now use validation-selected operating thresholds in the new `*_threshold_tuned.*` outputs.
- The shared multilabel model now has one tuned threshold per label instead of one global `0.5`.
- Five binary models were tuned the same way; `binary_toxic` stayed at `0.5` because `validation_predictions.csv` was missing from the tracked folder.

## Downstream artifacts updated
- `08_experiment_summary/model_comparison_summary_threshold_tuned.csv`
- `08_experiment_summary/model_comparison_summary_threshold_tuned.json`
- `10_overall_summary/overall_handoff_summary_threshold_tuned.json`
- `11_test_results/multilabel_test_metrics_by_label_threshold_tuned.csv`
- `11_test_results/binary_test_metrics_by_label_threshold_tuned.csv`
- `11_test_results/multilabel_test_summary_threshold_tuned.json`
- `11_test_results/binary_test_summary_threshold_tuned.json`
- `11_test_results/multilabel_vs_binary_test_comparison_threshold_tuned.csv`
- `11_test_results/overall_test_results_summary_threshold_tuned.json`
- `11_test_results/f1_by_label_multilabel_vs_binary_threshold_tuned.png`

## Status of test_labeled_predictions.csv
- The original `test_labeled_predictions.csv` files were left untouched to avoid silently overwriting legacy fixed-`0.5` outputs.
- New companion exports named `test_labeled_predictions_threshold_tuned.csv` were written in each model folder.
- These companion exports contain the original probabilities plus `*_selected_threshold` and `*_pred_threshold_tuned` columns.

## TF-IDF tuning sanity check
- The TF-IDF helper in `TF-IDF Section/utils/metrics_utils.py` tunes thresholds label by label on validation data, optimizes F1, and breaks ties toward the default threshold.
- The transformer postprocessor now follows that same validation-only F1-maximizing rule on probability grids.
- No obvious TF-IDF threshold bug was changed here; the goal was methodological alignment, not a TF-IDF rewrite.

## Remaining caveats
- `TransformersSection/TransResults/02_binary_toxic/validation_predictions.csv` was not present, so `binary_toxic` could not be re-tuned from validation probabilities and remains at `0.5` in the threshold-tuned summaries.
- The training/evaluation notebook still contains fixed-`0.5` logic. This fix intentionally avoids refactoring that notebook and instead writes postprocessed versioned outputs.
- Any external consumer that hard-codes the old fixed-`0.5` filenames will need to switch to the new `*_threshold_tuned.*` outputs to use the comparable operating thresholds.
