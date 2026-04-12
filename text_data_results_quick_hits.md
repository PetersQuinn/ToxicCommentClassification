# Text Data Results Quick Hits

| Approach | Task | Test AP | Test F1 | Threshold | Compute note |
| --- | --- | --- | --- | --- | --- |
| TF-IDF Logistic Regression | Multilabel | `0.576` | `0.541` | Tuned per label (`0.67` to `0.83`) | About `49s` CPU fit + eval |
| TF-IDF Linear SVM | Multilabel | `0.563` | `0.516` | Tuned per label (margin thresholds `0.0` to `0.177`) | About `53s` CPU fit + eval |
| DistilBERT Shared Multilabel | Multilabel | `0.681` | `0.611` | Tuned per label (`0.25` to `0.50`) | About `27.4` min GPU training |
| DistilBERT Binary Suite | Six one-label models | `0.644` | `0.591` | Mostly tuned (`0.30` to `0.50`), but toxic stayed at `0.50` | About `17.5` min GPU total across 6 models |

- The shared DistilBERT multilabel model is the best overall text approach in the repo.
- TF-IDF logistic regression is the strongest classical baseline and the best low-cost / interpretable option.
- TF-IDF linear SVM stayed close to logistic regression, but finished lower on final-test AP.
- Complement NB is very fast, but it is not competitive enough to recommend as the main text model.
- The shared transformer beats the binary transformer on AP for every label, even after threshold tuning.
- After tuning, the shared transformer also beats the binary transformer on tuned F1 for severe_toxic, obscene, threat, and insult.
- The binary `identity_hate` transformer is still worth mentioning because it keeps a tiny tuned-F1 edge (`0.6344` vs `0.6320`), even though AP falls.
- TF-IDF and transformers are now mostly comparable at the threshold stage because both use post-training validation-selected thresholds.
- The transformer quality gain over the best TF-IDF baseline is large enough to matter for the final recommendation.
- Rare-label transformer thresholds moved substantially lower: shared severe_toxic shifted to `0.25` and shared identity_hate to `0.31`.

- Open question: the repo does not track full TF-IDF final-test metric bundles for every label, only logs plus the `identity_hate` export summary.
- Open question: final TF-IDF test precision/recall/ROC-AUC tables were not found and may need regeneration from `step_6_final_model_evaluation.py`.
- Open question: `TransformersSection/TransResults/02_binary_toxic/validation_predictions.csv` is missing, so `binary_toxic` could not be threshold-tuned beyond the legacy `0.5` operating point.
- Open question: any downstream fairness workflow must be pointed at the new `*_threshold_tuned.csv` exports rather than the legacy fixed-`0.5` files.
- Open question: `TF-IDF Section/outputs/final_identity_hate_prediction_exports_summary.md` is referenced in logs but missing on disk.
