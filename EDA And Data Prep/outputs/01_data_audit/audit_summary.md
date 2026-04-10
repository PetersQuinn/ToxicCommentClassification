# Audit Summary

- We expect `train.csv`, `test.csv`, `test_labels.csv`, and `sample_submission.csv` in `data/`.
- The train table matches the six-label Jigsaw format used by the rest of our project.
- `test_labels.csv` mixes usable rows with `-1` placeholders, so we keep it as a partial labeled reference instead of a standard split.
- Comment lengths are long-tailed, which matters for both EDA and later tabular feature design.
