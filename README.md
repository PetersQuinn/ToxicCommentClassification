# Toxic Comment Classification

This repository compares multiple modeling approaches for detecting toxic comments in the Jigsaw toxic comment dataset. We evaluate different strategies—classical ML, transformer-based, and hybrid feature engineering—alongside fairness analysis across demographic subgroups.

## Structure

The project is organized into four main sections, each handling a different modeling workflow:

### [EDA And Data Prep](EDA%20And%20Data%20Prep)

Early-stage exploratory analysis and data preparation. We validate the raw Jigsaw CSV files, clean comment text, generate diagnostic summaries, and produce the cleaned parquet inputs that downstream workflows depend on. This section also experiments with dense and sparse feature engineering.

- Inputs: Raw CSV files from Jigsaw
- Outputs: Cleaned parquets, EDA tables, diagnostic figures, TF-IDF and tabular feature artifacts

### [TF-IDF Section](TF-IDF%20Section)

Classical ML baseline using sparse TF-IDF features. We train and tune three model families—logistic regression, linear SVM, and complement Naïve Bayes—on both multilabel and label-specific binary tasks. Results are evaluated using macro PR-AUC and threshold-tuned metrics like F1.

- Inputs: Cleaned parquets from EDA prep
- Model families: Logistic regression, linear SVM, complement NB
- Outputs: Tuning results, final model metrics, lightweight feature metadata

### [TransformersSection](TransformersSection)

Transformer-based modeling using DistilBERT for multilabel and binary classification. Heavy training was performed on GPU (Colab). This section retains lightweight evaluation summaries, comparison tables, and aggregate results without the full checkpoint tree.

- Inputs: Cleaned parquets from EDA prep
- Model: DistilBERT fine-tuning
- Outputs: Per-task evaluation metrics, cost analysis, summary comparisons

### [Engineered Features Section](Engineered%20Features%20Section)

Dense tabular feature engineering approach. Six notebooks build binary classifiers for each toxic label using hand-crafted text features (counts, ratios, lexical statistics, profanity signals). These notebooks provide an alternative to sparse and transformer-based approaches.

- Inputs: Engineered feature parquets from EDA prep
- Approach: Six binary classifiers on dense features
- Outputs: Per-notebook evaluation and fairness analysis

## Running the Project

Each section is self-contained and has its own README with detailed setup and execution instructions. Review the individual section READMEs for:

- Dependencies and environment setup
- Data input paths and requirements
- Step-by-step execution order
- Configuration overrides and optional flags

## What's Tracked vs. Local

To keep the repository lightweight and reviewable, we **ignore**:

- Raw data files (CSVs, parquets)
- Model checkpoints and binary artifacts
- Training logs and temporary exports
- Python cache and environment files
- OS and editor clutter

We **keep**:

- README files and documentation
- Lightweight metadata and summary files (JSON, CSV)
- Representative evaluation plots and tables
- Configuration and feature metadata

This balance preserves visibility into methodology and results while avoiding bloat.

## Key Files

- [EDA And Data Prep/README.md](EDA%20And%20Data%20Prep/README.md) – Details on data cleaning and preparation
- [TF-IDF Section/README.md](TF-IDF%20Section/README.md) – TF-IDF baseline setup and results
- [TransformersSection/README.md](TransformersSection/README.md) – Transformer model details
- [Engineered Features Section/README.md](Engineered%20Features%20Section/README.md) – Dense feature engineering approach
