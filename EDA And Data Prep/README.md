# Data Exploration and Tabular Prep

This folder is the early data section of our larger toxic comment classification project. We use it to inspect the Jigsaw data, clean comment text, and export the tabular artifacts that later modeling workflows use elsewhere in the repo.

What stays here:
- raw-data loading checks and schema audit
- exploratory data analysis tables and figures
- cleaned text exports
- dense features, TF-IDF artifacts, and small diagnostics for the tabular branch

What does not live here anymore:
- transformer training or inference
- hybrid or embedding feature branches
- final project reporting
- downstream model comparison code

To rebuild this section locally:

```powershell
python -m pip install -r requirements.txt
python run_option_a.py
```

We expect the four Jigsaw CSV files in `data/`. Raw data and bulky generated artifacts stay local and are ignored for GitHub, while small summaries and EDA figures remain tracked so reviewers can see what this section produces.
