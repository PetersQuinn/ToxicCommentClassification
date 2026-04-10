# Engineered Features

This folder is the engineered-features part of the larger toxic comment classification project built on the Jigsaw dataset. In the full project, we compare different modeling approaches and look at performance and fairness across subgroups. This folder is only the engineered-feature workflow, not the whole project by itself.

We use the six notebooks here for the six binary label tasks: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity`. They work from dense tabular text features rather than transformer representations.

The `data/` folder contains the train and test parquet files those notebooks use. These tables include engineered text features like counts, ratios, lexical statistics, punctuation signals, and profanity or lexicon-based features. Other parts of the larger repo cover the other modeling families and the broader fairness analysis.
