# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A small tweet hate-speech text-classification project. `Basic ML Model for Text Classification
2.ipynb` is the tutorial walkthrough (a step-by-step notebook). The same logic is reimplemented
as a tested Python package, `text_classification/`, so it can run and be extended outside the
notebook.

## Commands

```bash
pip install -r requirements.txt

python main.py             # train + evaluate both models
python main.py --basic     # only the hand-engineered-feature GaussianNB model
python main.py --tfidf     # only the TF-IDF + logistic regression model

python -m pytest           # run the test suite
python -m pytest tests/test_features.py::test_add_features_adds_expected_columns  # single test
```

## Architecture

Pipeline: `data.py` (load CSV + clean) → `features.py` (hand-engineered features, basic model
only) → `model.py` (train/evaluate) → `main.py` (CLI wiring).

- `text_classification/cleaning.py` — `clean_text()`: strips everything but letters/apostrophes
  and non-ASCII characters, lowercases. Used by both models.
- `text_classification/stopwords.py` — static word lists (`STOP_WORDS`, `QUESTION_WORDS`,
  `NEGATION_WORDS`) used only by the basic model's feature engineering.
- `text_classification/features.py` — `add_features()` turns `clean_text` into five numeric
  columns (`word_count`, `char_count`, `any_neg`, `is_question`, `any_rare`) for the basic model.
  `any_rare` is computed from the 100 least-frequent non-stopword words in the whole corpus
  (`word_frequencies()` + `RARE_WORD_COUNT`), so it must be computed over the full dataset before
  the train/test split.
- `text_classification/data.py` — `load_dataset()` reads `final_dataset_basicmlmodel.csv` and
  adds `clean_text`. `FEATURE_COLUMNS`, `TEXT_COLUMN`, `LABEL_COLUMN` are the shared column-name
  constants used across the package.
- `text_classification/model.py` — both models share the same train/test split
  (`RANDOM_STATE = 27`, `TEST_SIZE = 0.1`) for comparability:
  - `train_basic_model()` — `GaussianNB` over the five `FEATURE_COLUMNS`. This is the original
    tutorial approach; scores ~59% accuracy / 0.31 F1.
  - `train_tfidf_model()` — `TfidfVectorizer` (unigrams+bigrams) → `LogisticRegression` pipeline
    trained directly on `clean_text`. Scores ~88% accuracy / 0.85 F1, illustrating why raw-text
    vectorization beats a handful of hand-picked summary stats.
- `main.py` — CLI entrypoint; `--basic`/`--tfidf` flags select which model(s) run (default: both).

When changing feature engineering or column names, `FEATURE_COLUMNS` in `data.py` and the
`X = dataset[...]` selection in `train_basic_model()` must stay in sync.
