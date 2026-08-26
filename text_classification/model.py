"""Model training and evaluation for the tweet hate-speech classifier."""

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from .data import FEATURE_COLUMNS, LABEL_COLUMN

RANDOM_STATE = 27
TEST_SIZE = 0.1


@dataclass
class EvalResult:
    accuracy: float
    f1: float

    def __str__(self) -> str:
        return f"accuracy={self.accuracy * 100:.2f}% f1={self.f1:.3f}"


def _split(dataset: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def train_basic_model(dataset: pd.DataFrame) -> EvalResult:
    """Train/evaluate a GaussianNB model on the hand-engineered features.

    This mirrors the original tutorial model: a handful of coarse counts
    (word count, char count, negation/question/rare-word flags).
    """
    X = dataset[FEATURE_COLUMNS]
    y = dataset[LABEL_COLUMN]
    X_train, X_test, y_train, y_test = _split(dataset, X, y)

    model = GaussianNB()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return EvalResult(accuracy_score(y_test, pred), f1_score(y_test, pred))


def train_tfidf_model(dataset: pd.DataFrame, text_col: str = 'clean_text') -> EvalResult:
    """Train/evaluate a TF-IDF + logistic regression model on the raw text.

    Uses the actual word content of each tweet (rather than a few coarse
    counts), which captures far more signal for this task.
    """
    X = dataset[text_col]
    y = dataset[LABEL_COLUMN]
    X_train, X_test, y_train, y_test = _split(dataset, X, y)

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=2, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)

    return EvalResult(accuracy_score(y_test, pred), f1_score(y_test, pred))
