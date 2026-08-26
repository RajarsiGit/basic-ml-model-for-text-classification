"""Feature engineering for the basic (non-vectorized) tweet classifier."""

import re
from typing import Iterable

import pandas as pd

from .stopwords import NEGATION_WORDS, QUESTION_WORDS, STOP_WORDS

_CONTRACTED_NEGATION_RE = re.compile(r"\wn't")

#: Number of least-frequent words considered "rare" for the ``any_rare`` feature.
RARE_WORD_COUNT = 100


def word_frequencies(clean_texts: pd.Series) -> pd.Series:
    """Return word counts across ``clean_texts``, with stopwords removed."""
    word_list = []
    for words in clean_texts.str.split():
        word_list.extend(words)
    freq = pd.Series(word_list, dtype=object).value_counts()
    return freq[~freq.index.isin(STOP_WORDS)]


def any_negation(words: Iterable[str]) -> int:
    """1 if any word is a negation term (e.g. "not", "isn't"), else 0."""
    return int(any(
        word in NEGATION_WORDS or _CONTRACTED_NEGATION_RE.search(word)
        for word in words
    ))


def any_rare_word(words: Iterable[str], rare_words: set) -> int:
    """1 if any word is in ``rare_words``, else 0."""
    return int(any(word in rare_words for word in words))


def is_question(words: Iterable[str]) -> int:
    """1 if any word is a question word (who/what/when/why/how), else 0."""
    return int(any(word in QUESTION_WORDS for word in words))


def add_features(dataset: pd.DataFrame, text_col: str = 'clean_text') -> pd.DataFrame:
    """Return a copy of ``dataset`` with basic NLP feature columns added.

    Adds: word_count, char_count, any_neg, is_question, any_rare.
    """
    dataset = dataset.copy()
    split_words = dataset[text_col].str.split()

    freq = word_frequencies(dataset[text_col])
    rare_words = set(freq.tail(RARE_WORD_COUNT).index)

    dataset['word_count'] = split_words.apply(len)
    dataset['any_neg'] = split_words.apply(any_negation)
    dataset['is_question'] = split_words.apply(is_question)
    dataset['any_rare'] = split_words.apply(lambda words: any_rare_word(words, rare_words))
    dataset['char_count'] = dataset[text_col].apply(len)

    return dataset
