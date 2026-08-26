"""Dataset loading helpers."""

from pathlib import Path

import pandas as pd

from .cleaning import clean_text

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / 'final_dataset_basicmlmodel.csv'

FEATURE_COLUMNS = ['word_count', 'any_neg', 'any_rare', 'char_count', 'is_question']
TEXT_COLUMN = 'tweet'
LABEL_COLUMN = 'label'


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> pd.DataFrame:
    """Load the raw tweet dataset and add a cleaned-text column."""
    dataset = pd.read_csv(path)
    dataset['clean_text'] = dataset[TEXT_COLUMN].apply(clean_text)
    return dataset
