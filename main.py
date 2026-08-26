"""CLI entrypoint: train and evaluate tweet hate-speech classifiers.

Usage:
    python main.py            # run both the basic and TF-IDF models
    python main.py --basic    # run only the basic hand-engineered-feature model
    python main.py --tfidf    # run only the TF-IDF + logistic regression model
"""

import argparse

from text_classification.data import load_dataset
from text_classification.features import add_features
from text_classification.model import train_basic_model, train_tfidf_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--basic', action='store_true', help='run only the basic model')
    parser.add_argument('--tfidf', action='store_true', help='run only the TF-IDF model')
    args = parser.parse_args()

    run_basic = args.basic or not args.tfidf
    run_tfidf = args.tfidf or not args.basic

    dataset = load_dataset()

    if run_basic:
        featured = add_features(dataset)
        result = train_basic_model(featured)
        print(f"Basic model:  {result}")

    if run_tfidf:
        result = train_tfidf_model(dataset)
        print(f"TF-IDF model: {result}")


if __name__ == '__main__':
    main()
