## Build a Basic ML Model for Text Classification

- In this notebook, you'll learn how to implement a text classification task using machine learning.
- You'll learn to create basic NLP-based features from raw text, then evaluate a model's performance on held-out test data.

To make things interesting, the task is to build a machine learning model to **classify** whether a particular tweet is **hate speech** or **not**. I'll explain more as you proceed further, so let's start without much ado!

### Table of Contents

1. [About the Dataset](#1-about-the-dataset)
2. [Text Cleaning](#2-text-cleaning)
3. [Feature Engineering](#3-feature-engineering)
4. [Train an ML Model for Text Classification](#4-train-an-ml-model-for-text-classification)
5. [Evaluate the ML Model](#5-evaluate-the-ml-model)
6. [Conclusion](#6-conclusion)
7. [Project Layout](#project-layout)

---

### 1. About the Dataset

The dataset is a collection of tweets labeled for **hate speech** detection: `final_dataset_basicmlmodel.csv`, with columns:

| column | meaning |
|---|---|
| `id` | row identifier |
| `label` | `1` = hate speech, `0` = not hate speech |
| `tweet` | the raw tweet text — the main data NLP techniques are applied to |

```python
import pandas as pd

dataset = pd.read_csv('final_dataset_basicmlmodel.csv')
dataset.head()
```

| id | label | tweet |
|---|---|---|
| 1 | 0 | @user when a father is dysfunctional and is s... |
| 2 | 0 | @user @user thanks for #lyft credit i can't us... |
| 3 | 0 | bihday your majesty |
| 4 | 0 | #model   i love u take with u all the time in ... |
| 5 | 0 | factsguide: society now    #motivation |

**Noise present in the tweets:**

- Hashtags (`#topic`) — we want the words but not the `#` symbol.
- Stray unicode characters like `â` and `ð` (mojibake) that carry no meaning.
- Numerals, punctuation, and percentages that add noise rather than signal.

### 2. Text Cleaning

```python
import re

def clean_text(text):
    # Keep only letters and apostrophes
    text = re.sub(r"[^a-zA-Z']", ' ', text)
    # Drop any remaining non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.lower()

dataset['clean_text'] = dataset.tweet.apply(clean_text)
```

### 3. Feature Engineering

Feature engineering is the science (and art) of extracting more information from data you
already have. A model can't read raw text directly, so we turn each tweet into a handful of
numeric features that stand in for the underlying text:

| feature | meaning |
|---|---|
| `word_count` | number of words in the cleaned tweet |
| `char_count` | number of characters in the cleaned tweet |
| `any_neg` | 1 if the tweet contains a negation word (`not`, `isn't`, ...) |
| `is_question` | 1 if the tweet contains a question word (`who`, `what`, `why`, ...) |
| `any_rare` | 1 if the tweet contains one of the 100 rarest words in the corpus |

```python
word_freq = gen_freq(dataset.clean_text)          # word -> frequency, stopwords removed
rare_100 = word_freq.tail(100)                    # 100 rarest words in the dataset

dataset['word_count'] = dataset.clean_text.str.split().apply(len)
dataset['any_neg'] = dataset.clean_text.str.split().apply(any_neg)
dataset['is_question'] = dataset.clean_text.str.split().apply(is_question)
dataset['any_rare'] = dataset.clean_text.str.split().apply(lambda w: any_rare(w, rare_100))
dataset['char_count'] = dataset.clean_text.apply(len)
```

Top 10 most common (non-stopword) words in the corpus:

```
user      3351
amp        439
love       320
day        254
trump      214
happy      207
will       191
people     186
new        171
u          158
```

### 4. Train an ML Model for Text Classification

The dataset is split into train and test sets so the model's performance can be evaluated on
data it has never seen — a standard practice in machine learning.

```python
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

X = dataset[['word_count', 'any_neg', 'any_rare', 'char_count', 'is_question']]
y = dataset.label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=27)

model = GaussianNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

### 5. Evaluate the ML Model

```python
from sklearn.metrics import accuracy_score

print("Accuracy:", accuracy_score(y_test, pred) * 100, "%")
# Accuracy: 59.05 %
```

### 6. Conclusion

Since we used very basic, hand-engineered NLP features, the classification accuracy and F1
score aren't that impressive (~59% accuracy). The goal of this exercise is to build intuition
for the model-building process: clean text → engineer features → split → train → evaluate.

For comparison, the [companion Python package](#project-layout) in this repo also includes a
**TF-IDF + logistic regression** model trained directly on the tweet text (rather than a few
coarse counts), which reaches **~88% accuracy / 0.85 F1** on the same split — a good illustration
of how much signal is lost by hand-picking only five summary statistics instead of letting the
model see the actual words.

---

### Project Layout

The walkthrough above lives in `Basic ML Model for Text Classification 2.ipynb`. The same logic
is also available as a small, tested Python package so it can be run and extended outside the
notebook:

```
text_classification/
    cleaning.py    # raw text -> cleaned text
    features.py    # cleaned text -> hand-engineered features
    stopwords.py   # stopword / question-word / negation-word lists
    data.py        # dataset loading
    model.py        # training + evaluation for both models
main.py            # CLI entrypoint
tests/             # pytest unit tests for cleaning/feature logic
```

Install dependencies and run:

```bash
pip install -r requirements.txt

python main.py             # train + evaluate both models
python main.py --basic     # only the hand-engineered-feature GaussianNB model
python main.py --tfidf     # only the TF-IDF + logistic regression model

python -m pytest           # run the test suite
```
