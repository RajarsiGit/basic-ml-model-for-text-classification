import pandas as pd

from text_classification.features import (
    add_features,
    any_negation,
    any_rare_word,
    is_question,
    word_frequencies,
)


def test_any_negation_detects_plain_and_contracted_forms():
    assert any_negation(['this', 'is', 'not', 'ok']) == 1
    assert any_negation(['it', "isn't", 'great']) == 1
    assert any_negation(['this', 'is', 'fine']) == 0


def test_is_question_detects_question_words():
    assert is_question(['why', 'so', 'serious']) == 1
    assert is_question(['this', 'is', 'fine']) == 0


def test_any_rare_word():
    rare = {'zebra', 'quokka'}
    assert any_rare_word(['a', 'zebra', 'ran'], rare) == 1
    assert any_rare_word(['a', 'cat', 'ran'], rare) == 0


def test_word_frequencies_drops_stopwords():
    texts = pd.Series(['the cat sat', 'the cat ran'])
    freq = word_frequencies(texts)
    assert 'the' not in freq.index
    assert freq['cat'] == 2


def test_add_features_adds_expected_columns():
    dataset = pd.DataFrame({'clean_text': ["why isn't the cat here", 'a calm day']})
    result = add_features(dataset)
    for col in ['word_count', 'any_neg', 'is_question', 'any_rare', 'char_count']:
        assert col in result.columns
    assert result.loc[0, 'word_count'] == 5
    assert result.loc[0, 'any_neg'] == 1
    assert result.loc[0, 'is_question'] == 1
