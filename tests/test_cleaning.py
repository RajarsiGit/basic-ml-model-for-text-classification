from text_classification.cleaning import clean_text


def test_strips_punctuation_and_digits():
    assert clean_text("Hello, World! 123") == 'hello  world     '


def test_keeps_apostrophes():
    assert clean_text("can't") == "can't"


def test_strips_non_ascii():
    assert clean_text("angryð#got7") == 'angry  got '


def test_lowercases():
    assert clean_text("SHOUTING") == 'shouting'
