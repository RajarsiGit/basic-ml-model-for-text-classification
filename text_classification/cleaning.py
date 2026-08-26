"""Text cleaning utilities for raw tweet text."""

import re

# Anything that isn't a letter or an apostrophe becomes whitespace, then any
# remaining non-ASCII characters (emoji, mojibake) are dropped outright.
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z']")
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")


def clean_text(text: str) -> str:
    """Lowercase ``text`` and strip everything but letters and apostrophes."""
    text = _NON_ALPHA_RE.sub(' ', text)
    text = _NON_ASCII_RE.sub('', text)
    return text.lower()
