"""Loading and filtering the word list used by the solver.

The bundled list is ENABLE1 (Enhanced North American Benchmark Lexicon), the
same public-domain lexicon used by most Boggle / Word-Hunt solvers. It is a good
match for the kind of dictionary skill-word games like Triumph's WordLink accept.

You can swap in your own list with ``load_words(path=...)`` if the game turns out
to accept a different set of words.
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Set

from .trie import Trie

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
# Full ENABLE1 lexicon (~172k words) — maximal coverage, but includes many
# obscure words that games like Triumph's WordLink reject.
FULL_DICTIONARY = os.path.join(_DATA_DIR, "enable1.txt")
# Frequency-filtered subset (~18k of the most common everyday ENABLE1 words).
# This is the default for actually *playing*: it drops the rare loanword /
# Scrabble-dictionary tail (e.g. SPORTIF, BEAUX, ROCOCO) that games like
# Triumph's WordLink reject, so far fewer attempts are wasted.
COMMON_DICTIONARY = os.path.join(_DATA_DIR, "common.txt")
BUNDLED_DICTIONARY = COMMON_DICTIONARY


def load_words(
    path: Optional[str] = None,
    *,
    min_length: int = 3,
    max_length: Optional[int] = None,
    allowed_letters: Optional[Iterable[str]] = None,
) -> List[str]:
    """Read a word list from disk and normalise it.

    Args:
        path: File with one word per line. Defaults to the bundled ENABLE1 list.
        min_length: Drop words shorter than this. WordLink technically allows
            2-letter words, but they score almost nothing and the board rarely
            benefits from them, so the default is 3.
        max_length: Drop words longer than this (``None`` keeps all).
        allowed_letters: If given, only keep words made entirely of these
            letters. Passing the board's distinct letters here shrinks the trie
            substantially and speeds up construction.

    Returns:
        A de-duplicated list of uppercase words.
    """
    path = path or BUNDLED_DICTIONARY
    allowed: Optional[Set[str]] = None
    if allowed_letters is not None:
        allowed = {c.upper() for c in allowed_letters}

    words: List[str] = []
    seen: Set[str] = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            word = raw.strip().upper()
            if not word or not word.isalpha():
                continue
            if len(word) < min_length:
                continue
            if max_length is not None and len(word) > max_length:
                continue
            if allowed is not None and not set(word) <= allowed:
                continue
            if word in seen:
                continue
            seen.add(word)
            words.append(word)
    return words


def load_trie(
    path: Optional[str] = None,
    *,
    min_length: int = 3,
    max_length: Optional[int] = None,
    allowed_letters: Optional[Iterable[str]] = None,
) -> Trie:
    """Convenience wrapper: load words and build a :class:`Trie` from them."""
    words = load_words(
        path,
        min_length=min_length,
        max_length=max_length,
        allowed_letters=allowed_letters,
    )
    return Trie.from_words(words)
