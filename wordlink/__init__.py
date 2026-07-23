"""WordLink automation toolkit.

A solver + iOS automation pipeline for Boggle/Word-Hunt-style letter games such
as Triumph's WordLink. The pieces are independent so you can use just the solver
(pure Python, no extra deps) or the full capture-and-play loop.
"""

from .dictionary import load_trie, load_words
from .solver import Board, Solution, Solver, score_word, valid_path
from .trie import Trie

__all__ = [
    "Board",
    "Solution",
    "Solver",
    "Trie",
    "load_words",
    "load_trie",
    "score_word",
    "valid_path",
]

__version__ = "0.1.0"
