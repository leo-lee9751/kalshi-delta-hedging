"""Tests for the WordLink solver, trie, scoring and gesture construction."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wordlink.automation import DragTiming, build_pointer_actions, pixels_to_points
from wordlink.solver import Board, Solver, score_word, valid_path
from wordlink.trie import Trie


def make_solver(words, min_length=3):
    trie = Trie.from_words(w.upper() for w in words)
    return Solver(trie, min_length=min_length)


# ---------------------------------------------------------------------------
# Trie
# ---------------------------------------------------------------------------

def test_trie_membership_and_prefix():
    t = Trie.from_words(["CAT", "CATS", "DOG"])
    assert "CAT" in t
    assert "CATS" in t
    assert "COW" not in t
    assert t.has_prefix("CA")
    assert t.has_prefix("CAT")
    assert not t.has_prefix("CB")
    assert len(t) == 3


def test_trie_dedup_counts_once():
    t = Trie()
    t.insert("CAT")
    t.insert("CAT")
    assert len(t) == 1


# ---------------------------------------------------------------------------
# Board parsing
# ---------------------------------------------------------------------------

def test_board_from_lines_char_split():
    b = Board.from_lines(["EI", "LL"])
    assert b.rows == 2 and b.cols == 2
    assert b.grid == [["E", "I"], ["L", "L"]]


def test_board_from_lines_whitespace_split():
    b = Board.from_lines(["E I T", "L L F"])
    assert b.cols == 3
    assert b.grid[0] == ["E", "I", "T"]


def test_board_rejects_ragged_rows():
    try:
        Board(["ABC", "AB"])
    except ValueError:
        return
    raise AssertionError("expected ValueError for ragged rows")


def test_board_distinct_letters():
    b = Board.from_lines(["AAB", "BCC"])
    assert b.distinct_letters == "ABC"


# ---------------------------------------------------------------------------
# Solver correctness
# ---------------------------------------------------------------------------

def test_finds_simple_horizontal_word():
    board = Board.from_lines(["CAT", "XXX", "XXX"])
    sols = make_solver(["CAT"]).solve(board)
    words = {s.word for s in sols}
    assert "CAT" == sols[0].word
    assert "CAT" in words


def test_uses_diagonal_adjacency():
    # C at (0,0), A at (1,1), T at (2,2): purely diagonal path.
    board = Board.from_lines(["CXX", "XAX", "XXT"])
    sols = make_solver(["CAT"]).solve(board)
    assert any(s.word == "CAT" for s in sols)


def test_path_is_valid_and_adjacent():
    board = Board.from_lines(["CAT", "XXX", "XXX"])
    sol = make_solver(["CAT"]).solve(board)[0]
    assert sol.path == ((0, 0), (0, 1), (0, 2))
    assert valid_path(sol.path)


def test_no_tile_reuse():
    # Only one 'O'; "OOO" must NOT be findable without reusing the tile.
    board = Board.from_lines(["OXX", "XXX", "XXX"])
    sols = make_solver(["OOO"]).solve(board)
    assert all(s.word != "OOO" for s in sols)


def test_word_needs_connected_path():
    # C and A adjacent, but T is isolated in the far corner (not adjacent to A).
    board = Board.from_lines(["CAX", "XXX", "XXT"])
    sols = make_solver(["CAT"]).solve(board)
    assert all(s.word != "CAT" for s in sols)


def test_min_length_filter():
    board = Board.from_lines(["AT", "XX"])
    sols = make_solver(["AT"], min_length=3).solve(board)
    assert sols == []
    sols2 = make_solver(["AT"], min_length=2).solve(board)
    assert any(s.word == "AT" for s in sols2)


def test_results_sorted_by_score_desc():
    scores = [s.score for s in make_solver(["CAT", "CATS", "ACT"]).solve(
        Board.from_lines(["CATS", "XXXX", "XXXX", "XXXX"])
    )]
    assert scores == sorted(scores, reverse=True)


def test_duplicate_word_returned_once():
    # Two ways to spell "OXO"-ish; ensure a word appears at most once.
    board = Board.from_lines(["ANA", "XXX", "XXX"])
    sols = make_solver(["ANA"]).solve(board)
    assert len([s for s in sols if s.word == "ANA"]) <= 1


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def test_score_word_rewards_length_and_value():
    assert score_word("QUIZ") > score_word("CAT")
    assert score_word("CATS") > score_word("CAT")


def test_score_word_letter_values():
    # "AAA": 3 * value(A=1) + length_bonus(1)*3 = 6
    assert score_word("AAA", length_bonus=1) == 6


# ---------------------------------------------------------------------------
# Gesture construction (automation)
# ---------------------------------------------------------------------------

def test_build_pointer_actions_structure():
    payload = build_pointer_actions([(10, 10), (20, 20), (30, 30)], DragTiming())
    seq = payload["actions"][0]["actions"]
    types = [a["type"] for a in seq]
    assert types[0] == "pointerMove"
    assert "pointerDown" in types
    assert types[-1] == "pointerUp"
    # Two pointerMove entries for the two subsequent tiles + the initial move.
    assert types.count("pointerMove") == 3


def test_build_pointer_actions_requires_two_points():
    try:
        build_pointer_actions([(1, 1)], DragTiming())
    except ValueError:
        return
    raise AssertionError("expected ValueError for single-point path")


def test_pixels_to_points_scale():
    assert pixels_to_points((300, 600), scale=3.0) == (100.0, 200.0)


# ---------------------------------------------------------------------------
# Integration with the bundled dictionary
# ---------------------------------------------------------------------------

def test_bundled_dictionary_finds_real_words():
    from wordlink.dictionary import load_trie

    board = Board.from_lines(["EITP", "LLFO", "IISE", "ETLL"])
    trie = load_trie(min_length=3, allowed_letters=board.distinct_letters)
    sols = Solver(trie, min_length=3).solve(board)
    words = {s.word for s in sols}
    # These are all traceable on the example board with real English words.
    assert "TILE" in words
    assert "LIFE" in words or "FILE" in words
    assert len(sols) > 20
    for s in sols:
        assert valid_path(s.path)
