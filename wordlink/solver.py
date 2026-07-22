"""Core word-finding logic for a WordLink-style letter grid.

Given a rectangular board of letter tiles, find every dictionary word that can be
traced by moving between *adjacent* tiles (horizontally, vertically or
diagonally) without reusing a tile. Each found word comes with the exact path of
tile coordinates, which the automation layer replays as a finger drag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .trie import Trie

# Standard Scrabble letter values. WordLink scores a word from the value of its
# letters plus a bonus for length, which this mirrors closely enough to rank
# words the way the game does (highest-value words first).
LETTER_VALUES: Dict[str, int] = {
    **dict.fromkeys("AEILNORSTU", 1),
    **dict.fromkeys("DG", 2),
    **dict.fromkeys("BCMP", 3),
    **dict.fromkeys("FHVWY", 4),
    "K": 5,
    **dict.fromkeys("JX", 8),
    **dict.fromkeys("QZ", 10),
}

# The 8 king-move neighbour offsets.
_NEIGHBOURS: Tuple[Tuple[int, int], ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)

Coord = Tuple[int, int]


def score_word(word: str, *, length_bonus: int = 1) -> int:
    """Approximate WordLink's score for a single word.

    Sum of Scrabble letter values plus ``length_bonus`` per letter. The exact
    constants the game uses are not published, but any monotonic-in-length,
    monotonic-in-letter-value function ranks words in the same order, which is
    all the solver needs to "play the best words first".
    """
    base = sum(LETTER_VALUES.get(ch, 1) for ch in word)
    return base + length_bonus * len(word)


@dataclass(frozen=True)
class Solution:
    """A single found word and the tile path that spells it."""

    word: str
    path: Tuple[Coord, ...]
    score: int

    @property
    def length(self) -> int:
        return len(self.word)


@dataclass
class Board:
    """A rectangular grid of letter tiles.

    ``grid`` is a list of rows; each cell is the tile's text. Tiles are usually a
    single letter but may be multi-character (e.g. a "Qu" tile), which the solver
    handles transparently.
    """

    grid: List[List[str]]

    def __post_init__(self) -> None:
        self.grid = [[str(cell).upper() for cell in row] for row in self.grid]
        widths = {len(row) for row in self.grid}
        if len(widths) > 1:
            raise ValueError(f"Board rows have inconsistent widths: {widths}")

    @property
    def rows(self) -> int:
        return len(self.grid)

    @property
    def cols(self) -> int:
        return len(self.grid[0]) if self.grid else 0

    def letter(self, coord: Coord) -> str:
        r, c = coord
        return self.grid[r][c]

    @property
    def distinct_letters(self) -> str:
        chars = {ch for row in self.grid for cell in row for ch in cell}
        return "".join(sorted(chars))

    @classmethod
    def from_lines(cls, lines: Sequence[str]) -> "Board":
        """Build a board from text lines.

        Each line becomes a row; tiles are split on whitespace when present,
        otherwise each character is its own tile. So both of these describe the
        same 2x2 board::

            "E I\nL L"
            "EI\nLL"
        """
        grid: List[List[str]] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            tiles = line.split() if (" " in line or "\t" in line) else list(line)
            grid.append(tiles)
        if not grid:
            raise ValueError("Cannot build a board from empty input")
        return cls(grid)

    @classmethod
    def from_string(cls, text: str) -> "Board":
        return cls.from_lines(text.splitlines())

    def __str__(self) -> str:
        width = max((len(cell) for row in self.grid for cell in row), default=1)
        return "\n".join(" ".join(cell.rjust(width) for cell in row) for row in self.grid)


class Solver:
    """Finds all dictionary words on a board using trie-pruned DFS."""

    def __init__(self, trie: Trie, *, min_length: int = 3, length_bonus: int = 1) -> None:
        self.trie = trie
        self.min_length = min_length
        self.length_bonus = length_bonus

    def solve(self, board: Board) -> List[Solution]:
        """Return every unique word on the board, best (highest score) first.

        When a word can be traced along multiple paths, the highest-scoring path
        is kept (scores are equal for a fixed word, so effectively the first path
        found is used).
        """
        best: Dict[str, Solution] = {}
        rows, cols = board.rows, board.cols

        # Reusable visited matrix avoids per-path allocation.
        visited = [[False] * cols for _ in range(rows)]

        def dfs(r: int, c: int, node, prefix: str, path: List[Coord]) -> None:
            tile = board.grid[r][c]
            child = node
            # Walk the (possibly multi-character) tile through the trie.
            for ch in tile:
                child = child.children.get(ch)
                if child is None:
                    return
            prefix += tile
            path.append((r, c))
            visited[r][c] = True

            if child.is_word is not None and len(child.is_word) >= self.min_length:
                word = child.is_word
                if word not in best:
                    best[word] = Solution(
                        word=word,
                        path=tuple(path),
                        score=score_word(word, length_bonus=self.length_bonus),
                    )

            # Only keep exploring if some real word still extends this prefix.
            if child.children:
                for dr, dc in _NEIGHBOURS:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]:
                        dfs(nr, nc, child, prefix, path)

            visited[r][c] = False
            path.pop()

        for r in range(rows):
            for c in range(cols):
                dfs(r, c, self.trie.root, "", [])

        solutions = list(best.values())
        solutions.sort(key=lambda s: (-s.score, -s.length, s.word))
        return solutions


def valid_path(path: Sequence[Coord]) -> bool:
    """True if every consecutive pair of coords is king-adjacent and unique."""
    if len(set(path)) != len(path):
        return False
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if max(abs(r1 - r2), abs(c1 - c2)) != 1:
            return False
    return True
