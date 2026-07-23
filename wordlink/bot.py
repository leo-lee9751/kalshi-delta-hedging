"""Glue that turns a screen into gestures: capture -> read -> solve -> play.

Triumph's WordLink is a *dynamic* board: when a word is accepted the used tiles
are removed and new letters fall in, so a precomputed word list goes stale after
the very first word. The bot therefore runs a per-word loop:

    screenshot -> OCR -> solve -> drag the best word -> screenshot again
      -> if the board changed, the word was accepted: re-solve the new board
      -> if the board is unchanged, the word was rejected: blacklist it and try
         the next candidate (this avoids getting stuck replaying a word Triumph
         doesn't know)

When no playable word remains it can tap Reshuffle to get a fresh board.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .automation import DragTiming, WDAClient, pixels_to_points
from .dictionary import load_trie
from .solver import Board, Solution, Solver
from .vision import GridRegion, detect_grid_region, read_board

Grid = Tuple[Tuple[str, ...], ...]


@dataclass
class BotConfig:
    rows: int = 4
    cols: int = 4
    min_length: int = 3
    round_seconds: Optional[float] = None
    dictionary_path: Optional[str] = None
    region: Optional[GridRegion] = None
    timing: DragTiming = field(default_factory=DragTiming)
    # Seconds to wait after a drag for the refill animation before re-reading.
    settle_after_word: float = 0.45
    # Screen point to tap when out of words (the Reshuffle button). None = stop.
    reshuffle_xy: Optional[Tuple[float, float]] = None
    reshuffle_pause: float = 1.0


def _grid_key(board: Board) -> Grid:
    return tuple(tuple(row) for row in board.grid)


class WordLinkBot:
    """Plays WordLink on a connected device, re-scanning after every word."""

    def __init__(self, config: BotConfig, client: Optional[WDAClient] = None) -> None:
        self.config = config
        self.client = client or WDAClient()

    def _build_solver(self, board: Board) -> Solver:
        trie = load_trie(
            self.config.dictionary_path,
            min_length=self.config.min_length,
            max_length=board.rows * board.cols,
            allowed_letters=board.distinct_letters,
        )
        return Solver(trie, min_length=self.config.min_length)

    def _region_for(self, image) -> GridRegion:
        return self.config.region or detect_grid_region(
            image, self.config.rows, self.config.cols
        )

    def _read_board(self, image, region: GridRegion) -> Board:
        return read_board(image, region)

    def _play_word(self, sol: Solution, region: GridRegion, scale: float) -> None:
        points = [
            pixels_to_points(region.cell_center(r, c), scale) for (r, c) in sol.path
        ]
        self.client.perform_drag(points, self.config.timing)

    def _reshuffle(self, scale: float) -> bool:
        if self.config.reshuffle_xy is None:
            return False
        x, y = self.config.reshuffle_xy
        self.client.perform_tap((x / scale, y / scale))
        time.sleep(self.config.reshuffle_pause)
        return True

    def run(self, rounds: int = 1) -> None:
        """Play continuously until the timer expires (or ``rounds`` reshuffles)."""
        win_w, _ = self.client.window_size()

        image = self.client.screenshot()
        scale = image.shape[1] / win_w
        region = self._region_for(image)
        board = self._read_board(image, region)
        print(f"Start board:\n{board}\nscale={scale:.3f}")

        deadline = (
            time.monotonic() + self.config.round_seconds
            if self.config.round_seconds
            else None
        )
        played = 0
        rejected = 0
        reshuffles = 0
        blacklist: set[str] = set()

        while deadline is None or time.monotonic() < deadline:
            solver = self._build_solver(board)
            solutions = [s for s in solver.solve(board) if s.word not in blacklist]

            if not solutions:
                if reshuffles < rounds - 1 and self._reshuffle(scale):
                    reshuffles += 1
                    blacklist.clear()
                    image = self.client.screenshot()
                    region = self._region_for(image)
                    board = self._read_board(image, region)
                    print(f"Reshuffled ({reshuffles}). New board:\n{board}")
                    continue
                print("No playable words left; stopping.")
                break

            sol = solutions[0]
            self._play_word(sol, region, scale)
            time.sleep(self.config.settle_after_word)

            # Re-read to see whether the word was accepted (board changed).
            image = self.client.screenshot()
            new_board = self._read_board(image, region)

            if _grid_key(new_board) != _grid_key(board):
                played += 1
                blacklist.clear()
                board = new_board
                print(f"[{played}] played {sol.word} (score {sol.score})")
            else:
                rejected += 1
                blacklist.add(sol.word)

        print(
            f"Done. accepted={played} rejected={rejected} reshuffles={reshuffles}"
        )
