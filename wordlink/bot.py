"""Glue that turns a screen into gestures: capture -> read -> solve -> play."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from .automation import DragTiming, WDAClient, pixels_to_points
from .dictionary import load_trie
from .solver import Board, Solution, Solver
from .trie import Trie
from .vision import GridRegion, detect_grid_region, read_board


@dataclass
class BotConfig:
    rows: int = 4
    cols: int = 4
    min_length: int = 3
    max_words_per_round: Optional[int] = None
    round_seconds: Optional[float] = None
    dictionary_path: Optional[str] = None
    region: Optional[GridRegion] = None
    timing: DragTiming = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.timing is None:
            self.timing = DragTiming()


class WordLinkBot:
    """Plays one or more rounds of WordLink on a connected device."""

    def __init__(self, config: BotConfig, client: Optional[WDAClient] = None) -> None:
        self.config = config
        self.client = client or WDAClient()
        # The dictionary only needs words made of letters that can appear; we
        # rebuild the trie per board so out-of-alphabet words are pruned early.
        self._trie: Optional[Trie] = None

    def _build_solver(self, board: Board) -> Solver:
        trie = load_trie(
            self.config.dictionary_path,
            min_length=self.config.min_length,
            max_length=board.rows * board.cols,
            allowed_letters=board.distinct_letters,
        )
        return Solver(trie, min_length=self.config.min_length)

    def read_current_board(self) -> tuple[Board, GridRegion]:
        image = self.client.screenshot()
        region = self.config.region or detect_grid_region(
            image, self.config.rows, self.config.cols
        )
        board = read_board(image, region)
        return board, region

    def play_round(self, board: Board, region: GridRegion, scale: float) -> List[Solution]:
        """Solve ``board`` and drag out each word, best-scoring first.

        Returns the solutions that were played.
        """
        solver = self._build_solver(board)
        solutions = solver.solve(board)

        if self.config.max_words_per_round is not None:
            solutions = solutions[: self.config.max_words_per_round]

        played: List[Solution] = []
        deadline = (
            time.monotonic() + self.config.round_seconds
            if self.config.round_seconds
            else None
        )
        for sol in solutions:
            if deadline is not None and time.monotonic() >= deadline:
                break
            points = [
                pixels_to_points(region.cell_center(r, c), scale) for (r, c) in sol.path
            ]
            self.client.perform_drag(points, self.config.timing)
            played.append(sol)
            time.sleep(self.config.timing.between_words_ms / 1000.0)
        return played

    def run(self, rounds: int = 1) -> None:
        win_w, _ = self.client.window_size()
        for i in range(rounds):
            image = self.client.screenshot()
            scale = image.shape[1] / win_w  # screenshot px per logical point
            region = self.config.region or detect_grid_region(
                image, self.config.rows, self.config.cols
            )
            board = read_board(image, region)
            print(f"Round {i + 1}: board read as\n{board}")
            played = self.play_round(board, region, scale)
            print(f"Round {i + 1}: played {len(played)} words")
