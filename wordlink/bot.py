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

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

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
    round_seconds: Optional[float] = None
    dictionary_path: Optional[str] = None
    region: Optional[GridRegion] = None
    timing: DragTiming = field(default_factory=DragTiming)
    # Short wait after a drag before the quick "did anything change?" check.
    quick_check: float = 0.13
    # Extra wait, only on an accepted word, for the refill animation to finish
    # before we OCR the new board.
    settle_after_word: float = 0.35
    # Mean grayscale pixel delta of the grid crop above which we treat the board
    # as possibly changed (and then confirm via OCR letters).
    change_threshold: float = 8.0
    # Screen point to tap when out of words (the Reshuffle button). None = stop.
    reshuffle_xy: Optional[Tuple[float, float]] = None
    reshuffle_pause: float = 1.0
    verbose: bool = False
    # WDA screenshot compression: 0 = original PNG, 1 = medium JPEG (faster),
    # 2 = low JPEG (fastest). JPEG is plenty for OCR and change detection.
    screenshot_quality: Optional[int] = 1
    # Directory to persist learned data across sessions. Writes accepted.txt (a
    # growing map of Triumph's real dictionary) and rejected.txt (words to skip).
    learn_dir: Optional[str] = None


class WordLinkBot:
    """Plays WordLink on a connected device, re-scanning after every word."""

    def __init__(self, config: BotConfig, client: Optional[WDAClient] = None) -> None:
        self.config = config
        self.client = client or WDAClient()
        # Build the dictionary/trie ONCE and reuse it for every board. The board
        # changes on each refill, but the dictionary does not, so rebuilding it
        # per word (as the first version did) was the main source of lag.
        self._solver: Optional[Solver] = None
        # Words Triumph has rejected (its dictionary is fixed, so a reject is
        # permanent). Never retried, on any board.
        self._rejected_words: Set[str] = set()
        # Words confirmed accepted — the empirically-learned Triumph dictionary.
        self._accepted_words: Set[str] = set()
        self._accept_path: Optional[str] = None
        self._reject_path: Optional[str] = None
        self._load_learned()

    def _load_learned(self) -> None:
        """Load previously learned accepted/rejected words from ``learn_dir``."""
        if not self.config.learn_dir:
            return
        os.makedirs(self.config.learn_dir, exist_ok=True)
        self._accept_path = os.path.join(self.config.learn_dir, "accepted.txt")
        self._reject_path = os.path.join(self.config.learn_dir, "rejected.txt")
        for path, target in ((self._accept_path, self._accepted_words),
                             (self._reject_path, self._rejected_words)):
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        w = line.strip().upper()
                        if w:
                            target.add(w)
        print(
            f"Loaded learned data: {len(self._accepted_words)} accepted, "
            f"{len(self._rejected_words)} rejected (from {self.config.learn_dir})"
        )

    def _record_accept(self, word: str) -> None:
        if word in self._accepted_words:
            return
        self._accepted_words.add(word)
        if self._accept_path:
            with open(self._accept_path, "a", encoding="utf-8") as fh:
                fh.write(word + "\n")

    def _record_reject(self, word: str) -> None:
        newly = word not in self._rejected_words
        self._rejected_words.add(word)
        if newly and self._reject_path:
            with open(self._reject_path, "a", encoding="utf-8") as fh:
                fh.write(word + "\n")

    def _get_solver(self) -> Solver:
        if self._solver is None:
            trie: Trie = load_trie(
                self.config.dictionary_path,
                min_length=self.config.min_length,
            )
            self._solver = Solver(trie, min_length=self.config.min_length)
        return self._solver

    @staticmethod
    def _region_crop(image, region: GridRegion):
        return image[
            region.top : region.top + region.height,
            region.left : region.left + region.width,
        ]

    @staticmethod
    def _regions_differ(a, b, threshold: float = 6.0) -> bool:
        """Cheap board-change test: mean pixel difference of the grid crop.

        Accepted words repaint the grid (new letters), rejected words leave it
        unchanged, so this distinguishes the two without re-running OCR.
        """
        import cv2

        if a.shape != b.shape:
            return True
        return float(cv2.absdiff(a, b).mean()) > threshold

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
        solver = self._get_solver()
        thresh = self.config.change_threshold
        if self.config.screenshot_quality is not None:
            self.client.set_screenshot_quality(self.config.screenshot_quality)

        image = self.client.screenshot()
        scale = image.shape[1] / win_w
        region = self._region_for(image)
        board = self._read_board(image, region)
        prev_crop = self._region_crop(image, region)
        print(f"Start board:\n{board}\nscale={scale:.3f}")

        deadline = (
            time.monotonic() + self.config.round_seconds
            if self.config.round_seconds
            else None
        )
        played = 0
        rejected = 0
        reshuffles = 0
        errors = 0
        # Words Triumph refused, remembered for the whole session so we never
        # waste another attempt on them (a word it rejects once won't appear in
        # its dictionary on any later board either).
        rejected_words: set[str] = self._rejected_words

        while deadline is None or time.monotonic() < deadline:
            try:
                solutions = [s for s in solver.solve(board) if s.word not in rejected_words]

                if not solutions:
                    # Recovery: maybe the board changed and our copy is stale.
                    image = self.client.screenshot()
                    if self._regions_differ(prev_crop, self._region_crop(image, region), thresh):
                        board = self._read_board(image, region)
                        prev_crop = self._region_crop(image, region)
                        continue
                    if reshuffles < rounds - 1 and self._reshuffle(scale):
                        reshuffles += 1
                        image = self.client.screenshot()
                        region = self._region_for(image)
                        board = self._read_board(image, region)
                        prev_crop = self._region_crop(image, region)
                        print(f"Reshuffled ({reshuffles}). New board:\n{board}")
                        continue
                    print("No playable words left; stopping.")
                    break

                sol = solutions[0]
                self._play_word(sol, region, scale)

                # Quick check: rejected words don't animate, so if nothing changed
                # after a short delay, move on immediately (no full settle wait).
                time.sleep(self.config.quick_check)
                image = self.client.screenshot()
                crop = self._region_crop(image, region)
                if not self._regions_differ(prev_crop, crop, thresh):
                    rejected += 1
                    self._record_reject(sol.word)
                    if self.config.verbose:
                        print(f"  x {sol.word} (rejected)")
                    errors = 0
                    continue

                # Something changed — let the refill finish, then confirm via letters.
                time.sleep(self.config.settle_after_word)
                image = self.client.screenshot()
                new_board = self._read_board(image, region)
                prev_crop = self._region_crop(image, region)
                if new_board.grid != board.grid:
                    played += 1
                    board = new_board
                    self._record_accept(sol.word)
                    print(f"[{played}] {sol.word} (+{sol.score})")
                    # Brief gap after an accepted word so the game finishes
                    # settling before the next drag starts.
                    if self.config.timing.between_words_ms:
                        time.sleep(self.config.timing.between_words_ms / 1000.0)
                else:
                    # Pixels flickered but letters are unchanged: it was rejected.
                    rejected += 1
                    self._record_reject(sol.word)
                    if self.config.verbose:
                        print(f"  x {sol.word} (rejected)")
                errors = 0

            except Exception as exc:  # transient WDA / USB / session hiccup
                errors += 1
                print(f"warning: {type(exc).__name__}: {exc} (retry {errors}/8)")
                if errors >= 8:
                    print("Too many consecutive errors; stopping.")
                    break
                self.client.reset_session()
                time.sleep(min(0.3 * (2 ** errors), 3.0))
                try:  # re-sync with whatever is currently on screen
                    if self.config.screenshot_quality is not None:
                        self.client.set_screenshot_quality(self.config.screenshot_quality)
                    image = self.client.screenshot()
                    region = self.config.region or self._region_for(image)
                    board = self._read_board(image, region)
                    prev_crop = self._region_crop(image, region)
                except Exception:
                    pass

        print(f"Done. accepted={played} rejected={rejected} reshuffles={reshuffles}")
