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
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from .automation import (
    DragTiming,
    WDAClient,
    jitter_point,
    jittered_ms,
    pixels_to_points,
)
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

    # --- Humanization: make the bot's behaviour look less machine-like ---
    # A human doesn't play at a constant superhuman rate or find every word;
    # these knobs shape *which* words are played, *how many*, and *how fast*.
    #
    # Instead of always the single highest-scoring word, choose randomly among
    # the top ``top_choice_n`` candidates, weighted by score. 1 = strict best.
    top_choice_n: int = 4
    # Per-word chance to skip a findable word for the rest of this board (humans
    # miss plenty). Longer/rarer words are skipped a little more often.
    skip_prob: float = 0.06
    # Cap on how fast accepted words are played (words/minute). None disables.
    max_words_per_minute: Optional[float] = 40.0
    # Stop after roughly this many accepted words per game (jittered +/-15%).
    # None plays until the timer/words run out.
    max_words: Optional[int] = None
    # Occasionally pause longer, as if searching the board ("thinking").
    think_prob: float = 0.12
    think_seconds: Tuple[float, float] = (0.6, 1.8)
    # Rarely take a longer idle break mid-round.
    idle_prob: float = 0.03
    idle_seconds: Tuple[float, float] = (2.0, 5.0)
    # Gradually slow down over a session (delays grow by this fraction / minute,
    # capped). 0 disables fatigue drift.
    fatigue_per_min: float = 0.1
    # Chance of a "false start": a tiny hesitation gesture near the first tile
    # before the real drag (never long enough to register a word).
    false_start_prob: float = 0.05


def select_word(candidates: List[Solution], rng: random.Random, top_n: int) -> Solution:
    """Pick a word to play from ``candidates`` (already best-first).

    Rather than always the single top word (a deterministic, optimal-every-time
    signature), sample among the top ``top_n`` weighted by score, so a good word
    is played but the *order* varies the way a human's does.
    """
    top = candidates[: max(1, top_n)]
    if len(top) == 1:
        return top[0]
    weights = [max(1, c.score) for c in top]
    return rng.choices(top, weights=weights, k=1)[0]


class WordLinkBot:
    """Plays WordLink on a connected device, re-scanning after every word."""

    def __init__(self, config: BotConfig, client: Optional[WDAClient] = None) -> None:
        self.config = config
        self.client = client or WDAClient()
        # RNG for timing jitter and behavioural randomness (drags jitter too).
        self._rng = random.Random()
        # Words intentionally skipped on the current board (cleared on refill /
        # reshuffle). Distinct from rejected words, which are gone permanently.
        self._skipped: Set[str] = set()
        # Timing state for rate limiting and fatigue.
        self._start_time: float = 0.0
        self._last_word_time: float = 0.0
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
        cell_w = region.width / region.cols
        cell_h = region.height / region.rows
        pj = self.config.timing.position_jitter
        points = []
        for (r, c) in sol.path:
            center = region.cell_center(r, c)
            aim = jitter_point(center, cell_w, cell_h, pj, self._rng)
            points.append(pixels_to_points(aim, scale))
        # Occasional hesitation before committing to the word.
        if self.config.false_start_prob and self._rng.random() < self.config.false_start_prob:
            self._false_start(points)
        self.client.perform_drag(points, self.config.timing, rng=self._rng)

    def _false_start(self, points) -> None:
        """A tiny wiggle near the first tile, then a pause — a human hesitation.

        Kept short (two near-identical points) so it can never spell a word.
        """
        x0, y0 = points[0]
        nudge = [(x0, y0), (x0 + self._rng.uniform(-4, 4), y0 + self._rng.uniform(-4, 4))]
        try:
            self.client.perform_drag(nudge, self.config.timing, rng=self._rng)
        except Exception:
            return
        time.sleep(self._rng.uniform(0.15, 0.45))

    def _skip_probability(self, sol: Solution) -> float:
        """Chance to skip a findable word — a bit higher for long/rare words."""
        base = self.config.skip_prob
        extra = 0.02 * max(0, sol.length - 5)
        return min(0.6, base + extra)

    def _fatigue_factor(self) -> float:
        """Multiplier (>=1) that slowly grows over the session, capped at 2.5."""
        if self.config.fatigue_per_min <= 0 or not self._start_time:
            return 1.0
        minutes = (time.monotonic() - self._start_time) / 60.0
        return min(1.0 + self.config.fatigue_per_min * minutes, 2.5)

    def _post_word_delay(self) -> None:
        """Pace the bot after an accepted word: rate limit + human pauses."""
        fatigue = self._fatigue_factor()
        # Rate limit: don't exceed the target words-per-minute.
        wpm = self.config.max_words_per_minute
        if wpm and wpm > 0:
            min_interval = (60.0 / wpm) * self._rng.uniform(0.85, 1.25)
            elapsed = time.monotonic() - self._last_word_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
        # Exactly one of: a rare long idle, a "thinking" pause, or the normal gap.
        r = self._rng.random()
        if r < self.config.idle_prob:
            time.sleep(self._rng.uniform(*self.config.idle_seconds))
        elif r < self.config.idle_prob + self.config.think_prob:
            time.sleep(self._rng.uniform(*self.config.think_seconds) * fatigue)
        elif self.config.timing.between_words_ms:
            gap = jittered_ms(
                self.config.timing.between_words_ms,
                self.config.timing.jitter,
                self._rng,
                self.config.timing.distribution,
            )
            time.sleep(gap / 1000.0 * fatigue)
        self._last_word_time = time.monotonic()

    def _reshuffle(self, scale: float) -> bool:
        if self.config.reshuffle_xy is None:
            return False
        x, y = self.config.reshuffle_xy
        self.client.perform_tap((x / scale, y / scale))
        self._skipped.clear()
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
        self._start_time = time.monotonic()
        self._last_word_time = time.monotonic()
        # Human-like completeness cap: don't clear every word — stop around a
        # jittered target so runs aren't identical or exhaustive.
        words_target: Optional[int] = None
        if self.config.max_words:
            words_target = max(1, int(self.config.max_words * self._rng.uniform(0.85, 1.15)))
            print(f"Word target this game: {words_target}")

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
                if words_target is not None and played >= words_target:
                    print(f"Reached word target ({played}); stopping.")
                    break

                solutions = [
                    s
                    for s in solver.solve(board)
                    if s.word not in rejected_words and s.word not in self._skipped
                ]
                # Randomly skip some findable words this board (humans miss words,
                # especially long/rare ones). Only ever skip when alternatives
                # remain, so we never deadlock a board we could still play.
                for cand in list(solutions[: self.config.top_choice_n]):
                    if len(solutions) <= 1:
                        break
                    if self._rng.random() < self._skip_probability(cand):
                        self._skipped.add(cand.word)
                        solutions.remove(cand)

                if not solutions:
                    # Recovery: maybe the board changed and our copy is stale.
                    image = self.client.screenshot()
                    if self._regions_differ(prev_crop, self._region_crop(image, region), thresh):
                        board = self._read_board(image, region)
                        prev_crop = self._region_crop(image, region)
                        self._skipped.clear()
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

                sol = select_word(solutions, self._rng, self.config.top_choice_n)
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
                    # The board refilled — words skipped on the old layout are
                    # fair game again.
                    self._skipped.clear()
                    self._record_accept(sol.word)
                    print(f"[{played}] {sol.word} (+{sol.score})")
                    # Pace the next move: rate limit + human-like pauses (jittered
                    # gap, occasional "thinking" or idle break, fatigue drift).
                    self._post_word_delay()
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
