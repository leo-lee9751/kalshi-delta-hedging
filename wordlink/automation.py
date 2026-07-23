"""Drive an iPhone through WebDriverAgent (WDA) to play the board.

WebDriverAgent is Facebook/Appium's on-device automation server. Once it is
running on the iPhone (see README_wordlink.md) it exposes an HTTP API on a port
that is forwarded to the controlling machine. This module speaks that API
directly with ``requests`` so it can:

  * grab a screenshot of the current board,
  * read the device's logical screen size,
  * replay a solved word as a single continuous finger drag using W3C pointer
    actions (the same primitive a real swipe uses).

Nothing here talks to the game directly; it just moves a finger along the tile
coordinates the solver produced. A ``dry_run`` mode logs the gestures instead of
sending them, so the whole pipeline can be exercised without a device.
"""

from __future__ import annotations

import base64
import math
import random
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import requests

Point = Tuple[float, float]

# Shared RNG for timing jitter. A module-level default keeps call sites simple;
# tests pass their own seeded ``random.Random`` for deterministic output.
_DEFAULT_RNG = random.Random()


@dataclass
class DragTiming:
    """Timing + motion knobs for how a word is dragged.

    Slower is more reliable; faster clears more words within the round timer.
    ``tile_dwell_ms`` is the important one for accuracy: pausing on each tile
    guarantees the game registers it instead of skipping a fast fly-over.

    Beyond raw speed, the remaining knobs make the swipe look *human* rather
    than machine-perfect (uniform timing and dead-straight lines to exact pixel
    centres are obvious automation tells):

    * ``jitter`` — randomize each hold/move/dwell duration by +/- this fraction.
    * ``distribution`` — ``"gaussian"`` clusters durations near the base value
      with the occasional outlier (more human than a flat ``"uniform"`` band).
    * ``curve`` — bow each tile-to-tile segment sideways by up to this fraction
      of its length, so the trace is a hand-drawn arc, not a polygon.
    * ``points_per_segment`` — how many intermediate points sample each arc.
    * ``overshoot_prob`` / ``overshoot_frac`` — occasionally slide just past a
      tile and correct back, the way a real finger does.
    * ``position_jitter`` — aim at a random spot inside the tile instead of the
      exact centre (applied by the bot, which knows the tile size).

    Setting the humanization knobs to ``0`` restores the old fixed behaviour.
    """

    press_hold_ms: int = 40
    move_ms_per_tile: int = 55
    tile_dwell_ms: int = 35
    settle_ms: int = 40
    between_words_ms: int = 120
    jitter: float = 0.4
    distribution: str = "gaussian"
    curve: float = 0.12
    points_per_segment: int = 4
    overshoot_prob: float = 0.12
    overshoot_frac: float = 0.18
    position_jitter: float = 0.28


def jittered_ms(
    value: int,
    jitter: float,
    rng: random.Random,
    distribution: str = "gaussian",
) -> int:
    """Randomize ``value`` by +/- ``jitter`` (a fraction), clamped at >= 0.

    ``distribution`` selects the sampling shape:

    * ``"gaussian"`` — normal draw centred on ``value`` (sigma = a third of the
      band) then clamped to ``[value*(1-jitter), value*(1+jitter)]``. Most
      samples sit near the base value with rarer extremes, like human timing.
    * ``"uniform"`` — flat draw across the whole band.

    With ``jitter <= 0`` (or a non-positive base) the value is returned
    unchanged, so timings stay uniform when jitter is disabled.
    """
    if value <= 0 or jitter <= 0:
        return int(value)
    low = value * (1.0 - jitter)
    high = value * (1.0 + jitter)
    if distribution == "gaussian":
        sample = rng.gauss(value, (value * jitter) / 3.0)
    else:
        sample = rng.uniform(low, high)
    return max(0, int(round(min(max(sample, low), high))))


def jitter_point(
    center: Point, cell_w: float, cell_h: float, frac: float, rng: random.Random
) -> Point:
    """Offset ``center`` by up to ``frac`` of the tile half-size in each axis.

    Aiming at a random spot inside the tile (rather than always the exact
    centre) removes the pixel-perfect landing that gives automation away, while
    staying well inside the tile so the right letter is still hit. ``frac <= 0``
    returns the centre unchanged.
    """
    if frac <= 0:
        return center
    cx, cy = center
    dx = rng.uniform(-1.0, 1.0) * frac * (cell_w / 2.0)
    dy = rng.uniform(-1.0, 1.0) * frac * (cell_h / 2.0)
    return (cx + dx, cy + dy)


def _segment_points(
    p0: Point, p1: Point, timing: DragTiming, rng: random.Random
) -> List[Point]:
    """Sample the (possibly curved) path from ``p0`` to ``p1``.

    Returns the intermediate points *plus* ``p1`` as the final entry. A single
    ``[p1]`` is returned when curving is disabled, reproducing a straight hop.
    """
    n = max(1, timing.points_per_segment)
    if n == 1 and timing.curve <= 0:
        return [p1]
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length == 0:
        return [p1]
    # Control point: midpoint pushed perpendicular to the segment by a random,
    # length-proportional amount -> a quadratic Bezier arc.
    ux, uy = -dy / length, dx / length
    offset = rng.uniform(-1.0, 1.0) * timing.curve * length
    cx = (x0 + x1) / 2.0 + ux * offset
    cy = (y0 + y1) / 2.0 + uy * offset
    pts: List[Point] = []
    for i in range(1, n + 1):
        t = i / n
        mt = 1.0 - t
        bx = mt * mt * x0 + 2 * mt * t * cx + t * t * x1
        by = mt * mt * y0 + 2 * mt * t * cy + t * t * y1
        pts.append((bx, by))
    return pts


def build_pointer_actions(
    waypoints: Sequence[Point],
    timing: DragTiming,
    rng: Optional[random.Random] = None,
) -> dict:
    """Construct the W3C ``/actions`` payload for a continuous drag.

    The finger presses down on the first tile, moves through each subsequent
    tile, then lifts. This is what makes the game register a single word rather
    than a series of taps. ``waypoints`` are the per-tile aim points (already
    position-jittered by the caller). Between consecutive waypoints the path is
    sampled along a slight arc, each sub-move's duration is jittered, and the
    finger occasionally overshoots a tile and corrects — so no two keystrokes
    take the same time and the trace isn't a machine-perfect polygon.
    """
    if len(waypoints) < 2:
        raise ValueError("A word path needs at least two tiles to drag between")

    rng = rng or _DEFAULT_RNG

    def j(value: int) -> int:
        return jittered_ms(value, timing.jitter, rng, timing.distribution)

    x0, y0 = waypoints[0]
    actions: List[dict] = [
        {"type": "pointerMove", "duration": 0, "x": int(x0), "y": int(y0)},
        {"type": "pointerDown", "button": 0},
        {"type": "pause", "duration": j(timing.press_hold_ms)},
    ]
    prev = waypoints[0]
    for wp in waypoints[1:]:
        seg = _segment_points(prev, wp, timing, rng)
        # Spread the per-tile move time across the sub-moves so overall speed is
        # preserved regardless of how finely the arc is sampled.
        per = max(1, int(round(timing.move_ms_per_tile / len(seg))))
        for (px, py) in seg:
            actions.append(
                {"type": "pointerMove", "duration": j(per), "x": int(px), "y": int(py)}
            )
        # Occasionally slide just past the tile, then snap back to it.
        if timing.overshoot_prob > 0 and rng.random() < timing.overshoot_prob:
            odx, ody = wp[0] - prev[0], wp[1] - prev[1]
            ox, oy = wp[0] + odx * timing.overshoot_frac, wp[1] + ody * timing.overshoot_frac
            actions.append({"type": "pointerMove", "duration": j(max(1, per // 2)), "x": int(ox), "y": int(oy)})
            actions.append({"type": "pointerMove", "duration": j(max(1, per // 2)), "x": int(wp[0]), "y": int(wp[1])})
        # Dwell on each tile so the game registers it before moving on.
        if timing.tile_dwell_ms:
            actions.append({"type": "pause", "duration": j(timing.tile_dwell_ms)})
        prev = wp
    actions.append({"type": "pause", "duration": j(timing.settle_ms)})
    actions.append({"type": "pointerUp", "button": 0})

    return {
        "actions": [
            {
                "type": "pointer",
                "id": "finger1",
                "parameters": {"pointerType": "touch"},
                "actions": actions,
            }
        ]
    }


@dataclass
class WDAClient:
    """Minimal WebDriverAgent HTTP client."""

    base_url: str = "http://localhost:8100"
    dry_run: bool = False
    timeout: float = 30.0
    _session_id: Optional[str] = field(default=None, init=False, repr=False)

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def reset_session(self) -> None:
        """Forget the cached session so the next call creates a fresh one.

        Used to recover after a transient WDA/USB error.
        """
        self._session_id = None

    def set_screenshot_quality(self, quality: int) -> None:
        """Ask WDA to return compressed JPEG screenshots (much smaller/faster).

        quality: 0 = original PNG (largest, slowest), 1 = medium JPEG,
        2 = low JPEG. Best-effort; ignored if the WDA build doesn't support it.
        """
        try:
            sid = self.session_id()
            requests.post(
                self._url(f"/session/{sid}/appium/settings"),
                json={"settings": {"screenshotQuality": quality}},
                timeout=self.timeout,
            )
        except Exception:
            pass

    def status(self) -> dict:
        resp = requests.get(self._url("/status"), timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def session_id(self) -> str:
        """Return an active session id, creating one if necessary."""
        if self._session_id:
            return self._session_id
        # Reuse the currently-focused app rather than launching one.
        resp = requests.post(
            self._url("/session"),
            json={"capabilities": {"alwaysMatch": {}, "firstMatch": [{}]}},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        self._session_id = data.get("sessionId") or data.get("value", {}).get("sessionId")
        if not self._session_id:
            raise RuntimeError(f"WDA did not return a session id: {data}")
        return self._session_id

    def window_size(self) -> Tuple[int, int]:
        """Logical screen size (points), used to scale screenshot pixels."""
        sid = self.session_id()
        resp = requests.get(self._url(f"/session/{sid}/window/size"), timeout=self.timeout)
        resp.raise_for_status()
        value = resp.json()["value"]
        return int(value["width"]), int(value["height"])

    def screenshot(self, save_path: Optional[str] = None):
        """Capture the screen. Returns a decoded BGR numpy image.

        Requires OpenCV/numpy only when actually called.
        """
        resp = requests.get(self._url("/screenshot"), timeout=self.timeout)
        resp.raise_for_status()
        png_bytes = base64.b64decode(resp.json()["value"])
        if save_path:
            with open(save_path, "wb") as fh:
                fh.write(png_bytes)
        import numpy as np  # local import: not needed for dry runs / tests
        import cv2

        arr = np.frombuffer(png_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    def perform_tap(self, point: Point) -> None:
        """Tap a single logical point (used for the Reshuffle button)."""
        x, y = point
        payload = {
            "actions": [
                {
                    "type": "pointer",
                    "id": "finger1",
                    "parameters": {"pointerType": "touch"},
                    "actions": [
                        {"type": "pointerMove", "duration": 0, "x": int(x), "y": int(y)},
                        {"type": "pointerDown", "button": 0},
                        {"type": "pause", "duration": 60},
                        {"type": "pointerUp", "button": 0},
                    ],
                }
            ]
        }
        if self.dry_run:
            print(f"[dry-run] tap: ({int(x)},{int(y)})")
            return
        sid = self.session_id()
        resp = requests.post(
            self._url(f"/session/{sid}/actions"), json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        requests.delete(self._url(f"/session/{sid}/actions"), timeout=self.timeout)

    def perform_drag(
        self,
        points: Sequence[Point],
        timing: DragTiming,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Replay ``points`` as one continuous finger drag."""
        payload = build_pointer_actions(points, timing, rng=rng)
        if self.dry_run:
            pretty = " -> ".join(f"({int(x)},{int(y)})" for x, y in points)
            print(f"[dry-run] drag: {pretty}")
            return
        sid = self.session_id()
        resp = requests.post(
            self._url(f"/session/{sid}/actions"), json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        # Release any lingering action state so the next drag starts clean.
        requests.delete(self._url(f"/session/{sid}/actions"), timeout=self.timeout)


def pixels_to_points(pixel_xy: Point, scale: float) -> Point:
    """Convert screenshot-pixel coords to WDA logical points.

    WDA screenshots are in device pixels but the actions API expects points, so
    divide by the Retina scale factor (screenshot_width / window_width).
    """
    x, y = pixel_xy
    return (x / scale, y / scale)
