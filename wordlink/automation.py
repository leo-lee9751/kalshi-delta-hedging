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
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import requests

Point = Tuple[float, float]


@dataclass
class DragTiming:
    """Timing knobs (milliseconds) for how a word is dragged.

    Slower is more reliable; faster clears more words within the round timer.
    ``tile_dwell_ms`` is the important one for accuracy: pausing on each tile
    guarantees the game registers it instead of skipping a fast fly-over.
    """

    press_hold_ms: int = 40
    move_ms_per_tile: int = 55
    tile_dwell_ms: int = 35
    settle_ms: int = 40
    between_words_ms: int = 120


def build_pointer_actions(points: Sequence[Point], timing: DragTiming) -> dict:
    """Construct the W3C ``/actions`` payload for a continuous drag.

    The finger presses down on the first tile, moves through each subsequent
    tile, then lifts. This is what makes the game register a single word rather
    than a series of taps.
    """
    if len(points) < 2:
        raise ValueError("A word path needs at least two tiles to drag between")

    x0, y0 = points[0]
    actions: List[dict] = [
        {"type": "pointerMove", "duration": 0, "x": int(x0), "y": int(y0)},
        {"type": "pointerDown", "button": 0},
        {"type": "pause", "duration": timing.press_hold_ms},
    ]
    for x, y in points[1:]:
        actions.append(
            {"type": "pointerMove", "duration": timing.move_ms_per_tile, "x": int(x), "y": int(y)}
        )
        # Dwell on each tile so the game registers it before moving on.
        if timing.tile_dwell_ms:
            actions.append({"type": "pause", "duration": timing.tile_dwell_ms})
    actions.append({"type": "pause", "duration": timing.settle_ms})
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

    def perform_drag(self, points: Sequence[Point], timing: DragTiming) -> None:
        """Replay ``points`` as one continuous finger drag."""
        payload = build_pointer_actions(points, timing)
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
