"""Turn a screenshot of the WordLink board into a :class:`Board`.

This is the least portable part of the bot: the exact pixel geometry depends on
the device model and where the grid sits on screen. It is written to be
configurable rather than magic. The pipeline is:

    screenshot (numpy image)
      -> crop to the grid region (auto-detected or supplied)
      -> split into an R x C matrix of cells
      -> OCR each cell into a single letter
      -> Board

Heavy dependencies (OpenCV, pytesseract, Pillow) are imported lazily so the
solver and tests run in environments without them installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .solver import Board


def _require(module: str, pip_name: Optional[str] = None):
    try:
        return __import__(module)
    except ImportError as exc:  # pragma: no cover - depends on environment
        pip_name = pip_name or module
        raise ImportError(
            f"'{module}' is required for board OCR. Install it with "
            f"`pip install {pip_name}` (see requirements-wordlink.txt)."
        ) from exc


@dataclass
class GridRegion:
    """Pixel rectangle containing the letter grid, plus its dimensions.

    Coordinates are in pixels from the top-left of the screenshot.
    """

    left: int
    top: int
    width: int
    height: int
    rows: int
    cols: int

    def cell_box(self, r: int, c: int, inset: float = 0.18) -> Tuple[int, int, int, int]:
        """Return the (left, top, right, bottom) pixel box of one cell.

        ``inset`` trims a fraction off each side so we OCR the letter, not the
        tile border / rounded corners, which otherwise confuse the recogniser.
        """
        cw = self.width / self.cols
        ch = self.height / self.rows
        x0 = self.left + c * cw
        y0 = self.top + r * ch
        pad_x = cw * inset
        pad_y = ch * inset
        return (
            int(x0 + pad_x),
            int(y0 + pad_y),
            int(x0 + cw - pad_x),
            int(y0 + ch - pad_y),
        )

    def cell_center(self, r: int, c: int) -> Tuple[int, int]:
        cw = self.width / self.cols
        ch = self.height / self.rows
        return (
            int(self.left + (c + 0.5) * cw),
            int(self.top + (r + 0.5) * ch),
        )


def load_image(path: str):
    """Load an image file into a BGR numpy array."""
    cv2 = _require("cv2", "opencv-python")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def ocr_cell(cell_img, *, tesseract_config: str = "--psm 10") -> str:
    """OCR a single tile image into one uppercase letter.

    ``--psm 10`` tells Tesseract to treat the image as a single character, which
    is the right mode for one letter per tile.
    """
    cv2 = _require("cv2", "opencv-python")
    pytesseract = _require("pytesseract")

    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    # Tiles are light letters on a light background with shadows; Otsu threshold
    # then normalise polarity so the glyph is dark on white for Tesseract.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thresh.mean() < 127:
        thresh = cv2.bitwise_not(thresh)

    config = (
        f"{tesseract_config} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    text = pytesseract.image_to_string(thresh, config=config)
    letters = [c for c in text.upper() if c.isalpha()]
    return letters[0] if letters else "?"


def read_board(image, region: GridRegion, *, tesseract_config: str = "--psm 10") -> Board:
    """OCR the whole grid inside ``region`` into a :class:`Board`."""
    grid: List[List[str]] = []
    for r in range(region.rows):
        row: List[str] = []
        for c in range(region.cols):
            x0, y0, x1, y1 = region.cell_box(r, c)
            cell = image[y0:y1, x0:x1]
            row.append(ocr_cell(cell, tesseract_config=tesseract_config))
        grid.append(row)
    return Board(grid)


def save_image(path: str, image) -> None:
    """Write a numpy image to disk (thin wrapper so cv2 stays lazily imported)."""
    cv2 = _require("cv2", "opencv-python")
    cv2.imwrite(path, image)


def annotate_board(image, region: GridRegion, board: Optional[Board] = None):
    """Return a copy of ``image`` with the grid region + cells (and OCR'd
    letters, if a ``board`` is given) drawn on top, for visual calibration.

    Green rectangle = the region you passed; blue rectangles = each cell box;
    red letters = what the OCR read there. If the blue boxes don't sit on the
    tiles, adjust ``--region`` / ``--rows`` / ``--cols`` and try again.
    """
    cv2 = _require("cv2", "opencv-python")
    out = image.copy()
    cv2.rectangle(
        out,
        (region.left, region.top),
        (region.left + region.width, region.top + region.height),
        (0, 255, 0),
        3,
    )
    for r in range(region.rows):
        for c in range(region.cols):
            x0, y0, x1, y1 = region.cell_box(r, c, inset=0.0)
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 2)
            if board is not None:
                ch = board.grid[r][c]
                cv2.putText(
                    out, ch, (x0 + 6, y0 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3,
                )
    return out


def detect_grid_region(image, rows: int, cols: int) -> GridRegion:
    """Best-effort automatic detection of the square grid region.

    Finds the largest roughly-square contour on screen and assumes it is the
    board. This works for the common layout where the grid is the dominant
    bright rounded rectangle, but supplying an explicit :class:`GridRegion` is
    more reliable for a known device.
    """
    cv2 = _require("cv2", "opencv-python")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0.0
    h, w = gray.shape[:2]
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        aspect = cw / ch if ch else 0
        # Grid is large and near-square.
        if area > best_area and 0.7 < aspect < 1.3 and area > 0.1 * w * h:
            best = (x, y, cw, ch)
            best_area = area

    if best is None:
        raise RuntimeError(
            "Could not auto-detect the grid. Pass an explicit GridRegion."
        )
    x, y, cw, ch = best
    return GridRegion(left=x, top=y, width=cw, height=ch, rows=rows, cols=cols)
