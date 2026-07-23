# WordLink Bot — automating Triumph's WordLink

A solver + iOS automation pipeline for **WordLink**, the Boggle / Word-Hunt-style
game in the Triumph app. You drag your finger across adjacent letter tiles
(including diagonals) to spell words; longer, higher-value words score more,
under a round timer. This tool reads the board off your iPhone screen, finds the
best words, and drags them out for you.

> **Heads up / fair use.** Triumph runs *cash* skill games and its terms
> prohibit bots and automation. Using this against real-money matches can get
> your account banned and may violate Triumph's Terms of Service (and, depending
> on jurisdiction, other rules). Treat this as an educational word-game solver
> and automation exercise — run it against practice boards, offline puzzles, or
> your own device, not to defraud opponents or a prize pool.

## How it works

```
iPhone screen ──screenshot──▶ vision (OCR) ──Board──▶ solver ──paths──▶ automation ──drag──▶ iPhone
     ▲                                                                                        │
     └────────────────────────────── WebDriverAgent (HTTP) ──────────────────────────────────┘
```

Four independent pieces (use as much or as little as you want):

| Module | Responsibility | Extra deps |
|---|---|---|
| `wordlink/trie.py`, `wordlink/solver.py`, `wordlink/dictionary.py` | Find every word on a grid, with tile paths, ranked by score | none (stdlib) |
| `wordlink/vision.py` | Crop a screenshot into cells and OCR each tile | `opencv-python`, `pytesseract`, `tesseract` binary |
| `wordlink/automation.py` | Screenshot + replay a word as one finger drag via WebDriverAgent | `requests` |
| `wordlink/bot.py` | Orchestrate capture → read → solve → play | all of the above |

Two word lists ship with the bot, both derived from the public-domain **ENABLE1**
lexicon (the standard list for this genre):

- `wordlink/data/common.txt` (~18k everyday words) — the **default**. Dropping the
  rare loanword / Scrabble-only tail (e.g. `SPORTIF`, `ROCOCO`) means far fewer
  attempts are wasted on words Triumph's dictionary rejects.
- `wordlink/data/enable1.txt` (~172k words) — maximal coverage. Pass it with
  `--dictionary wordlink/data/enable1.txt` when you want every possible word.

Swap in your own list with `--dictionary` at any time.

## Quick start (no device needed)

The solver is pure Python — no install step beyond having this repo.

```bash
# Solve a board you type in (rows separated by '/')
python3 -m wordlink solve --board "EITP/LLFO/IISE/ETLL" --top 20

# Solve a board from a file (one row per line)
python3 -m wordlink solve --board-file board.txt

# See the full solve + a dry-run of the drag gestures for the example board
python3 -m wordlink demo
```

Each result shows the word, its score, and the tile path as `(row,col)` pairs:

```
  1. FELSITE   score=17   path=(1,2) (2,3) (3,2) (2,2) (2,1) (3,1) (3,0)
```

## Playing on a device

Automation talks to **WebDriverAgent (WDA)**, the same on-device server Appium
uses. You need a Mac (or Windows with the right tooling), Xcode, and your iPhone.

1. **Build & run WebDriverAgent** on the iPhone (via Xcode or `xcodebuild`), then
   forward its port to your machine:
   ```bash
   iproxy 8100 8100          # from libimobiledevice; or use `xcodebuild test` output
   ```
   Confirm it's up: `curl http://localhost:8100/status` returns JSON.
2. **Open WordLink** on the phone so the board is on screen.
3. **Install optional deps** for OCR/automation:
   ```bash
   pip install -r requirements-wordlink.txt
   # plus the tesseract binary: brew install tesseract  (macOS)
   ```
4. **Dry run first** (reads the board, plans gestures, sends nothing):
   ```bash
   python3 -m wordlink play --wda http://localhost:8100 --rows 4 --cols 4 --dry-run
   ```
5. **Play for real** once the board reads correctly:
   ```bash
   python3 -m wordlink play --rows 4 --cols 4 --round-seconds 80 --max-words 40
   ```

### Getting the grid region right

OCR needs to know where the grid is. Auto-detection (`detect_grid_region`) finds
the largest near-square shape, which works for the typical layout, but it is more
reliable to pass the exact pixel box:

```bash
python3 -m wordlink play --region "LEFT,TOP,WIDTH,HEIGHT" --rows 4 --cols 4
```

Take a screenshot (`GET /screenshot`), open it in any image editor, and read off
the pixel rectangle that bounds the tiles. Note WDA screenshots are in *device
pixels* while gestures use *logical points*; the bot computes the scale factor
(`screenshot_width / window_size_width`) automatically.

### Tuning speed vs. reliability

`--press-hold-ms`, `--move-ms` (per tile), and `--between-words-ms` control the
drag. Faster clears more words within the timer but risks mis-registered swipes.
Start slow, then speed up until words stop registering, then back off.

## Notes on the algorithm

- **Search.** Depth-first from every tile, pruned by a trie: a partial path is
  abandoned the instant it stops being a prefix of any real word. A 6×6 board
  solves in a couple of milliseconds.
- **Adjacency.** All 8 neighbours (king moves); each tile used at most once per
  word. Multi-character tiles (e.g. a `Qu` tile) are supported.
- **Scoring.** Scrabble letter values plus a per-letter length bonus. The exact
  constants Triumph uses aren't published, but any function that increases with
  length and letter value ranks words in the same order, so "best words first"
  holds regardless.

## Tests

```bash
pip install pytest
python3 -m pytest tests/test_wordlink.py -q
```

Covers the trie, board parsing, adjacency/no-reuse rules, diagonal paths,
scoring, gesture payload construction, and an integration check against the
bundled dictionary.
