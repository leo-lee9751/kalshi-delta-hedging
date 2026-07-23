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

### Getting the grid region right (and calibrating a new device)

OCR needs to know where the grid is. Different devices (iPhone vs iPad, or two
different iPads) put the board at different pixel coordinates and sometimes use a
different number of tiles, so you calibrate per device. Resolution itself is
handled for you — WDA screenshots are in *device pixels* while gestures use
*logical points*, and the bot computes the scale factor
(`screenshot_width / window_size_width`) automatically.

The easy way is the **`calibrate`** command. With the game on screen and WDA
running, it screenshots the device, reads the board, and saves an annotated
preview plus the exact `play` flags to copy:

```bash
# Let it try to auto-detect the grid first
python3 -m wordlink calibrate --wda http://localhost:8100 --rows 4 --cols 4

# ...or pass a region you want to check
python3 -m wordlink calibrate --rows 4 --cols 4 --region "LEFT,TOP,WIDTH,HEIGHT"
```

It writes `calibration.png` (raw screenshot) and `calibration_annotated.png`
(the region as a green box, each cell as a blue box, and the OCR'd letters in
red). Open the annotated image: if the blue boxes sit on the tiles and the red
letters are right, copy the printed `play` command. If not, tweak `--region`
(and `--rows`/`--cols` — **count the tiles on your device**, it may not be 4×4)
and re-run until it lines up.

To find a region by hand instead: open `calibration.png` in any image editor and
read off the pixel rectangle bounding the tiles, then pass it as
`--region "LEFT,TOP,WIDTH,HEIGHT"`.

### Tuning speed vs. reliability

`--press-hold-ms`, `--move-ms` (per tile), and `--between-words-ms` control the
drag. Faster clears more words within the timer but risks mis-registered swipes.
Start slow, then speed up until words stop registering, then back off.

### Looking less like a bot (humanization)

Perfectly uniform timing, dead-straight drags to exact pixel centres, always
playing the single best word, and clearing every word at superhuman speed are
all obvious automation tells. The `play` command applies a set of humanization
behaviours by default (all under the `humanization` group in `--help`, and each
can be turned off by setting it to `0`):

**Motion** — the finger no longer moves like a machine:

- `--position-jitter` (0.28) — aim at a random spot inside each tile, not the
  exact centre.
- `--curve` (0.12) + `--points-per-segment` (4) — bow each tile-to-tile segment
  into a slight hand-drawn arc instead of a straight polygon edge.
- `--overshoot-prob` (0.12) — occasionally slide just past a tile and correct.
- `--false-start-prob` (0.05) — a tiny hesitation gesture before some words.

**Timing** — delays vary the way a person's do:

- `--jitter` (0.4) — randomize each hold/move/dwell by +/- this fraction.
- `--distribution` (gaussian) — cluster durations near the base value with rare
  outliers, rather than a flat `uniform` band.
- `--think-prob` (0.12) — occasional longer "searching the board" pauses.
- `--idle-prob` (0.03) — rare multi-second breaks.
- `--fatigue-per-min` (0.1) — gradually slow down over a session.

**Decisions & pace** — *what* and *how much* you play (the strongest signals):

- `--max-wpm` (40) — cap accepted words per minute. **This is the single biggest
  anti-detection lever**; finding every word at machine speed is the top tell.
- `--max-words` — stop after roughly this many words per game (jittered +/-15%)
  so you don't exhaustively clear the board or play identical-length games.
- `--top-choice-n` (4) — play a weighted-random pick among the top N words, not
  always the strict maximum, so the order isn't deterministically optimal.
- `--skip-prob` (0.06) — randomly skip some findable words (humans miss plenty,
  especially long/rare ones).

Turn the whole lot off for maximum speed with, e.g., `--jitter 0 --curve 0
--position-jitter 0 --overshoot-prob 0 --skip-prob 0 --top-choice-n 1 --max-wpm 0`.

> None of this defeats server-side detection that relies on signals you can't
> touch from here (device attestation, input-event provenance, aggregate
> behavioural models). Treat it as making the *client-side* behaviour plausible,
> and see the fair-use note above.

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
