"""Command-line entry point for the WordLink bot.

Examples
--------
Solve a board typed on the command line (each row separated by ``/``)::

    python -m wordlink solve --board "EITP/LLFO/IISE/ETLL"

Solve a board from a file (one row per line)::

    python -m wordlink solve --board-file board.txt --top 20

See the gesture plan for a board without a device (dry run)::

    python -m wordlink demo

Play on a connected iPhone running WebDriverAgent::

    python -m wordlink play --wda http://localhost:8100 --rows 4 --cols 4
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .automation import DragTiming, WDAClient, pixels_to_points
from .bot import BotConfig, WordLinkBot
from .dictionary import load_trie
from .solver import Board, Solver


def _read_board(args) -> Board:
    if args.board:
        rows = args.board.replace("|", "/").split("/")
        return Board.from_lines(rows)
    if args.board_file:
        with open(args.board_file, "r", encoding="utf-8") as fh:
            return Board.from_string(fh.read())
    data = sys.stdin.read()
    if not data.strip():
        raise SystemExit("No board provided. Use --board, --board-file, or stdin.")
    return Board.from_string(data)


def _solve_board(board: Board, min_length: int, dictionary: Optional[str]) -> List:
    trie = load_trie(
        dictionary,
        min_length=min_length,
        max_length=board.rows * board.cols,
        allowed_letters=board.distinct_letters,
    )
    solver = Solver(trie, min_length=min_length)
    return solver.solve(board)


def _format_path(path) -> str:
    return " ".join(f"({r},{c})" for r, c in path)


def cmd_solve(args) -> int:
    board = _read_board(args)
    print("Board:")
    print(board)
    print()
    solutions = _solve_board(board, args.min_length, args.dictionary)
    top = solutions[: args.top] if args.top else solutions
    print(f"Found {len(solutions)} words (showing {len(top)}):\n")
    for i, sol in enumerate(top, 1):
        print(f"{i:>3}. {sol.word:<15} score={sol.score:<4} path={_format_path(sol.path)}")
    return 0


def cmd_demo(args) -> int:
    # Board transcribed from the example screenshot in the task.
    board = Board.from_lines(["EITP", "LLFO", "IISE", "ETLL"])
    print("Demo board:")
    print(board)
    print()
    solutions = _solve_board(board, args.min_length, args.dictionary)
    top = solutions[: args.top]
    print(f"Found {len(solutions)} words. Top {len(top)} by score:\n")
    for i, sol in enumerate(top, 1):
        print(f"{i:>3}. {sol.word:<15} score={sol.score:<4} path={_format_path(sol.path)}")

    print("\nGesture plan (dry run, using synthetic tile coordinates):")
    from .vision import GridRegion

    region = GridRegion(left=100, top=400, width=800, height=800, rows=4, cols=4)
    client = WDAClient(dry_run=True)
    timing = DragTiming()
    for sol in top[: args.play]:
        points = [pixels_to_points(region.cell_center(r, c), scale=3.0) for r, c in sol.path]
        print(f"  {sol.word}:")
        client.perform_drag(points, timing)
    return 0


def cmd_calibrate(args) -> int:
    """Screenshot the device and preview how the board will be read.

    Handles a new device/screen size: it computes the pixel<->point scale for
    you, reads the board from the given (or auto-detected) region, saves an
    annotated overlay so you can eyeball the fit, and prints the exact ``play``
    flags to copy once it looks right.
    """
    from .vision import GridRegion, annotate_board, detect_grid_region, save_image

    client = WDAClient(base_url=args.wda)
    win_w, win_h = client.window_size()
    image = client.screenshot(save_path=args.out)
    sh, sw = image.shape[:2]
    scale = sw / win_w
    print(f"Device logical size : {win_w} x {win_h} pts")
    print(f"Screenshot size     : {sw} x {sh} px")
    print(f"Scale factor        : {scale:.3f} px/pt (handled automatically)")
    print(f"Saved screenshot    : {args.out}")

    if args.region:
        left, top, width, height = (int(v) for v in args.region.split(","))
        region = GridRegion(left, top, width, height, args.rows, args.cols)
    else:
        try:
            region = detect_grid_region(image, args.rows, args.cols)
            print(
                f"Auto-detected region: "
                f"{region.left},{region.top},{region.width},{region.height}"
            )
        except Exception as exc:
            print(f"\nCould not auto-detect the grid: {exc}")
            print(f"Open {args.out}, read off the pixel box around the tiles, then re-run:")
            print(
                f"  python3 -m wordlink calibrate --wda {args.wda} "
                f"--rows {args.rows} --cols {args.cols} --region LEFT,TOP,WIDTH,HEIGHT"
            )
            return 1

    board = read_board_from_image(image, region)
    print("\nBoard read from that region:")
    print(board)

    annotated = annotate_board(image, region, board)
    save_image(args.annotated, annotated)
    print(f"\nSaved annotated preview: {args.annotated}")
    print("Open it and confirm the blue cell boxes sit on the tiles and the")
    print("red letters are correct. If not, tweak --region / --rows / --cols.")
    print("\nWhen it looks right, play with:")
    print(
        f"  python3 -m wordlink play --wda {args.wda} "
        f"--rows {args.rows} --cols {args.cols} "
        f"--region {region.left},{region.top},{region.width},{region.height}"
    )
    return 0


def read_board_from_image(image, region):
    from .vision import read_board

    return read_board(image, region)


def cmd_play(args) -> int:
    region = None
    if args.region:
        left, top, width, height = (int(v) for v in args.region.split(","))
        from .vision import GridRegion

        region = GridRegion(left, top, width, height, args.rows, args.cols)

    reshuffle_xy = None
    if args.reshuffle_xy:
        rx, ry = (float(v) for v in args.reshuffle_xy.split(","))
        reshuffle_xy = (rx, ry)

    config = BotConfig(
        rows=args.rows,
        cols=args.cols,
        min_length=args.min_length,
        round_seconds=args.round_seconds,
        dictionary_path=args.dictionary,
        region=region,
        quick_check=args.quick_check_ms / 1000.0,
        settle_after_word=args.settle_ms / 1000.0,
        change_threshold=args.change_threshold,
        reshuffle_xy=reshuffle_xy,
        verbose=args.verbose,
        screenshot_quality=args.screenshot_quality,
        learn_dir=args.learn_dir,
        top_choice_n=args.top_choice_n,
        skip_prob=args.skip_prob,
        max_words_per_minute=(None if args.max_wpm <= 0 else args.max_wpm),
        max_words=args.max_words,
        think_prob=args.think_prob,
        idle_prob=args.idle_prob,
        fatigue_per_min=args.fatigue_per_min,
        false_start_prob=args.false_start_prob,
        timing=DragTiming(
            press_hold_ms=args.press_hold_ms,
            move_ms_per_tile=args.move_ms,
            tile_dwell_ms=args.tile_dwell_ms,
            between_words_ms=args.between_words_ms,
            jitter=args.jitter,
            distribution=args.distribution,
            curve=args.curve,
            points_per_segment=args.points_per_segment,
            overshoot_prob=args.overshoot_prob,
            position_jitter=args.position_jitter,
        ),
    )
    client = WDAClient(base_url=args.wda, dry_run=args.dry_run)
    bot = WordLinkBot(config, client=client)
    bot.run(rounds=args.rounds)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wordlink",
        description="Solve and auto-play WordLink-style letter grids.",
    )
    parser.add_argument(
        "--dictionary",
        help="Path to a custom word list. Default is the bundled common-word list "
        "(~18k everyday words) for high acceptance; pass the full ENABLE1 list "
        "(wordlink/data/enable1.txt) or your own for maximum coverage.",
    )
    parser.add_argument(
        "--min-length", type=int, default=3, help="Minimum word length (default 3)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_solve = sub.add_parser("solve", help="Solve a board and print words + paths.")
    p_solve.add_argument("--board", help="Board with rows separated by '/' e.g. EITP/LLFO/IISE/ETLL")
    p_solve.add_argument("--board-file", help="File with one board row per line.")
    p_solve.add_argument("--top", type=int, default=25, help="Show only the top N words.")
    p_solve.set_defaults(func=cmd_solve)

    p_demo = sub.add_parser("demo", help="Run the solver + a dry-run gesture plan.")
    p_demo.add_argument("--top", type=int, default=15, help="Words to display.")
    p_demo.add_argument("--play", type=int, default=5, help="Words to show gestures for.")
    p_demo.set_defaults(func=cmd_demo)

    p_cal = sub.add_parser(
        "calibrate",
        help="Screenshot the device and preview the grid region + OCR (for a new device/screen).",
    )
    p_cal.add_argument("--wda", default="http://localhost:8100", help="WDA base URL.")
    p_cal.add_argument("--rows", type=int, default=4)
    p_cal.add_argument("--cols", type=int, default=4)
    p_cal.add_argument("--region", help="Grid pixel box 'left,top,width,height'. Omit to auto-detect.")
    p_cal.add_argument("--out", default="calibration.png", help="Where to save the raw screenshot.")
    p_cal.add_argument("--annotated", default="calibration_annotated.png",
                       help="Where to save the annotated overlay preview.")
    p_cal.set_defaults(func=cmd_calibrate)

    p_play = sub.add_parser("play", help="Play on a device via WebDriverAgent.")
    p_play.add_argument("--wda", default="http://localhost:8100", help="WDA base URL.")
    p_play.add_argument("--rows", type=int, default=4)
    p_play.add_argument("--cols", type=int, default=4)
    p_play.add_argument("--region", help="Grid pixel box 'left,top,width,height'. Omit to auto-detect.")
    p_play.add_argument("--rounds", type=int, default=1, help="Max board reshuffles (needs --reshuffle-xy).")
    p_play.add_argument("--round-seconds", type=float, default=None, help="Stop after N seconds (the game timer).")
    p_play.add_argument("--reshuffle-xy", help="Screen pixel 'x,y' of the Reshuffle button, tapped when out of words.")
    p_play.add_argument("--quick-check-ms", type=int, default=90, help="Delay before checking if a word was accepted.")
    p_play.add_argument("--settle-ms", type=int, default=240, help="Extra wait on an accepted word for tiles to refill.")
    p_play.add_argument("--change-threshold", type=float, default=8.0, help="Pixel-change sensitivity for accept detection.")
    p_play.add_argument("--press-hold-ms", type=int, default=45, help="Finger press time before moving.")
    p_play.add_argument("--move-ms", type=int, default=65, help="Drag time per tile (lower = faster).")
    p_play.add_argument("--tile-dwell-ms", type=int, default=50, help="Pause on each tile so it registers (raise if it misses tiles).")
    p_play.add_argument("--between-words-ms", type=int, default=20, help="Pause between words.")

    human = p_play.add_argument_group(
        "humanization",
        "Make the bot look less like a bot. Set any knob to 0 to disable it.",
    )
    human.add_argument("--jitter", type=float, default=0.4,
                       help="Randomize each keystroke's timing by +/- this fraction "
                            "(0.4 = +/-40%%) so swipes aren't uniform.")
    human.add_argument("--distribution", choices=["gaussian", "uniform"], default="gaussian",
                       help="Timing jitter shape: gaussian clusters near the base value "
                            "with rare outliers (more human); uniform is a flat band.")
    human.add_argument("--curve", type=float, default=0.12,
                       help="Bow each swipe segment sideways by up to this fraction of its "
                            "length so the trace is a hand-drawn arc, not a straight line.")
    human.add_argument("--points-per-segment", type=int, default=4,
                       help="Intermediate points sampled along each curved segment.")
    human.add_argument("--overshoot-prob", type=float, default=0.12,
                       help="Chance to slide just past a tile and correct back.")
    human.add_argument("--position-jitter", type=float, default=0.28,
                       help="Aim at a random spot within this fraction of each tile "
                            "instead of the exact centre.")
    human.add_argument("--top-choice-n", type=int, default=4,
                       help="Choose randomly among the top N words (weighted by score) "
                            "instead of always the single best. 1 = strict best.")
    human.add_argument("--skip-prob", type=float, default=0.06,
                       help="Per-word chance to skip a findable word (humans miss words).")
    human.add_argument("--max-wpm", type=float, default=40.0,
                       help="Cap accepted words per minute (0 = unlimited). The biggest "
                            "anti-detection lever: superhuman speed is the top tell.")
    human.add_argument("--max-words", type=int, default=None,
                       help="Stop after ~this many words per game (jittered +/-15%%). "
                            "Omit to play until the timer/words run out.")
    human.add_argument("--think-prob", type=float, default=0.12,
                       help="Chance of a longer 'thinking' pause between words.")
    human.add_argument("--idle-prob", type=float, default=0.03,
                       help="Chance of a rare multi-second idle break between words.")
    human.add_argument("--fatigue-per-min", type=float, default=0.1,
                       help="Gradually slow down: delays grow by this fraction per minute.")
    human.add_argument("--false-start-prob", type=float, default=0.05,
                       help="Chance of a small hesitation gesture before a word.")
    p_play.add_argument("--screenshot-quality", type=int, default=1, choices=[0, 1, 2],
                        help="WDA screenshot compression: 0=PNG (best/slowest), 1=JPEG, 2=low JPEG (fastest).")
    p_play.add_argument("--dry-run", action="store_true", help="Log gestures instead of sending them.")
    p_play.add_argument("--verbose", action="store_true", help="Also print rejected words.")
    p_play.add_argument("--learn-dir", help="Persist learned accepted.txt / rejected.txt here to map Triumph's dictionary and skip known-bad words across runs.")
    p_play.set_defaults(func=cmd_play)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
