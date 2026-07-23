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
        settle_after_word=args.settle_ms / 1000.0,
        reshuffle_xy=reshuffle_xy,
        timing=DragTiming(
            press_hold_ms=args.press_hold_ms,
            move_ms_per_tile=args.move_ms,
            between_words_ms=args.between_words_ms,
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
        "(~45k words) for high acceptance; pass the full ENABLE1 list or your own "
        "for maximum coverage.",
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

    p_play = sub.add_parser("play", help="Play on a device via WebDriverAgent.")
    p_play.add_argument("--wda", default="http://localhost:8100", help="WDA base URL.")
    p_play.add_argument("--rows", type=int, default=4)
    p_play.add_argument("--cols", type=int, default=4)
    p_play.add_argument("--region", help="Grid pixel box 'left,top,width,height'. Omit to auto-detect.")
    p_play.add_argument("--rounds", type=int, default=1, help="Max board reshuffles (needs --reshuffle-xy).")
    p_play.add_argument("--round-seconds", type=float, default=None, help="Stop after N seconds (the game timer).")
    p_play.add_argument("--reshuffle-xy", help="Screen pixel 'x,y' of the Reshuffle button, tapped when out of words.")
    p_play.add_argument("--settle-ms", type=int, default=450, help="Wait after each word for tiles to refill.")
    p_play.add_argument("--press-hold-ms", type=int, default=25, help="Finger press time before moving.")
    p_play.add_argument("--move-ms", type=int, default=35, help="Drag time per tile (lower = faster).")
    p_play.add_argument("--between-words-ms", type=int, default=60, help="Pause between words.")
    p_play.add_argument("--dry-run", action="store_true", help="Log gestures instead of sending them.")
    p_play.set_defaults(func=cmd_play)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
