from __future__ import annotations

import sys

from . import __version__

USAGE = """racing-AI - a neural network that learns to drive by natural selection

  racing train --live         evolve the drivers in a window and watch it happen
  racing train                same thing, headless and much faster
  racing train --resume       carry on from the last checkpoint
  racing watch                watch the trained drivers race an unseen track
  racing play                 drive it yourself with the arrow keys
  racing demo                 watch generation zero, untrained, crash everywhere
  racing eval                 score the best genome on held-out tracks

every command takes --help, for example:

  racing train --help
  racing train --live --population 150 --tracks 8
  racing watch --cars 40 --camera follow
"""


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if not args or args[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    if args[0] in ("-V", "--version"):
        print(f"racing-AI {__version__}")
        return 0

    command, rest = args[0], args[1:]

    if command == "train":
        from .train import main as run
        return run(rest)
    if command == "watch":
        from .watch import main as run
        return run(rest)
    if command == "play":
        from .watch import main as run
        return run([*rest, "--human"])
    if command == "demo":
        from .watch import main as run
        return run([*rest, "--random"])
    if command == "eval":
        from .evaluate import main as run
        return run(rest)

    print(f"unknown command: {command}\n", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
