from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .checkpoint import load_field
from .config import Config
from .network import genome_size, random_genomes
from .render import run_viewer
from .track import holdout_seeds


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Watch the evolved drivers race a track they have never seen.")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.npz"))
    p.add_argument("--cars", type=int, default=24)
    p.add_argument("--tracks", type=int, default=25)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--human", action="store_true", help="drive it yourself with the arrow keys")
    p.add_argument("--random", action="store_true", help="untrained drivers, generation zero")
    p.add_argument("--size", type=int, nargs=2, default=[1560, 900])
    p.add_argument("--camera", choices=["fit", "follow"], default="fit")
    p.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"], default=None,
                   help="defaults to the difficulty the model was validated on")
    p.add_argument("--screenshot", type=Path, default=None)
    p.add_argument("--screenshot-steps", type=int, default=300)
    args = p.parse_args(argv)

    if args.random:
        cfg = Config()
        genomes = random_genomes(np.random.default_rng(0), args.cars, cfg.layer_sizes)
        generation = 0
    else:
        if not args.checkpoint.exists():
            p.error(f"no checkpoint at {args.checkpoint} - run racing-train first")
        genomes, cfg, generation = load_field(args.checkpoint, 1 if args.human else args.cars)

    print(f"generation {generation}   {genomes.shape[0]} car(s)   "
          f"{genome_size(cfg.layer_sizes)} weights   {' -> '.join(map(str, cfg.layer_sizes))}")
    run_viewer(
        genomes, cfg, holdout_seeds(args.tracks, args.offset),
        size=tuple(args.size), human=args.human, camera=args.camera,
        difficulty=args.difficulty,
        screenshot=args.screenshot, screenshot_steps=args.screenshot_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
