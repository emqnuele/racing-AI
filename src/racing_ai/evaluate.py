from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .checkpoint import load_best
from .env import evaluate as evaluate_population
from .network import PopulationNetwork
from .track import make_track, holdout_seeds


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Race a trained genome on tracks it has never seen.")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.npz"))
    p.add_argument("--tracks", type=int, default=20)
    p.add_argument("--offset", type=int, default=0, help="shift into the held-out seed range")
    p.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"], default=None,
                   help="defaults to the difficulty the model was validated on")
    p.add_argument("--render", action="store_true", help="watch it drive in a window")
    p.add_argument("--screenshot", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.checkpoint.exists():
        p.error(f"no checkpoint at {args.checkpoint} - run the trainer first")

    genome, cfg, generation = load_best(args.checkpoint)
    level = args.difficulty or cfg.test_difficulty
    seeds = holdout_seeds(args.tracks, args.offset)
    print(f"genome from generation {generation}   topology {' -> '.join(map(str, cfg.layer_sizes))}")
    print(f"{len(seeds)} held-out {level} tracks "
          f"(seeds {seeds[0]}..{seeds[-1]}, never used in training)\n")

    net = PopulationNetwork(genome[None, :], cfg.layer_sizes)
    tracks = [make_track(s, cfg.track, level) for s in seeds]
    _, results = evaluate_population(net, tracks, cfg)

    print(f"{'seed':>8}  {'laps':>6}  {'outcome':>9}  {'steps':>6}  "
          f"{'avg':>6}  {'min':>6}  {'max':>6}")
    print("-" * 60)
    laps, finished = [], []
    for track, r in zip(tracks, results):
        outcome = "finished" if r.finished[0] else ("crashed" if r.crashed[0] else "timeout")
        print(f"{track.seed:>8}  {r.laps[0]:>6.3f}  {outcome:>9}  {r.steps[0]:>6}  "
              f"{r.mean_speed[0] * cfg.car.max_speed:>6.1f}  {r.min_speed[0]:>6.1f}  "
              f"{r.max_speed[0]:>6.1f}")
        laps.append(float(r.laps[0]))
        finished.append(bool(r.finished[0]))

    laps_arr = np.array(laps)
    print("-" * 60)
    print(f"completed {sum(finished)}/{len(finished)} tracks ({100 * np.mean(finished):.0f}%)   "
          f"mean {laps_arr.mean():.3f} laps   worst {laps_arr.min():.3f}   best {laps_arr.max():.3f}")
    swing = np.array([r.max_speed[0] - r.min_speed[0] for r in results])
    print(f"throttle modulation: speed swings {swing.mean():.0f} on average "
          f"(0 would mean it never lifts off)")

    if args.render or args.screenshot:
        from .render import run_viewer
        run_viewer(genome[None, :], cfg, seeds, difficulty=level,
                   screenshot=args.screenshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
