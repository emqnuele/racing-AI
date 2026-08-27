from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import (
    Config, EpisodeConfig, EvolutionConfig, NetworkConfig, SensorConfig, TrackConfig,
)
from .trainer import Trainer


def build_config(args: argparse.Namespace) -> Config:
    return replace(
        Config(),
        seed=args.seed,
        tracks_per_generation=args.tracks,
        validation_every=args.validate_every,
        validation_tracks=args.validation_tracks,
        fitness_aggregation=args.aggregation,
        difficulty=args.difficulty,
        test_difficulty=args.test_difficulty,
        evolution=replace(
            EvolutionConfig(),
            population=args.population,
            elite=args.elite,
            survivors=args.survivors,
            mutation_rate=args.mutation_rate,
            mutation_sigma=args.mutation_sigma,
            crossover=args.crossover,
        ),
        network=NetworkConfig(hidden=tuple(args.hidden)),
        sensors=replace(SensorConfig(), n_rays=args.rays),
        episode=replace(EpisodeConfig(), max_steps=args.max_steps, target_laps=args.laps),
        track=replace(TrackConfig(), width_override=args.track_width),
    )


HEADER = (
    f"{'gen':>5}  {'best':>7}  {'mean':>7}  {'median':>7}  {'worst':>7}  "
    f"{'laps':>5}  {'fin%':>5}  {'crash%':>6}  {'sigma':>5}  {'val_laps':>8}  {'val_fin%':>8}  {'sec':>5}"
)


def format_row(row: dict[str, Any]) -> str:
    val_laps = "-" if row.get("val_laps") is None else f"{row['val_laps']:.3f}"
    val_fin = "-" if row.get("val_finish_rate") is None else f"{100 * row['val_finish_rate']:.0f}%"
    return (
        f"{row['generation']:>5}  {row['best']:>7.3f}  {row['mean']:>7.3f}  "
        f"{row['median']:>7.3f}  {row['worst']:>7.3f}  {row['best_laps']:>5.2f}  "
        f"{100 * row['finish_rate']:>4.0f}%  {100 * row['crash_rate']:>5.0f}%  "
        f"{row['sigma']:>5.3f}  {val_laps:>8}  {val_fin:>8}  {row['seconds']:>5.1f}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="racing train",
        description="Evolve a neural network that drives a race car.")
    p.add_argument("--live", action="store_true",
                   help="open a window and watch the population evolve, generation by generation")
    p.add_argument("--generations", type=int, default=300)
    p.add_argument("--population", type=int, default=100)
    p.add_argument("--elite", type=int, default=5)
    p.add_argument("--survivors", type=int, default=25)
    p.add_argument("--tracks", type=int, default=6, help="fresh tracks evaluated per generation")
    p.add_argument("--rays", type=int, default=7)
    p.add_argument("--hidden", type=int, nargs=3, default=[16, 12, 8], help="three hidden layers")
    p.add_argument("--mutation-rate", type=float, default=0.12)
    p.add_argument("--mutation-sigma", type=float, default=0.25)
    p.add_argument("--crossover", choices=["uniform", "one_point", "blend"], default="uniform")
    p.add_argument("--aggregation", choices=["mean", "min", "mean_min"], default="mean_min")
    p.add_argument("--max-steps", type=int, default=1400)
    p.add_argument("--laps", type=float, default=1.0)
    p.add_argument("--track-width", type=float, default=0.0,
                   help="override the width the difficulty level would pick")
    p.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"],
                   default="mixed", help="track difficulty used for training")
    p.add_argument("--test-difficulty", choices=["easy", "medium", "hard", "mixed"],
                   default="hard", help="difficulty of the held-out validation tracks")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--validate-every", type=int, default=10)
    p.add_argument("--validation-tracks", type=int, default=12)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument("--resume", action="store_true", help="continue from checkpoints/latest.npz")
    p.add_argument("--camera", choices=["fit", "follow"], default="fit")
    p.add_argument("--size", type=int, nargs=2, default=[1560, 900])
    p.add_argument("--no-plot", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    latest = args.checkpoint_dir / "latest.npz"
    if args.resume:
        if not latest.exists():
            p.error(f"nothing to resume: {latest} does not exist")
        trainer = Trainer.resume(args.checkpoint_dir, args.save_every)
        print(f"resumed from {latest} at generation {trainer.generation}")
    else:
        trainer = Trainer.start(build_config(args), args.checkpoint_dir, args.save_every)

    cfg = trainer.cfg
    print(f"genome {trainer.evo.genome_size} weights   "
          f"topology {' -> '.join(map(str, cfg.layer_sizes))}")
    print(f"population {cfg.evolution.population}   tracks/gen {cfg.tracks_per_generation}   "
          f"held-out validation tracks {cfg.validation_tracks}")

    if args.live:
        from .live import run_live_training
        print("opening the training window - close it or press ESC to stop and save")
        run_live_training(trainer, generations=args.generations,
                          size=tuple(args.size), camera=args.camera)
    else:
        print(HEADER)
        print("-" * len(HEADER))
        target = trainer.generation + args.generations
        try:
            while trainer.generation < target:
                print(format_row(trainer.run_generation()))
        except KeyboardInterrupt:
            print("\ninterrupted, saving state")
        trainer.finish(plot=not args.no_plot)

    laps, finish = trainer.validate(trainer.evo.best_genome)
    print(f"\nbest genome: fitness {trainer.evo.best_fitness:.3f}   "
          f"unseen tracks -> {laps:.3f} laps, {100 * finish:.0f}% completed")
    print(f"checkpoints in {args.checkpoint_dir}/   watch it drive with: racing watch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
