from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import load, save, stats_to_row, write_history_csv
from .config import Config
from .env import EpisodeResult, combine_fitness, evaluate
from .ga import Evolution
from .network import PopulationNetwork
from .track import Track, holdout_seeds, make_track, training_seeds


def validation_score(finish: float, time_left: float, n_tracks: int) -> float:
    # ordine lessicografico: il tempo fa da spareggio e non può mai valere
    # quanto un tracciato completato in più
    return finish + time_left / (n_tracks + 1)


class Trainer:
    def __init__(self, evo: Evolution, history: list[dict[str, Any]],
                 ckpt_dir: Path, save_every: int = 50):
        self.evo = evo
        self.cfg = evo.cfg
        self.history = history
        self.ckpt_dir = ckpt_dir
        self.save_every = save_every
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._started = time.perf_counter()

    @classmethod
    def start(cls, cfg: Config, ckpt_dir: Path, save_every: int = 50) -> "Trainer":
        return cls(Evolution(cfg), [], ckpt_dir, save_every)

    @classmethod
    def resume(cls, ckpt_dir: Path, save_every: int = 50) -> "Trainer":
        evo, history = load(ckpt_dir / "latest.npz")
        return cls(evo, history, ckpt_dir, save_every)

    @property
    def generation(self) -> int:
        return self.evo.generation

    def tracks(self) -> list[Track]:
        rng = np.random.default_rng([self.cfg.seed, self.evo.generation])
        return [make_track(s, self.cfg.track, self.cfg.difficulty)
                for s in training_seeds(rng, self.cfg.tracks_per_generation)]

    def network(self) -> PopulationNetwork:
        return PopulationNetwork(self.evo.genomes, self.cfg.layer_sizes)

    def combine(self, results: list[EpisodeResult]) -> np.ndarray:
        return combine_fitness(results, self.cfg.fitness_aggregation)

    def record(self, fitness: np.ndarray, results: list[EpisodeResult],
               seconds: float) -> dict[str, Any]:
        laps = np.mean([r.laps for r in results], axis=0)
        finished = np.mean([r.finished for r in results], axis=0)
        crashed = np.mean([r.crashed for r in results], axis=0)
        row = stats_to_row(
            self.evo.stats(fitness, laps, self.lap_steps(results), finished, crashed))

        every = self.cfg.validation_every
        if every > 0 and self.evo.generation % every == 0:
            champion = self.evo.genomes[int(np.argmax(fitness))]
            row["val_laps"], row["val_finish_rate"], score = self.validate_full(champion)
            self.evo.remember_best(champion, score)
        elif every <= 0:
            # senza validazione resta solo la fitness di generazione, sorteggio compreso
            self.evo.remember_best(self.evo.genomes[int(np.argmax(fitness))],
                                   float(np.max(fitness)))

        row["seconds"] = seconds
        self.history.append(row)
        return row

    @staticmethod
    def lap_steps(results: list[EpisodeResult]) -> np.ndarray:
        # i giri non finiti non hanno un tempo, quindi mediano solo quelli chiusi
        done = np.stack([r.finished for r in results])
        steps = np.stack([r.steps for r in results]).astype(float)
        count = done.sum(axis=0)
        return np.where(count > 0, (steps * done).sum(axis=0) / np.maximum(count, 1), np.nan)

    def advance(self, fitness: np.ndarray) -> None:
        self.evo.evolve(fitness)
        if self.save_every > 0 and self.evo.generation % self.save_every == 0:
            save(self.ckpt_dir / f"gen_{self.evo.generation:05d}.npz", self.evo, self.history)
        save(self.ckpt_dir / "latest.npz", self.evo, self.history)

    def run_generation(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        fitness, results = evaluate(self.network(), self.tracks(), self.cfg)
        row = self.record(fitness, results, time.perf_counter() - t0)
        self.advance(fitness)
        return row

    def validate(self, genome: np.ndarray) -> tuple[float, float]:
        laps, finish, _ = self.validate_full(genome)
        return laps, finish

    def validate_full(self, genome: np.ndarray) -> tuple[float, float, float]:
        net = PopulationNetwork(genome[None, :], self.cfg.layer_sizes)
        tracks = [make_track(s, self.cfg.track, self.cfg.test_difficulty)
                  for s in holdout_seeds(self.cfg.validation_tracks)]
        _, results = evaluate(net, tracks, self.cfg)
        laps = float(np.mean([r.laps[0] for r in results]))
        done = np.array([bool(r.finished[0]) for r in results])
        finish = float(done.mean())
        steps = np.array([int(r.steps[0]) for r in results], dtype=float)
        time_left = float((1.0 - steps[done] / self.cfg.episode.max_steps).mean()) if done.any() else 0.0
        return laps, finish, validation_score(finish, time_left, self.cfg.validation_tracks)

    def finish(self, plot: bool = True) -> tuple[float, float]:
        if self.cfg.validation_every > 0 and self.history:
            # l'ultima generazione non è quasi mai una di quelle validate
            champion = self.evo.genomes[0]
            self.evo.remember_best(champion, self.validate_full(champion)[2])
        save(self.ckpt_dir / "latest.npz", self.evo, self.history)
        write_history_csv(self.ckpt_dir / "history.csv", self.history)
        if plot:
            plot_history(self.history, self.ckpt_dir / "training_curve.png",
                         self.cfg.episode.dt)
        return self.validate(self.evo.best_genome)


def plot_history(history: list[dict[str, Any]], path: Path, dt: float = 0.05) -> None:
    if not history:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    gens = [r["generation"] for r in history]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 9.5), sharex=True)
    ax1.plot(gens, [r["best"] for r in history], label="best")
    ax1.plot(gens, [r["mean"] for r in history], label="mean")
    ax1.plot(gens, [r["median"] for r in history], label="median", alpha=0.6)
    ax1.set_ylabel("fitness")
    ax1.legend()
    ax1.grid(alpha=0.3)

    val = [(r["generation"], r["val_laps"]) for r in history if r.get("val_laps") is not None]
    if val:
        ax2.plot(*zip(*val), marker="o", color="tab:green", label="laps on unseen tracks")
    ax2.plot(gens, [r["best_laps"] for r in history], color="tab:blue", alpha=0.5,
             label="laps on training tracks")
    ax2.set_ylabel("laps")
    ax2.legend()
    ax2.grid(alpha=0.3)

    lap_time = [(r["generation"], r["best_lap_steps"] * dt) for r in history
                if r.get("best_lap_steps") is not None and not np.isnan(r["best_lap_steps"])]
    if lap_time:
        ax3.plot(*zip(*lap_time), color="tab:red", label="lap time of the best driver")
        ax3.legend()
    ax3.set_xlabel("generation")
    ax3.set_ylabel("seconds")
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
