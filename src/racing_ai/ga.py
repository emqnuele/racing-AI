from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .network import genome_size, random_genomes


@dataclass
class GenerationStats:
    generation: int
    best: float
    mean: float
    median: float
    worst: float
    best_laps: float
    best_lap_steps: float
    finish_rate: float
    crash_rate: float
    diversity: float
    sigma: float


class Evolution:

    def __init__(self, cfg: Config, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng(cfg.seed)
        self.generation = 0
        self.genome_size = genome_size(cfg.layer_sizes)
        self.genomes = random_genomes(
            self.rng, cfg.evolution.population, cfg.layer_sizes, cfg.evolution.init_scale
        )
        self.best_genome = self.genomes[0].copy()
        self.best_fitness = -np.inf

    @property
    def sigma(self) -> float:
        e = self.cfg.evolution
        return max(e.mutation_sigma_min, e.mutation_sigma * (e.mutation_decay ** self.generation))

    def _tournament(self, ranked: np.ndarray) -> int:
        k = min(self.cfg.evolution.tournament, ranked.size)
        picks = self.rng.integers(0, ranked.size, size=k)
        return int(ranked[picks.min()])

    def _crossover(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mode = self.cfg.evolution.crossover
        if mode == "uniform":
            mask = self.rng.random(self.genome_size) < 0.5
            return np.where(mask, a, b)
        if mode == "one_point":
            cut = int(self.rng.integers(1, self.genome_size))
            child = a.copy()
            child[cut:] = b[cut:]
            return child
        if mode == "blend":
            alpha = self.rng.random(self.genome_size).astype(np.float32)
            return alpha * a + (1.0 - alpha) * b
        raise ValueError(f"unknown crossover: {mode!r}")

    def _mutate(self, genome: np.ndarray) -> np.ndarray:
        e = self.cfg.evolution
        sigma = self.sigma
        hits = self.rng.random(self.genome_size) < e.mutation_rate
        noise = self.rng.normal(0.0, sigma, size=self.genome_size).astype(np.float32)
        genome = genome + np.where(hits, noise, 0.0).astype(np.float32)

        if e.reset_rate > 0.0:
            resets = self.rng.random(self.genome_size) < e.reset_rate
            if resets.any():
                fresh = self.rng.normal(0.0, 0.5, size=self.genome_size).astype(np.float32)
                genome = np.where(resets, fresh, genome).astype(np.float32)
        return genome

    def stats(self, fitness: np.ndarray, laps: np.ndarray, lap_steps: np.ndarray,
              finished: np.ndarray, crashed: np.ndarray) -> GenerationStats:
        order = np.argsort(fitness)[::-1]
        return GenerationStats(
            generation=self.generation,
            best=float(fitness[order[0]]),
            mean=float(fitness.mean()),
            median=float(np.median(fitness)),
            worst=float(fitness[order[-1]]),
            best_laps=float(laps[order[0]]),
            best_lap_steps=float(lap_steps[order[0]]),
            finish_rate=float(finished.mean()),
            crash_rate=float(crashed.mean()),
            diversity=float(np.mean(np.std(self.genomes, axis=0))),
            sigma=self.sigma,
        )

    def remember_best(self, genome: np.ndarray, score: float) -> bool:
        # la fitness di generazioni diverse non è confrontabile: i tracciati cambiano
        if score <= self.best_fitness:
            return False
        self.best_fitness = float(score)
        self.best_genome = genome.copy()
        return True

    def evolve(self, fitness: np.ndarray) -> None:
        e = self.cfg.evolution
        order = np.argsort(fitness)[::-1]

        survivors = order[: max(2, e.survivors)]
        children = np.empty_like(self.genomes)

        n_elite = min(e.elite, e.population)
        children[:n_elite] = self.genomes[order[:n_elite]]

        n_immigrants = min(e.immigrants, e.population - n_elite)
        if n_immigrants > 0:
            children[e.population - n_immigrants:] = random_genomes(
                self.rng, n_immigrants, self.cfg.layer_sizes, e.init_scale
            )

        for i in range(n_elite, e.population - n_immigrants):
            pa = self._tournament(survivors)
            pb = self._tournament(survivors)
            child = self._crossover(self.genomes[pa], self.genomes[pb])
            children[i] = self._mutate(child)

        self.genomes = children.astype(np.float32)
        self.generation += 1
