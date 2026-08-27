from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config, config_from_dict
from .ga import Evolution, GenerationStats

HISTORY_HEADER = [
    "generation", "best", "mean", "median", "worst", "best_laps",
    "best_lap_steps", "finish_rate", "crash_rate", "diversity", "sigma", "val_laps",
    "val_finish_rate", "seconds",
]


def save(path: Path, evo: Evolution, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        genomes=evo.genomes,
        best_genome=evo.best_genome,
        best_fitness=np.float64(evo.best_fitness),
        generation=np.int64(evo.generation),
        config=json.dumps(evo.cfg.to_dict()),
        rng_state=json.dumps(evo.rng.bit_generator.state),
        history=json.dumps(history),
    )


def load(path: Path) -> tuple[Evolution, list[dict[str, Any]]]:
    data = np.load(path, allow_pickle=False)
    cfg = config_from_dict(json.loads(str(data["config"])))
    evo = Evolution(cfg)
    evo.genomes = data["genomes"].astype(np.float32)
    evo.best_genome = data["best_genome"].astype(np.float32)
    evo.best_fitness = float(data["best_fitness"])
    evo.generation = int(data["generation"])
    evo.rng.bit_generator.state = json.loads(str(data["rng_state"]))
    history = json.loads(str(data["history"]))
    return evo, history


def load_best(path: Path) -> tuple[np.ndarray, Config, int]:
    data = np.load(path, allow_pickle=False)
    cfg = config_from_dict(json.loads(str(data["config"])))
    return data["best_genome"].astype(np.float32), cfg, int(data["generation"])


def write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(HISTORY_HEADER)]
    for row in history:
        lines.append(",".join(_fmt(row.get(k)) for k in HISTORY_HEADER))
    path.write_text("\n".join(lines) + "\n")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return "" if np.isnan(value) else f"{value:.6f}"
    return str(value)


def stats_to_row(stats: GenerationStats) -> dict[str, Any]:
    return asdict(stats)


def load_field(path: Path, count: int) -> tuple[np.ndarray, Config, int]:
    data = np.load(path, allow_pickle=False)
    cfg = config_from_dict(json.loads(str(data["config"])))
    best = data["best_genome"].astype(np.float32)[None, :]
    if count <= 1:
        return best, cfg, int(data["generation"])
    field = data["genomes"].astype(np.float32)[: count - 1]
    return np.vstack([best, field]), cfg, int(data["generation"])
