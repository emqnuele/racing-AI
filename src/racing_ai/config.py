from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


@dataclass(frozen=True)
class TrackConfig:
    n_points: int = 400
    width_override: float = 0.0
    wall_stride: int = 2
    n_gates: int = 24
    start_lateral_jitter: float = 0.25
    start_heading_jitter_deg: float = 6.0


@dataclass(frozen=True)
class CarConfig:
    length: float = 15.0
    width: float = 7.5
    max_speed: float = 110.0
    accel: float = 100.0
    brake: float = 190.0
    drag: float = 0.45
    # limite dello sterzo, vincolante a bassa velocità
    max_turn_rate_deg: float = 150.0
    # limite di aderenza: il raggio di curva cresce col quadrato della velocità
    max_lateral_accel: float = 150.0
    # evita che una macchina quasi ferma giri su se stessa
    low_speed_ref: float = 4.0


@dataclass(frozen=True)
class SensorConfig:
    n_rays: int = 7
    fov_deg: float = 150.0
    max_range: float = 220.0


@dataclass(frozen=True)
class NetworkConfig:
    hidden: tuple[int, ...] = (16, 12, 8)


@dataclass(frozen=True)
class EpisodeConfig:
    dt: float = 0.05
    max_steps: int = 1400
    target_laps: float = 1.0
    stall_steps: int = 80
    stall_progress: float = 4.0
    # quanto vale il tempo sul giro rispetto al giro stesso
    w_time: float = 1.0
    crash_penalty: float = 0.10


@dataclass(frozen=True)
class EvolutionConfig:
    population: int = 100
    elite: int = 5
    survivors: int = 25
    tournament: int = 3
    immigrants: int = 3
    crossover: str = "uniform"
    mutation_rate: float = 0.12
    mutation_sigma: float = 0.25
    mutation_sigma_min: float = 0.03
    mutation_decay: float = 0.995
    reset_rate: float = 0.01
    init_scale: float = 1.0


@dataclass(frozen=True)
class Config:
    track: TrackConfig = field(default_factory=TrackConfig)
    car: CarConfig = field(default_factory=CarConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)

    seed: int = 0
    tracks_per_generation: int = 6
    validation_every: int = 10
    validation_tracks: int = 12
    fitness_aggregation: str = "mean_min"
    difficulty: str = "mixed"
    test_difficulty: str = "hard"

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        return (self.sensors.n_rays + 1, *self.network.hidden, 2)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_NESTED = {
    "track": TrackConfig,
    "car": CarConfig,
    "sensors": SensorConfig,
    "network": NetworkConfig,
    "episode": EpisodeConfig,
    "evolution": EvolutionConfig,
}


def config_from_dict(data: dict[str, Any]) -> Config:
    kwargs: dict[str, Any] = {}
    for key, value in data.items():
        if key in _NESTED:
            sub = _NESTED[key]
            allowed = {f.name for f in fields(sub)}
            clean = {k: v for k, v in value.items() if k in allowed}
            if key == "network" and "hidden" in clean:
                clean["hidden"] = tuple(clean["hidden"])
            kwargs[key] = sub(**clean)
        else:
            kwargs[key] = value
    allowed_top = {f.name for f in fields(Config)}
    kwargs = {k: v for k, v in kwargs.items() if k in allowed_top}
    return Config(**kwargs)
