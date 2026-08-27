from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .network import PopulationNetwork
from .sensors import cast_rays, ray_offsets
from .track import Track


@dataclass
class EpisodeResult:
    fitness: np.ndarray
    laps: np.ndarray
    crashed: np.ndarray
    finished: np.ndarray
    steps: np.ndarray
    mean_speed: np.ndarray
    min_speed: np.ndarray
    max_speed: np.ndarray


class Simulation:

    def __init__(self, net: PopulationNetwork, track: Track, cfg: Config):
        self.net = net
        self.track = track
        self.cfg = cfg
        self.offsets = ray_offsets(cfg.sensors)

        p = net.size
        self.pos = np.repeat(track.start_pos[None, :], p, axis=0)
        self.heading = np.full(p, track.start_heading, dtype=np.float64)
        self.speed = np.zeros(p, dtype=np.float64)
        self.rays = np.full((p, cfg.sensors.n_rays), cfg.sensors.max_range)
        self.inputs = np.zeros((p, cfg.sensors.n_rays + 1))
        self.controls = np.zeros((p, 2))

        self.lap = np.zeros(p, dtype=np.int64)
        self.last_index = np.full(p, track.start_index, dtype=np.int64)
        self.start_s = float(track.cum_len[track.start_index])
        self.progress = np.zeros(p, dtype=np.float64)
        self.best_progress = np.zeros(p, dtype=np.float64)
        self.stall = np.zeros(p, dtype=np.int64)

        self.crashed = np.zeros(p, dtype=bool)
        self.finished = np.zeros(p, dtype=bool)
        self.steps = np.zeros(p, dtype=np.int64)
        self.speed_sum = np.zeros(p, dtype=np.float64)
        self.speed_min = np.full(p, np.inf)
        self.speed_max = np.zeros(p)

        self.active = np.arange(p)
        self.step_count = 0

    @property
    def done(self) -> bool:
        return self.active.size == 0 or self.step_count >= self.cfg.episode.max_steps

    def sense(self, sel: np.ndarray) -> np.ndarray:
        rays = cast_rays(
            self.pos[sel], self.heading[sel], self.track,
            self.offsets, self.cfg.sensors.max_range,
        )
        self.rays[sel] = rays
        inputs = np.empty((sel.size, self.cfg.sensors.n_rays + 1))
        inputs[:, : self.cfg.sensors.n_rays] = rays / self.cfg.sensors.max_range
        inputs[:, -1] = self.speed[sel] / self.cfg.car.max_speed
        return inputs

    def step(self) -> None:
        if self.done:
            return
        sel = self.active
        cfg = self.cfg
        car = cfg.car
        dt = cfg.episode.dt

        inputs = self.sense(sel)
        out = self.net.forward(inputs, sel)
        steer = np.clip(out[:, 0].astype(np.float64), -1.0, 1.0)
        throttle = np.clip(out[:, 1].astype(np.float64), -1.0, 1.0)
        self.inputs[sel] = inputs
        self.controls[sel, 0] = steer
        self.controls[sel, 1] = throttle

        v = self.speed[sel]
        accel = np.where(throttle >= 0.0, throttle * car.accel, throttle * car.brake)
        v = v + accel * dt
        v -= car.drag * v * dt
        v = np.clip(v, 0.0, car.max_speed)

        # sterzo al limite quando va piano, aderenza al limite quando va forte
        rack = np.deg2rad(car.max_turn_rate_deg)
        omega_cap = np.minimum(rack, car.max_lateral_accel / np.maximum(v, 1e-3))
        omega_cap *= v / (v + car.low_speed_ref)
        self.heading[sel] += steer * omega_cap * dt
        self.speed[sel] = v
        self.pos[sel] += np.stack([np.cos(self.heading[sel]), np.sin(self.heading[sel])], axis=1) * (v * dt)[:, None]

        idx, lateral = self.track.locate(self.pos[sel])

        n = self.track.n_points
        prev = self.last_index[sel]
        forward_wrap = (prev - idx) > (n // 2)
        backward_wrap = (idx - prev) > (n // 2)
        self.lap[sel] += forward_wrap.astype(np.int64) - backward_wrap.astype(np.int64)
        self.last_index[sel] = idx

        self.progress[sel] = self.track.cum_len[idx] + self.lap[sel] * self.track.length - self.start_s
        self.speed_sum[sel] += v
        self.speed_min[sel] = np.minimum(self.speed_min[sel], v)
        self.speed_max[sel] = np.maximum(self.speed_max[sel], v)
        self.steps[sel] += 1

        gained = self.progress[sel] > self.best_progress[sel] + cfg.episode.stall_progress
        self.best_progress[sel] = np.maximum(self.best_progress[sel], self.progress[sel])
        self.stall[sel] = np.where(gained, 0, self.stall[sel] + 1)

        off_track = lateral > self.track.half_width
        stalled = self.stall[sel] > cfg.episode.stall_steps
        done_lap = self.progress[sel] >= cfg.episode.target_laps * self.track.length

        self.crashed[sel[off_track]] = True
        self.finished[sel[done_lap & ~off_track]] = True

        self.active = sel[~(off_track | stalled | done_lap)]
        self.step_count += 1

    def result(self) -> EpisodeResult:
        cfg = self.cfg.episode
        steps = np.maximum(self.steps, 1)
        mean_speed = self.speed_sum / steps / self.cfg.car.max_speed
        laps = self.progress / self.track.length

        # cancello duro: chi finisce sta sempre sopra chi non finisce, così il
        # tempo sul giro non è mai barattabile con il rischio di schiantarsi
        time_left = 1.0 - self.steps / cfg.max_steps
        fitness = np.where(
            self.finished,
            1.0 + cfg.w_time * time_left,
            np.clip(laps / cfg.target_laps, 0.0, 1.0)
            - np.where(self.crashed, cfg.crash_penalty, 0.0),
        )

        return EpisodeResult(
            fitness=fitness,
            laps=laps,
            crashed=self.crashed.copy(),
            finished=self.finished.copy(),
            steps=self.steps.copy(),
            mean_speed=mean_speed,
            min_speed=np.where(np.isinf(self.speed_min), 0.0, self.speed_min),
            max_speed=self.speed_max.copy(),
        )


def run_episode(net: PopulationNetwork, track: Track, cfg: Config) -> EpisodeResult:
    sim = Simulation(net, track, cfg)
    while not sim.done:
        sim.step()
    return sim.result()


def combine_fitness(results: list[EpisodeResult], mode: str) -> np.ndarray:
    stacked = np.stack([r.fitness for r in results], axis=0)
    if mode == "mean":
        return stacked.mean(axis=0)
    if mode == "min":
        return stacked.min(axis=0)
    if mode == "mean_min":
        return 0.7 * stacked.mean(axis=0) + 0.3 * stacked.min(axis=0)
    raise ValueError(f"unknown fitness_aggregation: {mode!r}")


def evaluate(
    net: PopulationNetwork,
    tracks: list[Track],
    cfg: Config,
) -> tuple[np.ndarray, list[EpisodeResult]]:
    results = [run_episode(net, track, cfg) for track in tracks]
    return combine_fitness(results, cfg.fitness_aggregation), results
