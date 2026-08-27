import numpy as np

from racing_ai.config import Config
from racing_ai.env import Simulation, evaluate, run_episode
from racing_ai.network import PopulationNetwork, random_genomes
from racing_ai.track import make_track


def _cfg(**kw):
    return Config(seed=0, **kw)


class _Constant:

    def __init__(self, size, steer, throttle):
        self.size = size
        self._out = np.array([[steer, throttle]], dtype=np.float32)

    def forward(self, x, sel=None):
        return np.repeat(self._out, x.shape[0], axis=0)


def test_car_starts_still_and_makes_no_progress_without_throttle():
    cfg = _cfg()
    track = make_track(5, cfg.track)
    sim = Simulation(_Constant(1, 0.0, 0.0), track, cfg)
    assert sim.speed[0] == 0.0
    for _ in range(30):
        sim.step()
    assert sim.progress[0] == 0.0


def test_a_stalled_car_is_retired():
    cfg = _cfg()
    track = make_track(5, cfg.track)
    sim = Simulation(_Constant(1, 0.0, 0.0), track, cfg)
    for _ in range(cfg.episode.stall_steps + 5):
        sim.step()
    assert sim.active.size == 0
    assert sim.step_count <= cfg.episode.stall_steps + 2


def test_full_throttle_straight_ahead_gains_progress_then_crashes():
    cfg = _cfg()
    track = make_track(5, cfg.track)
    sim = Simulation(_Constant(1, 0.0, 1.0), track, cfg)
    for _ in range(60):
        sim.step()
    assert sim.progress[0] > 0
    result = sim.result()
    assert result.laps[0] > 0


def test_leaving_the_track_counts_as_a_crash():
    cfg = _cfg()
    track = make_track(5, cfg.track)
    sim = Simulation(_Constant(1, 1.0, 1.0), track, cfg)
    while not sim.done:
        sim.step()
    assert sim.crashed[0]
    assert not sim.finished[0]


def test_population_episode_shapes():
    cfg = _cfg()
    rng = np.random.default_rng(0)
    net = PopulationNetwork(random_genomes(rng, 12, cfg.layer_sizes), cfg.layer_sizes)
    result = run_episode(net, make_track(9, cfg.track), cfg)
    for arr in (result.fitness, result.laps, result.crashed, result.finished,
                result.steps, result.mean_speed, result.min_speed, result.max_speed):
        assert arr.shape == (12,)
    assert np.all(result.mean_speed >= 0)
    assert np.all(result.min_speed <= result.max_speed)


def test_retiring_cars_does_not_change_the_survivors_outcome():
    cfg = _cfg()
    rng = np.random.default_rng(7)
    genomes = random_genomes(rng, 16, cfg.layer_sizes)
    track = make_track(21, cfg.track)

    group = run_episode(PopulationNetwork(genomes, cfg.layer_sizes), track, cfg)
    for i in (0, 5, 11):
        alone = run_episode(PopulationNetwork(genomes[i:i + 1], cfg.layer_sizes), track, cfg)
        assert np.isclose(group.laps[i], alone.laps[0], atol=1e-9)
        assert group.steps[i] == alone.steps[0]


def test_episode_is_deterministic():
    cfg = _cfg()
    genomes = random_genomes(np.random.default_rng(3), 8, cfg.layer_sizes)
    track = make_track(31, cfg.track)
    a = run_episode(PopulationNetwork(genomes, cfg.layer_sizes), track, cfg)
    b = run_episode(PopulationNetwork(genomes, cfg.layer_sizes), track, cfg)
    assert np.array_equal(a.fitness, b.fitness)


def test_fitness_aggregation_modes():
    genomes = random_genomes(np.random.default_rng(4), 10, Config().layer_sizes)
    tracks = [make_track(s) for s in (41, 42, 43)]

    def run(mode):
        cfg = _cfg(fitness_aggregation=mode)
        net = PopulationNetwork(genomes, cfg.layer_sizes)
        return evaluate(net, tracks, cfg)

    mean_fit, results = run("mean")
    stacked = np.stack([r.fitness for r in results])
    assert np.allclose(mean_fit, stacked.mean(axis=0))
    assert np.allclose(run("min")[0], stacked.min(axis=0))
    assert np.allclose(run("mean_min")[0],
                       0.7 * stacked.mean(axis=0) + 0.3 * stacked.min(axis=0))


def test_unknown_fitness_aggregation_is_rejected():
    import pytest
    cfg = _cfg(fitness_aggregation="nonsense")
    net = PopulationNetwork(random_genomes(np.random.default_rng(0), 4, cfg.layer_sizes),
                            cfg.layer_sizes)
    with pytest.raises(ValueError):
        evaluate(net, [make_track(1, cfg.track)], cfg)


def test_cars_never_collide_with_each_other():
    cfg = _cfg()
    genomes = random_genomes(np.random.default_rng(0), 60, cfg.layer_sizes)
    track = make_track(1_000_000, cfg.track, "hard")

    pack = run_episode(PopulationNetwork(genomes, cfg.layer_sizes), track, cfg)
    for i in (0, 17, 38, 59):
        alone = run_episode(PopulationNetwork(genomes[i:i + 1], cfg.layer_sizes), track, cfg)
        assert pack.laps[i] == alone.laps[0]
        assert pack.steps[i] == alone.steps[0]
        assert pack.crashed[i] == alone.crashed[0]


def _result_for(cfg, track, *, finished, crashed, steps, progress, speed_sum=None):
    sim = Simulation(_Constant(len(steps), 0.0, 0.0), track, cfg)
    sim.finished = np.array(finished, dtype=bool)
    sim.crashed = np.array(crashed, dtype=bool)
    sim.steps = np.array(steps, dtype=np.int64)
    sim.progress = np.array(progress, dtype=np.float64) * track.length
    if speed_sum is not None:
        sim.speed_sum = np.array(speed_sum, dtype=np.float64)
    return sim.result()


def test_finishing_always_outranks_not_finishing():
    cfg = _cfg()
    track = make_track(7, cfg.track)
    # il peggior finisher possibile contro il miglior non-finisher possibile
    r = _result_for(cfg, track,
                    finished=[True, False, False],
                    crashed=[False, False, True],
                    steps=[cfg.episode.max_steps, 10, 10],
                    progress=[1.0, 0.999, 0.999])
    assert r.fitness[0] > r.fitness[1] > r.fitness[2]


def test_lap_time_is_the_only_thing_separating_finishers():
    cfg = _cfg()
    track = make_track(8, cfg.track)
    r = _result_for(cfg, track,
                    finished=[True, True, True],
                    crashed=[False, False, False],
                    steps=[300, 600, 900],
                    progress=[1.0, 1.0, 1.0],
                    speed_sum=[9e4, 1.0, 9e4])
    assert r.fitness[0] > r.fitness[1] > r.fitness[2]
    gaps = np.diff(r.fitness)
    assert np.allclose(gaps[0], gaps[1])


def test_speed_earns_nothing_without_finishing_the_lap():
    cfg = _cfg()
    track = make_track(9, cfg.track)
    # stessa distanza percorsa, velocità media molto diversa
    r = _result_for(cfg, track,
                    finished=[False, False],
                    crashed=[False, False],
                    steps=[200, 800],
                    progress=[0.5, 0.5],
                    speed_sum=[2e4, 2e2])
    assert r.fitness[0] == r.fitness[1]


def test_among_non_finishers_progress_ranks_and_crashing_costs():
    cfg = _cfg()
    track = make_track(10, cfg.track)
    r = _result_for(cfg, track,
                    finished=[False, False, False],
                    crashed=[False, True, False],
                    steps=[400, 400, 400],
                    progress=[0.8, 0.8, 0.3])
    assert r.fitness[0] > r.fitness[1] > r.fitness[2]
    assert np.isclose(r.fitness[0] - r.fitness[1], cfg.episode.crash_penalty)
