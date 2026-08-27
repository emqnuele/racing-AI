import csv
import re
from pathlib import Path

import numpy as np
import pytest

from racing_ai.config import Config
from racing_ai.env import combine_fitness
from racing_ai.network import genome_size
from racing_ai.track import (
    TEST_SEED_BASE, _min_curvature_radius, holdout_seeds, make_track,
)

# il readme è a capo automatico, quindi si confronta a spazi normalizzati
README = re.sub(r"\s+", " ", (Path(__file__).parent.parent / "README.md").read_text())
CFG = Config()


def claims(*fragments):
    for f in fragments:
        assert f in README, f"il readme non dice piu: {f!r}"


def test_sensor_and_input_count():
    claims("Inputs: 8.", "Seven raycasts")
    assert CFG.sensors.n_rays == 7
    assert CFG.layer_sizes[0] == CFG.sensors.n_rays + 1 == 8


def test_network_shape_and_genome_size():
    claims(f"Brain: {genome_size(CFG.layer_sizes)} weights.",
           " → ".join(str(n) for n in CFG.layer_sizes))
    assert CFG.layer_sizes == (8, 16, 12, 8, 2)


def test_tracks_per_generation_and_seed_split():
    claims("Six procedurally generated tracks per generation")
    assert CFG.tracks_per_generation == 6
    claims("training `[0, 10⁶)`, evaluation `[10⁶, ∞)`")
    assert TEST_SEED_BASE == 10 ** 6


def test_fitness_shape():
    claims("`1 + time left`", f"− {CFG.episode.crash_penalty:.1f} · crash")
    assert CFG.episode.w_time == 1.0
    assert CFG.episode.crash_penalty == 0.10


def test_aggregation_weights():
    claims("`0.7 · mean + 0.3 · worst`")
    assert CFG.fitness_aggregation == "mean_min"

    class R:
        def __init__(self, f):
            self.fitness = np.array(f)

    got = combine_fitness([R([0.0]), R([1.0]), R([2.0])], "mean_min")
    assert np.isclose(got[0], 0.7 * 1.0 + 0.3 * 0.0)


def test_selection_parameters():
    e = CFG.evolution
    claims(f"{e.elite} elites", f"top {e.survivors}", f"tournament (k={e.tournament})",
           f"rate {e.mutation_rate}", f"σ {e.mutation_sigma}",
           f"×{e.mutation_decay}", f"floor of {e.mutation_sigma_min}",
           f"{e.immigrants} random immigrants", f"population {e.population}")
    assert e.crossover == "uniform" and "uniform crossover" in README


def test_champion_validation_interval():
    claims(f"Every {CFG.validation_every} generations the leader is raced on held-out tracks")


def test_physics_constants():
    car = CFG.car
    claims(f"below ~{car.low_speed_ref:.0f} u/s ({car.max_turn_rate_deg:.0f}°/s)",
           f"({car.max_lateral_accel:.0f} u/s² lateral)",
           f"At the {car.max_speed:.0f} u/s top speed")


def test_flat_out_turning_radius():
    car = CFG.car
    rack = np.deg2rad(car.max_turn_rate_deg)
    omega = min(rack, car.max_lateral_accel / car.max_speed)
    omega *= car.max_speed / (car.max_speed + car.low_speed_ref)
    radius = car.max_speed / omega
    stated = float(re.search(r"needs ~(\d+) units of radius", README).group(1))
    assert abs(radius - stated) < 1.0, f"il readme dice ~{stated:.0f}, la fisica dà {radius:.1f}"


def test_tightest_generated_corner():
    tightest = min(_min_curvature_radius(make_track(s, difficulty="hard").center)
                   for s in holdout_seeds(12))
    stated = float(re.search(r"corners go down to ~(\d+)", README).group(1))
    assert abs(tightest - stated) < 3.0, \
        f"il readme dice ~{stated:.0f}, i tracciati generano curve da {tightest:.1f}"


def test_the_car_cannot_take_the_tightest_corners_flat_out():
    # è l'affermazione che regge tutta la sezione physics
    flat_out = float(re.search(r"needs ~(\d+) units of radius", README).group(1))
    tightest = float(re.search(r"corners go down to ~(\d+)", README).group(1))
    assert tightest < flat_out


CHECKPOINT = Path(__file__).parent.parent / "checkpoints" / "latest.npz"
HISTORY = Path(__file__).parent.parent / "checkpoints" / "history.csv"


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="nessun checkpoint allenato")
def test_the_shipped_champion_matches_the_results_section():
    from racing_ai.checkpoint import load_best
    from racing_ai.env import evaluate
    from racing_ai.network import PopulationNetwork

    n, lap_time = re.search(
        r"completes (\d+) out of \d+ held-out tracks without crashing, averaging "
        r"([\d.]+) s a lap", README).groups()
    genome, cfg, _ = load_best(CHECKPOINT)
    tracks = [make_track(s, cfg.track, cfg.test_difficulty)
              for s in holdout_seeds(int(n), 500)]
    _, results = evaluate(PopulationNetwork(genome[None, :], cfg.layer_sizes), tracks, cfg)

    done = np.array([bool(r.finished[0]) for r in results])
    crashed = np.array([bool(r.crashed[0]) for r in results])
    steps = np.array([int(r.steps[0]) for r in results], dtype=float)
    assert done.all(), f"il readme dice {n}/{n}, ne completa {done.sum()}"
    assert not crashed.any()
    measured = (steps[done] * cfg.episode.dt).mean()
    assert abs(measured - float(lap_time)) < 0.5, \
        f"il readme dice {lap_time}s, il campione gira in {measured:.2f}s"


@pytest.mark.skipif(not HISTORY.exists(), reason="nessuna storia di training")
def test_training_numbers_match_the_results_section():
    rows = list(csv.DictReader(HISTORY.open()))
    first = rows[0]
    crash, laps, pace = re.search(
        r"Generation 0: (\d+)% crash rate, best driver completes ([\d.]+) laps "
        r"at a (\d+) s pace", README).groups()
    assert round(float(first["crash_rate"]) * 100) == int(crash)
    assert abs(float(first["best_laps"]) - float(laps)) < 0.01
    assert abs(float(first["best_lap_steps"]) * 0.05 - float(pace)) < 1.0

    stated = int(re.search(r"the field\s*finishes (\d+)% of the time", README).group(1))
    tail = [float(r["finish_rate"]) for r in rows[-50:]]
    assert round(100 * sum(tail) / len(tail)) == stated

    # il 3.3% del confronto viene da un run con la vecchia fitness, non riproducibile qui
    stated = float(re.search(r"times as often \(([\d.]+)% vs", README).group(1))
    tail = [float(r["crash_rate"]) for r in rows[-50:]]
    assert abs(100 * sum(tail) / len(tail) - stated) < 0.1

    val = [(int(r["generation"]), float(r["val_laps"])) for r in rows if r["val_laps"]]
    first = next(g for g, x in val if x >= 1.0)
    assert first == int(re.search(r"first reach 1\.0 at generation (\d+)", README).group(1))

    # "regge solo da 140": prima ci sono ricadute, dopo no fino alle ultime due misure
    held = int(re.search(r"only hold it from\s*(\d+) on", README).group(1))
    assert all(x >= 1.0 for g, x in val if held <= g <= 270)
    assert any(x < 1.0 for g, x in val if first < g < held)

    dip_gen, dip = re.search(r"as late as generation (\d+) the leader still drops "
                             r"to (\d+\.\d+)", README).groups()
    assert round(dict(val)[int(dip_gen)], 2) == float(dip)
    last_two = [round(x, 2) for _, x in val[-2:]]
    assert last_two == [float(x) for x in
                        re.search(r"last two checks fall to (\d+\.\d+) and (\d+\.\d+)",
                                  README).groups()]


@pytest.mark.skipif(not HISTORY.exists(), reason="nessuna storia di training")
def test_lap_time_curve_matches_the_results_section():
    # le generazioni in cui nessuno chiude il giro non hanno un tempo
    rows = [r for r in csv.DictReader(HISTORY.open()) if r["best_lap_steps"]]
    gen = np.array([int(r["generation"]) for r in rows])
    secs = np.array([float(r["best_lap_steps"]) for r in rows]) * 0.05

    start, worst, worst_gen, plateau_from, plateau = re.search(
        r"(\d+) s at generation 0, a (\d+) s worst at (\d+) before the field learns "
        r"to keep moving, a (?:(\d+) s median from generation )?(\d+)", README).groups()[:5]
    assert round(secs[0]) == int(start)
    assert round(secs.max()) == int(worst)
    assert gen[secs.argmax()] == int(worst_gen)

    median, since = re.search(r"a (\d+) s median from generation (\d+)", README).groups()
    assert round(np.median(secs[gen >= int(since)])) == int(median)

    # "200 generazioni per 0.2 s": il guadagno residuo dopo il plateau
    gain = float(re.search(r"generations\s*for ([\d.]+) s of it", README).group(1))
    early = np.median(secs[(gen >= 100) & (gen < 200)])
    late = np.median(secs[gen >= 200])
    assert abs((early - late) - gain) < 0.05
