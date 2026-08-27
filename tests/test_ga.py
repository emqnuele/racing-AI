import numpy as np
import pytest

from racing_ai.config import Config, EvolutionConfig
from racing_ai.ga import Evolution


def _cfg(**kw):
    return Config(seed=0, evolution=EvolutionConfig(population=20, elite=3,
                                                    survivors=8, immigrants=2, **kw))


def test_initial_population_shape():
    evo = Evolution(_cfg())
    assert evo.genomes.shape == (20, evo.genome_size)
    assert evo.genomes.dtype == np.float32


def test_population_size_is_stable_across_generations():
    evo = Evolution(_cfg())
    rng = np.random.default_rng(0)
    for _ in range(5):
        evo.evolve(rng.random(20))
        assert evo.genomes.shape == (20, evo.genome_size)


def test_elites_survive_untouched():
    evo = Evolution(_cfg())
    fitness = np.arange(20, dtype=float)
    champion = evo.genomes[19].copy()
    runner_up = evo.genomes[18].copy()
    evo.evolve(fitness)
    assert np.array_equal(evo.genomes[0], champion)
    assert np.array_equal(evo.genomes[1], runner_up)


def test_best_genome_is_kept_only_when_the_score_improves():
    evo = Evolution(_cfg())
    best = evo.genomes[4].copy()
    assert evo.remember_best(best, 10.0)
    assert evo.best_fitness == 10.0
    assert np.array_equal(evo.best_genome, best)

    # un punteggio peggiore non deve sovrascrivere il record
    assert not evo.remember_best(evo.genomes[7], 9.0)
    assert evo.best_fitness == 10.0
    assert np.array_equal(evo.best_genome, best)


def test_generation_fitness_never_touches_the_best_genome():
    evo = Evolution(_cfg())
    evo.remember_best(evo.genomes[4], 10.0)
    kept = evo.best_genome.copy()
    fitness = np.zeros(20)
    fitness[9] = 1e6
    evo.evolve(fitness)
    assert evo.best_fitness == 10.0
    assert np.array_equal(evo.best_genome, kept)


def test_mutation_sigma_decays_to_the_floor():
    evo = Evolution(_cfg())
    first = evo.sigma
    evo.generation = 5000
    assert evo.sigma < first
    assert evo.sigma == pytest.approx(evo.cfg.evolution.mutation_sigma_min)


@pytest.mark.parametrize("mode", ["uniform", "one_point", "blend"])
def test_crossover_children_come_from_their_parents(mode):
    evo = Evolution(_cfg(crossover=mode))
    a = np.zeros(evo.genome_size, dtype=np.float32)
    b = np.ones(evo.genome_size, dtype=np.float32)
    child = evo._crossover(a, b)
    assert child.shape == (evo.genome_size,)
    assert np.all(child >= 0.0) and np.all(child <= 1.0)


def test_unknown_crossover_is_rejected():
    evo = Evolution(_cfg(crossover="nonsense"))
    with pytest.raises(ValueError):
        evo._crossover(evo.genomes[0], evo.genomes[1])


def test_mutation_perturbs_only_some_genes():
    evo = Evolution(_cfg(mutation_rate=0.1, reset_rate=0.0))
    genome = np.zeros(evo.genome_size, dtype=np.float32)
    mutated = evo._mutate(genome)
    changed = np.count_nonzero(mutated != genome)
    assert 0 < changed < evo.genome_size


def test_evolution_is_reproducible_from_a_seed():
    fitness = np.random.default_rng(1).random(20)
    runs = []
    for _ in range(2):
        evo = Evolution(_cfg())
        for _ in range(3):
            evo.evolve(fitness)
        runs.append(evo.genomes.copy())
    assert np.array_equal(runs[0], runs[1])


def test_immigrants_are_fresh_blood():
    evo = Evolution(_cfg())
    before = evo.genomes.copy()
    evo.evolve(np.arange(20, dtype=float))
    tail = evo.genomes[-2:]
    for row in tail:
        assert not any(np.array_equal(row, old) for old in before)


def test_selection_prefers_fitter_individuals():
    evo = Evolution(_cfg())
    ranked = np.arange(8)
    picks = [evo._tournament(ranked) for _ in range(2000)]
    assert np.mean(picks) < 3.5


def test_completing_one_more_track_always_beats_any_lap_time():
    from racing_ai.trainer import validation_score
    n = 12
    for done in range(n):
        slower = validation_score((done + 1) / n, 0.0, n)
        faster = validation_score(done / n, 1.0, n)
        assert slower > faster


def test_lap_time_breaks_ties_between_equal_completion():
    from racing_ai.trainer import validation_score
    assert validation_score(1.0, 0.7, 12) > validation_score(1.0, 0.3, 12)
