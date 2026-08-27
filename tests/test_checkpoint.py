import numpy as np

from racing_ai.checkpoint import load, load_best, save, write_history_csv
from racing_ai.config import Config, EvolutionConfig, NetworkConfig
from racing_ai.ga import Evolution


def _cfg():
    return Config(seed=3, network=NetworkConfig(hidden=(10, 8, 6)),
                  evolution=EvolutionConfig(population=12))


def test_roundtrip_restores_the_whole_run(tmp_path):
    evo = Evolution(_cfg())
    evo.evolve(np.random.default_rng(0).random(12))
    history = [{"generation": 0, "best": 1.5, "seconds": 0.2, "val_laps": None}]

    path = tmp_path / "latest.npz"
    save(path, evo, history)
    restored, restored_history = load(path)

    assert np.array_equal(restored.genomes, evo.genomes)
    assert np.array_equal(restored.best_genome, evo.best_genome)
    assert restored.best_fitness == evo.best_fitness
    assert restored.generation == evo.generation
    assert restored.cfg == evo.cfg
    assert restored.cfg.network.hidden == (10, 8, 6)
    assert restored_history == history


def test_resuming_continues_the_same_random_stream(tmp_path):
    evo = Evolution(_cfg())
    path = tmp_path / "latest.npz"
    save(path, evo, [])

    fitness = np.random.default_rng(1).random(12)
    evo.evolve(fitness)

    restored, _ = load(path)
    restored.evolve(fitness)
    assert np.array_equal(restored.genomes, evo.genomes)


def test_load_best_returns_genome_and_config(tmp_path):
    evo = Evolution(_cfg())
    evo.evolve(np.arange(12, dtype=float))
    path = tmp_path / "latest.npz"
    save(path, evo, [])

    genome, cfg, generation = load_best(path)
    assert genome.shape == (evo.genome_size,)
    assert cfg == evo.cfg
    assert generation == 1


def test_history_csv(tmp_path):
    path = tmp_path / "history.csv"
    write_history_csv(path, [{"generation": 0, "best": 1.0, "val_laps": None, "seconds": 0.5}])
    lines = path.read_text().strip().split("\n")
    assert lines[0].startswith("generation,best,")
    assert lines[1].startswith("0,1.000000,")
    assert lines[1].endswith(",0.500000")
