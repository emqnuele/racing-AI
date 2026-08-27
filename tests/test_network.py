import numpy as np

from racing_ai.network import PopulationNetwork, genome_size, random_genomes


LAYERS = (8, 16, 12, 8, 2)


def test_genome_size_matches_the_topology():
    assert genome_size(LAYERS) == 8 * 16 + 16 + 16 * 12 + 12 + 12 * 8 + 8 + 8 * 2 + 2


def test_three_hidden_layers():
    net = PopulationNetwork(random_genomes(np.random.default_rng(0), 4, LAYERS), LAYERS)
    assert len(net.weights) == 4  # tre strati nascosti più lo strato di uscita
    assert [w.shape[1:] for w in net.weights] == [(8, 16), (16, 12), (12, 8), (8, 2)]


def test_forward_shape_and_range():
    rng = np.random.default_rng(1)
    net = PopulationNetwork(random_genomes(rng, 10, LAYERS), LAYERS)
    out = net.forward(rng.random((10, 8)))
    assert out.shape == (10, 2)
    assert np.all(np.abs(out) <= 1.0)


def test_individuals_are_independent():
    rng = np.random.default_rng(2)
    net = PopulationNetwork(random_genomes(rng, 6, LAYERS), LAYERS)
    x = np.tile(rng.random((1, 8)), (6, 1))
    out = net.forward(x)
    assert not np.allclose(out[0], out[1])


def test_selection_matches_the_full_forward_pass():
    rng = np.random.default_rng(3)
    net = PopulationNetwork(random_genomes(rng, 8, LAYERS), LAYERS)
    x = rng.random((8, 8))
    full = net.forward(x)
    sel = np.array([5, 2, 7])
    subset = net.forward(x[sel], sel)
    assert np.allclose(full[sel], subset, atol=1e-6)


def test_genome_roundtrip():
    rng = np.random.default_rng(4)
    genomes = random_genomes(rng, 3, LAYERS)
    net = PopulationNetwork(genomes, LAYERS)
    assert np.allclose(net.genomes, genomes)
