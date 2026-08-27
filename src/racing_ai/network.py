from __future__ import annotations

import numpy as np


def genome_size(layer_sizes: tuple[int, ...]) -> int:
    total = 0
    for a, b in zip(layer_sizes[:-1], layer_sizes[1:]):
        total += a * b + b
    return total


def random_genomes(
    rng: np.random.Generator,
    count: int,
    layer_sizes: tuple[int, ...],
    scale: float = 1.0,
) -> np.ndarray:
    chunks = []
    for a, b in zip(layer_sizes[:-1], layer_sizes[1:]):
        # lo scaling di xavier tiene le prime attivazioni fuori dalla saturazione di tanh
        std = scale / np.sqrt(a)
        chunks.append(rng.normal(0.0, std, size=(count, a * b)))
        chunks.append(rng.normal(0.0, 0.1, size=(count, b)))
    return np.concatenate(chunks, axis=1).astype(np.float32)


class PopulationNetwork:
    def __init__(self, genomes: np.ndarray, layer_sizes: tuple[int, ...]):
        expected = genome_size(layer_sizes)
        if genomes.ndim != 2 or genomes.shape[1] != expected:
            raise ValueError(
                f"expected genomes of shape (P, {expected}), got {genomes.shape}"
            )
        self.layer_sizes = tuple(layer_sizes)
        self.genomes = genomes.astype(np.float32, copy=False)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []

        p = genomes.shape[0]
        cursor = 0
        for a, b in zip(layer_sizes[:-1], layer_sizes[1:]):
            w = self.genomes[:, cursor:cursor + a * b].reshape(p, a, b)
            cursor += a * b
            bias = self.genomes[:, cursor:cursor + b]
            cursor += b
            self.weights.append(w)
            self.biases.append(bias)

    @property
    def size(self) -> int:
        return self.genomes.shape[0]

    def forward(self, x: np.ndarray, sel: np.ndarray | None = None) -> np.ndarray:
        h = x.astype(np.float32, copy=False)
        for w, b in zip(self.weights, self.biases):
            ws = w if sel is None else w[sel]
            bs = b if sel is None else b[sel]
            h = np.tanh(np.einsum("mi,mio->mo", h, ws) + bs)
        return h

    def activations(self, x: np.ndarray, sel: np.ndarray | None = None) -> list[np.ndarray]:
        h = x.astype(np.float32, copy=False)
        trace = [h]
        for w, b in zip(self.weights, self.biases):
            ws = w if sel is None else w[sel]
            bs = b if sel is None else b[sel]
            h = np.tanh(np.einsum("mi,mio->mo", h, ws) + bs)
            trace.append(h)
        return trace

    def layer_weights(self, index: int, individual: int = 0) -> np.ndarray:
        return self.weights[index][individual]
