from __future__ import annotations

import numpy as np

from .config import SensorConfig
from .track import Track


def ray_offsets(cfg: SensorConfig) -> np.ndarray:
    if cfg.n_rays == 1:
        return np.zeros(1)
    half = np.deg2rad(cfg.fov_deg) * 0.5
    return np.linspace(-half, half, cfg.n_rays)


def cast_rays(
    origins: np.ndarray,
    headings: np.ndarray,
    track: Track,
    offsets: np.ndarray,
    max_range: float,
) -> np.ndarray:
    p = origins.shape[0]
    r = offsets.shape[0]

    angles = headings[:, None] + offsets[None, :]
    dx = np.cos(angles).reshape(-1)
    dy = np.sin(angles).reshape(-1)
    ox = np.repeat(origins[:, 0], r)
    oy = np.repeat(origins[:, 1], r)

    ax = track.wall_a[:, 0]
    ay = track.wall_a[:, 1]
    ex = track.wall_b[:, 0] - ax
    ey = track.wall_b[:, 1] - ay

    # o + t*d = a + u*e  ->  t = (w x e) / (d x e), u = (w x d) / (d x e)
    denom = dx[:, None] * ey[None, :] - dy[:, None] * ex[None, :]
    wx = ax[None, :] - ox[:, None]
    wy = ay[None, :] - oy[:, None]

    safe = np.abs(denom) > 1e-12
    inv = np.where(safe, 1.0 / np.where(safe, denom, 1.0), 0.0)
    t = (wx * ey[None, :] - wy * ex[None, :]) * inv
    u = (wx * dy[:, None] - wy * dx[:, None]) * inv

    hit = safe & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    t = np.where(hit, t, np.inf)
    dist = np.min(t, axis=1)
    return np.minimum(dist, max_range).reshape(p, r)
