import numpy as np

from racing_ai.config import SensorConfig
from racing_ai.sensors import cast_rays, ray_offsets
from racing_ai.track import make_track


class _Walls:

    def __init__(self, a, b):
        self.wall_a = np.asarray(a, dtype=float)
        self.wall_b = np.asarray(b, dtype=float)


def test_ray_distance_and_range_clipping():
    wall = _Walls([[10.0, -50.0]], [[10.0, 50.0]])
    forward = cast_rays(np.array([[0.0, 0.0]]), np.array([0.0]), wall, np.array([0.0]), 100.0)
    assert np.isclose(forward[0, 0], 10.0)

    # girata dall'altra parte: niente da colpire, tagliata alla portata massima
    away = cast_rays(np.array([[0.0, 0.0]]), np.array([np.pi]), wall, np.array([0.0]), 100.0)
    assert np.isclose(away[0, 0], 100.0)

    # un raggio a 45 gradi percorre sqrt(2) volte la distanza per arrivare allo stesso muro
    diagonal = cast_rays(
        np.array([[0.0, 0.0]]), np.array([0.0]), wall, np.array([np.pi / 4]), 100.0
    )
    assert np.isclose(diagonal[0, 0], 10.0 * np.sqrt(2))


def test_segment_endpoints_are_respected():
    short = _Walls([[10.0, 1.0]], [[10.0, 5.0]])
    dist = cast_rays(np.array([[0.0, 0.0]]), np.array([0.0]), short, np.array([0.0]), 100.0)
    assert np.isclose(dist[0, 0], 100.0)


def test_rays_from_the_centreline_stay_inside_the_track():
    cfg = SensorConfig()
    track = make_track(11)
    offsets = ray_offsets(cfg)
    idx = np.arange(0, track.n_points, 20)
    headings = np.arctan2(track.tangent[idx, 1], track.tangent[idx, 0])
    dist = cast_rays(track.center[idx], headings, track, offsets, cfg.max_range)

    assert dist.shape == (idx.size, cfg.n_rays)
    assert np.all(dist > 0)
    # i raggi laterali devono trovare un muro a circa metà larghezza del tracciato
    assert np.allclose(dist[:, 0], track.half_width, atol=2.0)
    assert np.allclose(dist[:, -1], track.half_width, atol=2.0)


def test_ray_offsets_span_the_field_of_view():
    cfg = SensorConfig(n_rays=7, fov_deg=150.0)
    offsets = ray_offsets(cfg)
    assert offsets.size == 7
    assert np.isclose(np.rad2deg(offsets[0]), -75.0)
    assert np.isclose(np.rad2deg(offsets[-1]), 75.0)
    assert np.isclose(offsets[3], 0.0)
