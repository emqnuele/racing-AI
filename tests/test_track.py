import numpy as np
import pytest

from racing_ai.config import TrackConfig
from racing_ai.track import (
    TEST_SEED_BASE, _min_curvature_radius, _min_separation,
    holdout_seeds, make_track, resolve_difficulty, training_seeds,
)

LEVELS = ("easy", "medium", "hard")


def _segments_cross(a1, a2, b1, b2):
    def side(p, q, r):
        return np.sign((q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1])
                       - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0]))
    d1, d2 = side(b1, b2, a1), side(b1, b2, a2)
    d3, d4 = side(a1, a2, b1), side(a1, a2, b2)
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def _self_intersects(loop):
    n = len(loop)
    a1 = loop
    a2 = np.roll(loop, -1, axis=0)
    for offset in range(2, n - 1):
        b1 = np.roll(a1, -offset, axis=0)
        b2 = np.roll(a2, -offset, axis=0)
        if _segments_cross(a1, a2, b1, b2).any():
            return True
    return False


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 999, 123456])
@pytest.mark.parametrize("level", LEVELS)
def test_track_is_well_formed(seed, level):
    cfg = TrackConfig()
    track = make_track(seed, cfg, level)

    assert track.center.shape == (cfg.n_points, 2)
    assert track.length > 0
    assert track.difficulty == level

    d_left = np.linalg.norm(track.left - track.center, axis=1)
    d_right = np.linalg.norm(track.right - track.center, axis=1)
    assert np.allclose(d_left, track.half_width)
    assert np.allclose(d_right, track.half_width)

    step = np.linalg.norm(np.diff(track.center, axis=0), axis=1)
    assert step.std() / step.mean() < 0.02


@pytest.mark.parametrize("level", LEVELS)
def test_walls_never_cross_themselves(level):
    for seed in range(6):
        track = make_track(seed, difficulty=level)
        assert not _self_intersects(track.left[::2]), f"{level} seed {seed} left wall"
        assert not _self_intersects(track.right[::2]), f"{level} seed {seed} right wall"


@pytest.mark.parametrize("level", LEVELS)
def test_opposite_sides_of_the_track_stay_apart(level):
    for seed in range(6):
        track = make_track(seed, difficulty=level)
        assert _min_separation(track.center, 45) > track.width


@pytest.mark.parametrize("level", LEVELS)
def test_corners_are_wider_than_the_track(level):
    for seed in range(8):
        track = make_track(seed, difficulty=level)
        assert _min_curvature_radius(track.center) > track.half_width


def test_difficulty_levels_are_ordered():
    stats = {}
    for level in LEVELS:
        tracks = [make_track(s, difficulty=level) for s in range(12)]
        stats[level] = (
            np.median([_min_curvature_radius(t.center) for t in tracks]),
            tracks[0].width,
        )
    radii = [stats[l][0] for l in LEVELS]
    widths = [stats[l][1] for l in LEVELS]
    assert radii[0] > radii[1] > radii[2], f"corner radius not ordered: {radii}"
    assert widths[0] > widths[1] > widths[2], f"width not ordered: {widths}"


def test_mixed_uses_every_level_and_matches_its_weights():
    picks = [resolve_difficulty(s, "mixed").name for s in range(2000)]
    share = {level: picks.count(level) / len(picks) for level in LEVELS}
    assert set(share) == set(LEVELS)
    assert share["easy"] == pytest.approx(0.40, abs=0.05)
    assert share["medium"] == pytest.approx(0.40, abs=0.05)
    assert share["hard"] == pytest.approx(0.20, abs=0.05)


def test_unknown_difficulty_is_rejected():
    with pytest.raises(ValueError):
        resolve_difficulty(0, "impossible")


def test_width_override_wins():
    track = make_track(3, TrackConfig(width_override=17.0), "hard")
    assert track.width == 17.0


def test_locate_returns_lateral_distance():
    track = make_track(3)
    _, dist = track.locate(track.center[::37])
    assert np.all(dist < 1e-6)

    offset = track.center[::37] + track.normal[::37] * 8.0
    _, dist = track.locate(offset)
    assert np.allclose(dist, 8.0, atol=0.2)


@pytest.mark.parametrize("level", LEVELS)
def test_car_starts_on_the_track(level):
    for seed in range(8):
        track = make_track(seed, difficulty=level)
        _, dist = track.locate(track.start_pos[None, :])
        assert dist[0] < track.half_width


def test_training_and_test_seeds_never_overlap():
    train = set(training_seeds(np.random.default_rng(0), 5000))
    held_out = set(holdout_seeds(200))
    assert not train & held_out
    assert min(held_out) >= TEST_SEED_BASE
    assert max(train) < TEST_SEED_BASE


def test_a_track_is_reproducible_from_its_seed():
    a = make_track(11, difficulty="hard")
    b = make_track(11, difficulty="hard")
    assert np.array_equal(a.center, b.center)
    assert a.start_index == b.start_index
