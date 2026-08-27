from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import TrackConfig

# intervalli di seed disgiunti, così un tracciato di test non è mai comparso in training
TEST_SEED_BASE = 1_000_000

DIFFICULTIES = ("easy","medium", "hard")
MIXED_WEIGHTS = (0.40,0.40,0.20)


@dataclass(frozen=True)
class Difficulty:
    name: str
    width: float
    separation: float
    smoothing: float
    jitter: float
    bound: float
    spacing: float
    iterations: int
    max_points: int
    growth: float
    tolerance: float
    fillet: float
    min_radius_factor: float


PRESETS = {
    "easy": Difficulty("easy", 52.0, 178.0, 0.30, 1.0, 300.0, 28.0, 180, 84, 0.018, 82.0, 120.0, 1.15),
    "medium": Difficulty("medium", 40.0, 106.0, 0.22, 1.4, 340.0, 22.0, 240, 160, 0.026, 58.0, 50.0, 0.85),
    "hard": Difficulty("hard", 30.0, 70.0, 0.17, 1.7, 365.0, 19.0, 290, 215, 0.030, 46.0, 24.0, 0.62),
}


def resolve_difficulty(seed: int, mode: str) -> Difficulty:

    if mode in PRESETS:
        return PRESETS[mode]
    if mode != "mixed":
        raise ValueError(f"unknown difficulty: {mode!r}")

    pick = np.random.default_rng([seed, 0xD1FF]).random()
    cumulative = np.cumsum(MIXED_WEIGHTS)

    return PRESETS[DIFFICULTIES[int(np.searchsorted(cumulative, pick))]]


@dataclass(frozen=True)
class Track:
    seed: int
    difficulty: str
    width: float
    center: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    left: np.ndarray
    right: np.ndarray
    cum_len: np.ndarray
    length: float
    gates: np.ndarray
    wall_a: np.ndarray
    wall_b: np.ndarray
    start_index: int
    start_pos: np.ndarray
    start_heading: float

    @property
    def n_points(self) -> int:
        return self.center.shape[0]

    @property
    def half_width(self) -> float:
        return 0.5 * self.width

    def locate(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        diff = points[:, None, :] - self.center[None, :, :]
        d2 = np.einsum("pnk,pnk->pn", diff, diff)
        idx = np.argmin(d2, axis=1)

        n = self.n_points
        d_prev = _point_segment_distance(points, self.center[(idx - 1) % n], self.center[idx])
        d_next = _point_segment_distance(points, self.center[idx], self.center[(idx + 1) % n])
        return idx, np.minimum(d_prev, d_next)


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    ap = p - a
    denom = np.einsum("pk,pk->p", ab, ab)
    t = np.where(denom > 1e-12, np.einsum("pk,pk->p", ap, ab) / np.maximum(denom, 1e-12), 0.0)
    closest = a + np.clip(t, 0.0, 1.0)[:, None] * ab
    return np.linalg.norm(p - closest, axis=1)


def _resample_closed(points: np.ndarray, n_out: int) -> np.ndarray:
    closed = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])
    closed = closed[keep]
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    target = np.linspace(0.0, s[-1], n_out, endpoint=False)
    return np.stack([np.interp(target, s, closed[:, 0]),
                     np.interp(target, s, closed[:, 1])], axis=1)


def _min_curvature_radius(points: np.ndarray) -> float:
    a = np.roll(points, 1, axis=0)
    b = points
    c = np.roll(points, -1, axis=0)
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    cross = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                   - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    denom = 2.0 * cross
    return float(np.min(np.where(denom > 1e-9, ab * bc * ca / np.maximum(denom, 1e-9), np.inf)))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _min_separation(points: np.ndarray, exclude: int) -> float:
    n = len(points)
    d = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", d, d))
    i = np.arange(n)
    gap = np.abs(i[:, None] - i[None, :])
    return float(dist[np.minimum(gap, n - gap) > exclude].min())


def _fillet(vertices: np.ndarray, radius: float, step: float = 2.0) -> np.ndarray:
    n = len(vertices)
    out: list[np.ndarray] = []
    for i in range(n):
        a, b, c = vertices[i - 1], vertices[i], vertices[(i + 1) % n]
        u, v = _unit(a - b), _unit(c - b)
        theta = float(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0)))
        if theta < 1e-3 or theta > np.pi - 1e-3:
            out.append(b[None, :])
            continue

        half = 0.5 * theta
        reach = min(radius / np.tan(half),
                    0.45 * float(np.linalg.norm(a - b)),
                    0.45 * float(np.linalg.norm(c - b)))
        r_eff = reach * np.tan(half)
        centre = b + _unit(u + v) * (r_eff / np.sin(half))

        start = b + u * reach
        end = b + v * reach
        a0 = np.arctan2(*(start - centre)[::-1])
        a1 = np.arctan2(*(end - centre)[::-1])
        sweep = (a1 - a0 + np.pi) % (2.0 * np.pi) - np.pi
        # campiona più fitto del ricampionamento finale, altrimenti la polilinea
        # viene fuori come una catena di spigoli invece che come un arco
        t = np.linspace(0.0, 1.0, max(4, int(abs(sweep) * r_eff / step)))
        out.append(centre + r_eff * np.stack([np.cos(a0 + sweep * t),
                                              np.sin(a0 + sweep * t)], axis=1))
    return np.vstack(out)


def _self_intersects(loop: np.ndarray) -> bool:
    a1 = loop
    a2 = np.roll(loop, -1, axis=0)
    n = len(loop)

    def side(p, q, r):
        return np.sign((q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1])
                       - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0]))

    for offset in range(2, n - 1):
        b1, b2 = np.roll(a1, -offset, axis=0), np.roll(a2, -offset, axis=0)
        d1, d2 = side(b1, b2, a1), side(b1, b2, a2)
        d3, d4 = side(a1, a2, b1), side(a1, a2, b2)
        if ((d1 * d2 < 0) & (d3 * d4 < 0)).any():
            return True
    return False


def _walls_are_valid(center: np.ndarray, half: float, stride: int = 3) -> bool:
    # dentro un tornante la distanza fra indici non dice più niente, quindi
    # controlla gli incroci sui muri stessi
    t = _unit(np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0))
    normal = np.stack([-t[:, 1], t[:, 0]], axis=1)
    for sign in (1.0, -1.0):
        if _self_intersects((center + sign * half * normal)[::stride]):
            return False
    return True


def _grow_route(seed: int, level: Difficulty) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n0 = 24
    theta = np.linspace(0.0, 2.0 * np.pi, n0, endpoint=False)
    radius = level.bound * 0.22 * rng.uniform(0.75, 1.25, n0)
    p = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)

    aspect = float(rng.uniform(1.0, 1.35))
    rot = float(rng.uniform(0.0, np.pi))
    axes = np.array([level.bound * aspect, level.bound / aspect])
    rmat = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])

    for _ in range(level.iterations):
        n = len(p)
        prev, nxt = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)
        p = p + level.smoothing * (0.5 * (prev + nxt) - p)

        edge = nxt - p
        length = np.linalg.norm(edge, axis=1, keepdims=True)
        pull = 0.4 * (length - level.spacing) / np.maximum(length, 1e-9) * edge
        p = p + pull - np.roll(pull, 1, axis=0)

        d = p[:, None, :] - p[None, :, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", d, d)) + 1e-9
        i = np.arange(n)
        gap = np.abs(i[:, None] - i[None, :])
        close = (np.minimum(gap, n - gap) > 4) & (dist < level.separation)
        if close.any():
            w = np.where(close, (level.separation - dist) / level.separation, 0.0)
            push = np.einsum("ij,ijk->ik", w, d / dist[:, :, None])
            p = p + 0.38 * level.separation * push / np.maximum(w.sum(1), 1)[:, None]

        p = p + rng.normal(0.0, level.jitter, size=p.shape)

        local = p @ rmat
        over = np.linalg.norm(local / axes, axis=1, keepdims=True)
        p = np.where(over > 1.0, (local / over) @ rmat.T, p)

        # senza un perimetro che continua a crescere oltre lo spazio, il giro non si piega mai
        if n < level.max_points:
            count = min(max(1, int(n * level.growth)), level.max_points - n)
            pick = rng.choice(n, size=count, replace=False)
            p = np.insert(p, pick + 1, 0.5 * (p[pick] + p[(pick + 1) % n]), axis=0)
    return p


def _douglas_peucker(points: np.ndarray, tol: float) -> np.ndarray:
    keep = np.zeros(len(points), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        seg = points[b] - points[a]
        norm = float(np.linalg.norm(seg))
        rel = points[a + 1:b] - points[a]
        if norm < 1e-9:
            d = np.linalg.norm(rel, axis=1)
        else:
            d = np.abs(rel[:, 0] * seg[1] - rel[:, 1] * seg[0]) / norm
        k = int(np.argmax(d))
        if d[k] > tol:
            keep[a + 1 + k] = True
            stack.append((a, a + 1 + k))
            stack.append((a + 1 + k, b))
    return points[keep]


def _simplify_closed(points: np.ndarray, tol: float) -> np.ndarray:
    # sono i tratti dolci collassati in un unico segmento a creare i rettilinei
    rolled = np.roll(points, -int(np.argmax(points[:, 0])), axis=0)
    half = len(rolled) // 2
    a = _douglas_peucker(rolled[:half + 1], tol)
    b = _douglas_peucker(np.vstack([rolled[half:], rolled[:1]]), tol)
    return np.vstack([a[:-1], b[:-1]])


def make_track(seed: int, cfg: TrackConfig | None = None,
               difficulty: str = "medium") -> Track:
    cfg = cfg or TrackConfig()
    level = resolve_difficulty(seed, difficulty)
    width = level.width if cfg.width_override <= 0 else cfg.width_override
    half = 0.5 * width
    rng = np.random.default_rng(seed)
    floor = width * level.min_radius_factor
    # è il raccordo più largo ad aprire davvero la curva; la tolleranza decide
    # solo quanto del percorso sopravvive come rettilineo
    schedule = ((1.0, 1.0), (1.0, 1.5), (0.85, 2.1), (0.85, 3.0), (0.7, 4.2), (0.7, 6.0))
    center = None
    best = None
    best_radius = -np.inf
    for attempt in range(4):
        route = _grow_route(seed * 64 + attempt, level)
        for tol_scale, fillet_scale in schedule:
            poly = _simplify_closed(route, level.tolerance * tol_scale)
            candidate = _resample_closed(
                _fillet(poly, level.fillet * fillet_scale), cfg.n_points)
            if not _walls_are_valid(candidate, half):
                continue
            radius = _min_curvature_radius(candidate)
            if radius > best_radius:
                best, best_radius = candidate, radius
            if radius > floor:
                center = candidate
                break
        if center is not None:
            break
    if center is None:
        if best is None:
            raise RuntimeError(f"no drivable {level.name} track for seed {seed}")
        center = best

    if rng.random() < 0.5:
        center = center[::-1].copy()

    tangent = _unit(np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0))
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    left = center + normal * half
    right = center - normal * half

    seg = np.linalg.norm(np.diff(np.vstack([center, center[:1]]), axis=0), axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg)[:-1]])

    stride = max(1, cfg.wall_stride)
    coarse_left, coarse_right = left[::stride], right[::stride]

    start_index = int(rng.integers(0, cfg.n_points))
    lat = rng.uniform(-1.0, 1.0) * cfg.start_lateral_jitter * half
    start_heading = float(np.arctan2(tangent[start_index, 1], tangent[start_index, 0]))
    start_heading += np.deg2rad(rng.uniform(-1.0, 1.0) * cfg.start_heading_jitter_deg)

    return Track(
        seed=seed,
        difficulty=level.name,
        width=width,
        center=center,
        tangent=tangent,
        normal=normal,
        left=left,
        right=right,
        cum_len=cum_len,
        length=float(np.sum(seg)),
        gates=np.linspace(0, cfg.n_points, cfg.n_gates, endpoint=False).astype(np.int64),
        wall_a=np.vstack([coarse_left, coarse_right]),
        wall_b=np.vstack([np.roll(coarse_left, -1, axis=0), np.roll(coarse_right, -1, axis=0)]),
        start_index=start_index,
        start_pos=center[start_index] + normal[start_index] * lat,
        start_heading=start_heading,
    )


def training_seeds(rng: np.random.Generator, count: int) -> list[int]:
    return [int(s) for s in rng.integers(0, TEST_SEED_BASE, size=count)]


def holdout_seeds(count: int, offset: int = 0) -> list[int]:
    return [TEST_SEED_BASE + offset + i for i in range(count)]
