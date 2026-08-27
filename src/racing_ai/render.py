from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .env import Simulation
from .network import PopulationNetwork
from .sensors import ray_offsets
from .track import Track, make_track

BG = (13, 15, 20)
PANEL = (20, 23, 31)
PANEL_LINE = (38, 43, 55)
ASPHALT = (44, 48, 58)
KERB = (226, 232, 242)
GATE = (62, 70, 88)
GATE_DONE = (64, 170, 120)
CAR_FOCUS = (255, 82, 82)
CAR_ALIVE = (96, 158, 232)
CAR_DEAD = (58, 62, 74)
RAY = (72, 190, 140)
RAY_HIT = (255, 186, 84)
TEXT = (228, 233, 243)
DIM = (128, 137, 157)
POS = (255, 176, 74)
NEG = (86, 166, 255)
GAS = (74, 208, 130)
BRAKE = (240, 84, 84)

FOLLOW_SCALE = 2.0


def blend(a, b, t: float):
    t = max(0.0, min(1.0, t))
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def signed_colour(value: float, bg=PANEL) -> tuple[int, int, int]:
    return blend(bg, POS if value >= 0 else NEG, abs(value))


@dataclass
class View:
    scale: float
    offset_x: float
    offset_y: float

    def to_screen(self, points: np.ndarray) -> np.ndarray:
        out = points * np.array([self.scale, -self.scale])
        out[..., 0] += self.offset_x
        out[..., 1] += self.offset_y
        return out

    def shifted(self, dx: float, dy: float) -> "View":
        return View(self.scale, self.offset_x + dx, self.offset_y + dy)


def _view_for(scale: float, world_center: np.ndarray, screen_center: tuple[float, float]) -> View:
    return View(
        scale=scale,
        offset_x=screen_center[0] - world_center[0] * scale,
        offset_y=screen_center[1] + world_center[1] * scale,
    )


class TrackCanvas:
    def __init__(self, pygame, track: Track, viewport: tuple[int, int], margin: int = 30):
        pts = np.vstack([track.left, track.right])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        pad = track.width
        lo, hi = lo - pad, hi + pad
        span = np.maximum(hi - lo, 1e-6)
        self.world_center = (lo + hi) * 0.5

        big_size = (int(span[0] * FOLLOW_SCALE), int(span[1] * FOLLOW_SCALE))
        self.big_view = _view_for(FOLLOW_SCALE, self.world_center,
                                  (big_size[0] * 0.5, big_size[1] * 0.5))
        self.big = self._paint(pygame, big_size, track, self.big_view)

        fit_scale = min((viewport[0] - 2 * margin) / span[0],
                        (viewport[1] - 2 * margin) / span[1])
        fit_size = (max(1, int(span[0] * fit_scale)), max(1, int(span[1] * fit_scale)))
        self.fit = pygame.transform.smoothscale(self.big, fit_size)
        self.fit_view = _view_for(fit_scale, self.world_center,
                                  (fit_size[0] * 0.5, fit_size[1] * 0.5))

    @staticmethod
    def _paint(pygame, size, track: Track, view: View):
        surface = pygame.Surface(size)
        surface.fill(BG)
        left = view.to_screen(track.left)
        right = view.to_screen(track.right)
        n = len(left)
        for i in range(n):
            j = (i + 1) % n
            pygame.draw.polygon(surface, ASPHALT, [left[i], left[j], right[j], right[i]])
        for g in track.gates:
            pygame.draw.line(surface, GATE, left[g], right[g], 1)
        pygame.draw.lines(surface, KERB, True, [tuple(p) for p in left], 2)
        pygame.draw.lines(surface, KERB, True, [tuple(p) for p in right], 2)
        start = view.to_screen(track.center[track.start_index][None, :])[0]
        pygame.draw.circle(surface, GATE_DONE, start, 5, 2)
        return surface

    def blit(self, screen, viewport, follow: np.ndarray | None):
        cx, cy = viewport.centerx, viewport.centery
        if follow is None:
            pos = (cx - self.fit.get_width() // 2, cy - self.fit.get_height() // 2)
            screen.blit(self.fit, pos)
            return self.fit_view.shifted(pos[0], pos[1])

        car = self.big_view.to_screen(follow[None, :])[0]
        pos = (int(cx - car[0]), int(cy - car[1]))
        screen.blit(self.big, pos)
        return self.big_view.shifted(pos[0], pos[1])


def draw_field(pygame, screen, view, sim, cfg, focus: int, offsets,
               show_wrecks: bool = True) -> np.ndarray:
    alive = np.zeros(sim.net.size, dtype=bool)
    alive[sim.active] = True

    for i in np.argsort(alive.astype(int)):
        if i == focus or (not alive[i] and not show_wrecks):
            continue
        pygame.draw.polygon(
            screen, CAR_ALIVE if alive[i] else CAR_DEAD,
            view.to_screen(car_corners(sim.pos[i], float(sim.heading[i]),
                                       cfg.car.length, cfg.car.width)))

    pos, heading = sim.pos[focus], float(sim.heading[focus])
    if alive[focus]:
        origin = view.to_screen(pos[None, :])[0]
        for k, off in enumerate(offsets):
            d = float(sim.rays[focus, k])
            ang = heading + off
            tip = view.to_screen((pos + np.array([math.cos(ang), math.sin(ang)]) * d)[None, :])[0]
            hit = d < cfg.sensors.max_range - 1e-6
            pygame.draw.line(screen, RAY_HIT if hit else RAY, origin, tip, 1)
            if hit:
                pygame.draw.circle(screen, RAY_HIT, tip, 3)

    pygame.draw.polygon(
        screen, CAR_FOCUS if alive[focus] else CAR_DEAD,
        view.to_screen(car_corners(pos, heading, cfg.car.length, cfg.car.width)))
    return alive


def car_corners(pos: np.ndarray, heading: float, length: float, width: float) -> np.ndarray:
    c, s = math.cos(heading), math.sin(heading)
    hl, hw = length * 0.5, width * 0.5
    local = np.array([[hl, -hw], [hl, hw], [-hl, hw], [-hl, -hw]])
    return local @ np.array([[c, s], [-s, c]]) + pos


def draw_network(pygame, screen, rect, net, trace, layer_sizes, font, focus: int):
    pygame.draw.rect(screen, PANEL, rect)
    label = font.render("NEURAL NETWORK", True, DIM)
    screen.blit(label, (rect.x + 16, rect.y + 8))

    top = rect.y + 30
    height = rect.height - 44
    xs = np.linspace(rect.x + 44, rect.right - 68, len(layer_sizes))
    radius = max(4, min(11, int(height / (2 * max(layer_sizes) + 2))))

    coords = []
    for size, x in zip(layer_sizes, xs):
        gap = height / size
        ys = top + gap * (np.arange(size) + 0.5)
        coords.append(np.stack([np.full(size, x), ys], axis=1))

    for li in range(len(layer_sizes) - 1):
        acts = trace[li][0]
        w = net.layer_weights(li, focus)
        signal = acts[:, None] * w
        peak = float(np.abs(signal).max())
        if peak < 1e-9:
            continue
        strength = np.abs(signal) / peak
        a, b = coords[li], coords[li + 1]
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                s = float(strength[i, j])
                if s < 0.06:
                    continue
                colour = blend(PANEL, POS if signal[i, j] >= 0 else NEG, s * 0.85)
                pygame.draw.line(screen, colour, a[i], b[j], 1)

    for li, size in enumerate(layer_sizes):
        acts = trace[li][0]
        for i in range(size):
            centre = coords[li][i]
            value = float(acts[i])
            pygame.draw.circle(screen, signed_colour(value), centre, radius)
            pygame.draw.circle(screen, blend(PANEL, TEXT, 0.25), centre, radius, 1)

    names_in = [f"s{i + 1}" for i in range(layer_sizes[0] - 1)] + ["spd"]
    for i, name in enumerate(names_in):
        text = font.render(name, True, DIM)
        screen.blit(text, (coords[0][i][0] - radius - 8 - text.get_width(),
                           coords[0][i][1] - text.get_height() // 2))
    for i, name in enumerate(["steer", "gas"]):
        text = font.render(name, True, DIM)
        screen.blit(text, (coords[-1][i][0] + radius + 8,
                           coords[-1][i][1] - text.get_height() // 2))


def _bar(pygame, screen, rect, value, colour, centred: bool):
    pygame.draw.rect(screen, blend(PANEL, PANEL_LINE, 1.0), rect, border_radius=3)
    if centred:
        mid = rect.centerx
        width = int(abs(value) * rect.width * 0.5)
        if width > 0:
            x = mid if value >= 0 else mid - width
            pygame.draw.rect(screen, colour, pygame.Rect(x, rect.y, width, rect.height),
                             border_radius=3)
        pygame.draw.line(screen, blend(PANEL, TEXT, 0.4), (mid, rect.y), (mid, rect.bottom), 1)
    else:
        width = int(max(0.0, value) * rect.width)
        if width > 0:
            pygame.draw.rect(screen, colour, pygame.Rect(rect.x, rect.y, width, rect.height),
                             border_radius=3)


def draw_controls(pygame, screen, rect, sim, focus, cfg, font, bold):
    pygame.draw.rect(screen, PANEL, rect)
    screen.blit(font.render("SENSORS", True, DIM), (rect.x + 16, rect.y + 8))

    n_rays = cfg.sensors.n_rays
    inputs = sim.inputs[focus]
    x = rect.x + 16
    width = rect.width - 32
    slot = width / n_rays
    base = rect.y + 30
    for i in range(n_rays):
        value = float(inputs[i])
        bar = pygame.Rect(int(x + i * slot) + 2, base, int(slot) - 5, 46)
        pygame.draw.rect(screen, blend(PANEL, PANEL_LINE, 1.0), bar, border_radius=3)
        h = int(value * bar.height)
        if h > 0:
            pygame.draw.rect(screen, blend(RAY_HIT, RAY, value),
                             pygame.Rect(bar.x, bar.bottom - h, bar.width, h), border_radius=3)
        screen.blit(font.render(f"{i + 1}", True, DIM), (bar.x + bar.width // 2 - 3, bar.bottom + 3))

    y = base + 72
    steer, throttle = float(sim.controls[focus, 0]), float(sim.controls[focus, 1])

    screen.blit(font.render("STEER", True, DIM), (rect.x + 16, y))
    left = bold.render("<", True, POS if steer < -0.05 else PANEL_LINE)
    right = bold.render(">", True, POS if steer > 0.05 else PANEL_LINE)
    screen.blit(left, (rect.x + 76, y - 2))
    screen.blit(right, (rect.right - 88, y - 2))
    screen.blit(font.render(f"{steer:+.2f}", True, TEXT), (rect.right - 62, y))
    _bar(pygame, screen, pygame.Rect(rect.x + 96, y + 1, rect.width - 196, 12),
         steer, signed_colour(steer, PANEL_LINE), True)

    y += 30
    screen.blit(font.render("PEDAL", True, DIM), (rect.x + 16, y))
    gas = bold.render("GAS", True, GAS if throttle > 0.05 else PANEL_LINE)
    brake = bold.render("BRK", True, BRAKE if throttle < -0.05 else PANEL_LINE)
    screen.blit(brake, (rect.x + 64, y - 2))
    screen.blit(gas, (rect.right - 106, y - 2))
    screen.blit(font.render(f"{throttle:+.2f}", True, TEXT), (rect.right - 62, y + 18))
    _bar(pygame, screen, pygame.Rect(rect.x + 110, y + 1, rect.width - 226, 12),
         throttle, GAS if throttle >= 0 else BRAKE, True)

    y += 40
    speed = float(sim.speed[focus])
    screen.blit(font.render("SPEED", True, DIM), (rect.x + 16, y))
    screen.blit(font.render(f"{speed:5.1f}", True, TEXT), (rect.right - 62, y))
    _bar(pygame, screen, pygame.Rect(rect.x + 96, y + 1, rect.width - 196, 12),
         speed / cfg.car.max_speed, blend(GAS, BRAKE, speed / cfg.car.max_speed), False)


class HumanDriver:
    def __init__(self):
        self.size = 1
        self.steer = 0.0
        self.throttle = 0.0

    def forward(self, x, sel=None):
        out = np.array([[self.steer, self.throttle]], dtype=np.float32)
        return np.repeat(out, x.shape[0], axis=0)


def _draw_header(pygame, screen, rect, sim, track, focus, cfg, fonts, camera, speed_mult, human):
    font, bold, title = fonts["font"], fonts["bold"], fonts["title"]
    pygame.draw.rect(screen, PANEL, rect)
    screen.blit(title.render("racing-AI", True, TEXT), (rect.x + 16, rect.y + 12))
    screen.blit(font.render(f"seed {track.seed}  ·  {track.difficulty}  ·  held out", True, DIM),
                (rect.x + 16, rect.y + 42))

    alive = np.zeros(sim.net.size, dtype=bool)
    alive[sim.active] = True
    laps = sim.progress / track.length
    outcome = "driving" if alive[focus] else ("finished" if sim.finished[focus] else "crashed")

    rows = [
        ("car", f"#{focus}" + ("  (you)" if human else "")),
        ("state", outcome),
        ("laps", f"{laps[focus]:.3f}"),
        ("rank", f"{1 + int(np.sum(laps > laps[focus]))} / {sim.net.size}"),
        ("alive", f"{int(alive.sum())} / {sim.net.size}"),
        ("step", f"{sim.step_count} / {cfg.episode.max_steps}"),
        ("camera", camera),
        ("sim", f"x{speed_mult}"),
    ]
    y = rect.y + 68
    for key, value in rows:
        screen.blit(font.render(key, True, DIM), (rect.x + 16, y))
        screen.blit(bold.render(value, True, TEXT), (rect.x + 96, y))
        y += 19


def run_viewer(
    genomes: np.ndarray,
    cfg: Config,
    seeds: list[int],
    size: tuple[int, int] = (1560, 900),
    screenshot: Path | None = None,
    screenshot_steps: int = 300,
    human: bool = False,
    panel_width: int = 560,
    camera: str = "fit",
    difficulty: str | None = None,
) -> None:
    import pygame

    pygame.init()
    pygame.display.set_caption("racing-AI")
    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo,dejavusansmono,consolas,monospace", 13)
    bold = pygame.font.SysFont("menlo,dejavusansmono,consolas,monospace", 13, bold=True)
    title = pygame.font.SysFont("menlo,dejavusansmono,consolas,monospace", 22, bold=True)
    fonts = {"font": font, "bold": bold, "title": title}

    viewport = pygame.Rect(0, 0, size[0] - panel_width, size[1])
    header_rect = pygame.Rect(viewport.width, 0, panel_width, 236)
    net_rect = pygame.Rect(viewport.width, 240, panel_width, 388)
    ctrl_rect = pygame.Rect(viewport.width, 632, panel_width, size[1] - 632)

    offsets = ray_offsets(cfg.sensors)
    driver = HumanDriver() if human else None
    net = PopulationNetwork(genomes, cfg.layer_sizes)
    controller = driver if human else net

    seed_i = 0
    level = difficulty or cfg.test_difficulty
    track = make_track(seeds[seed_i], cfg.track, level)
    canvas = TrackCanvas(pygame, track, (viewport.width, viewport.height))
    sim = Simulation(controller, track, cfg)

    focus = 0
    speed_mult = 1
    paused = False
    running = True
    show_wrecks = True

    def reset(new_track: Track | None = None):
        nonlocal track, canvas, sim
        if new_track is not None:
            track = new_track
            canvas = TrackCanvas(pygame, track, (viewport.width, viewport.height))
        sim = Simulation(controller, track, cfg)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    reset()
                elif event.key in (pygame.K_n, pygame.K_p):
                    step = 1 if event.key == pygame.K_n else -1
                    seed_i = (seed_i + step) % len(seeds)
                    reset(make_track(seeds[seed_i], cfg.track, level))
                elif event.key == pygame.K_c:
                    camera = "follow" if camera == "fit" else "fit"
                elif event.key == pygame.K_w:
                    show_wrecks = not show_wrecks
                elif event.key == pygame.K_TAB:
                    focus = (focus + 1) % sim.net.size
                elif event.key == pygame.K_l and sim.active.size:
                    focus = int(sim.active[np.argmax(sim.progress[sim.active])])
                elif event.key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS):
                    speed_mult = min(32, speed_mult * 2)
                elif event.key in (pygame.K_DOWN, pygame.K_MINUS):
                    speed_mult = max(1, speed_mult // 2)

        if human:
            keys = pygame.key.get_pressed()
            driver.steer = float(keys[pygame.K_RIGHT]) - float(keys[pygame.K_LEFT])
            driver.throttle = float(keys[pygame.K_UP]) - float(keys[pygame.K_DOWN])

        if not paused:
            for _ in range(speed_mult):
                if sim.done:
                    break
                sim.step()

        screen.fill(BG)
        # con la camera che insegue, la superficie del tracciato è più grande del viewport
        screen.set_clip(viewport)
        follow = sim.pos[focus] if camera == "follow" else None
        view = canvas.blit(screen, viewport, follow)

        draw_field(pygame, screen, view, sim, cfg, focus, offsets, show_wrecks)

        screen.set_clip(None)
        _draw_header(pygame, screen, header_rect, sim, track, focus, cfg, fonts,
                     camera, speed_mult, human)
        if human:
            pygame.draw.rect(screen, PANEL, net_rect)
            screen.blit(font.render("NEURAL NETWORK", True, DIM), (net_rect.x + 16, net_rect.y + 8))
            screen.blit(title.render("HUMAN DRIVER", True, DIM),
                        (net_rect.x + 16, net_rect.centery - 12))
        else:
            trace = net.activations(sim.inputs[focus][None, :], np.array([focus]))
            draw_network(pygame, screen, net_rect, net, trace, cfg.layer_sizes, font, focus)
        draw_controls(pygame, screen, ctrl_rect, sim, focus, cfg, font, bold)

        help_text = ("arrows drive · " if human else "TAB car · L leader · W hide wrecks · ")
        help_text += "SPACE pause · N/P track · R restart · C camera · +/- speed · ESC quit"
        screen.blit(font.render(help_text, True, DIM), (16, size[1] - 24))

        pygame.display.flip()

        if screenshot is not None and sim.step_count >= screenshot_steps:
            pygame.image.save(screen, str(screenshot))
            running = False

        clock.tick(60)

    pygame.quit()
