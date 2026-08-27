from __future__ import annotations

import time

import numpy as np

from .env import Simulation, run_episode
from .render import (
    BG, DIM, GAS, KERB, NEG, PANEL, PANEL_LINE, POS, TEXT,
    TrackCanvas, draw_controls, draw_field, draw_network,
)
from .sensors import ray_offsets
from .trainer import Trainer


def draw_curve(pygame, screen, rect, history, font):
    pygame.draw.rect(screen, PANEL, rect)
    screen.blit(font.render("FITNESS", True, DIM), (rect.x + 16, rect.y + 8))
    if len(history) < 2:
        screen.blit(font.render("collecting...", True, DIM), (rect.x + 16, rect.centery))
        return

    plot = pygame.Rect(rect.x + 44, rect.y + 30, rect.width - 62, rect.height - 54)
    best = np.array([r["best"] for r in history], dtype=float)
    mean = np.array([r["mean"] for r in history], dtype=float)
    lo = float(min(best.min(), mean.min()))
    hi = float(max(best.max(), mean.max()))
    span = max(hi - lo, 1e-6)

    for frac in (0.0, 0.5, 1.0):
        y = plot.bottom - frac * plot.height
        pygame.draw.line(screen, PANEL_LINE, (plot.x, y), (plot.right, y), 1)
        label = font.render(f"{lo + frac * span:5.2f}", True, DIM)
        screen.blit(label, (rect.x + 6, y - label.get_height() // 2))

    xs = np.linspace(plot.x, plot.right, len(history))
    for series, colour in ((mean, NEG), (best, POS)):
        pts = [(x, plot.bottom - (v - lo) / span * plot.height) for x, v in zip(xs, series)]
        pygame.draw.lines(screen, colour, False, pts, 2)

    # i giri non sono fitness, quindi la validazione ha il suo asse 0..1 a destra
    val = [(x, min(1.0, max(0.0, r["val_laps"])))
           for x, r in zip(xs, history) if r.get("val_laps") is not None]
    if val:
        pts = [(x, plot.bottom - v * plot.height) for x, v in val]
        if len(pts) > 1:
            pygame.draw.lines(screen, GAS, False, pts, 2)
        for point in pts:
            pygame.draw.circle(screen, GAS, point, 3)

    for label, colour, dx in (("best", POS, 122), ("mean", NEG, 84),
                              ("unseen", GAS, 40)):
        screen.blit(font.render(label, True, colour), (plot.right - dx, rect.y + 8))


def draw_status(pygame, screen, rect, trainer, sim, track_i, fonts, speed_mult,
                paused, rendering, alive):
    font, bold, title = fonts["font"], fonts["bold"], fonts["title"]
    pygame.draw.rect(screen, PANEL, rect)
    screen.blit(title.render("TRAINING", True, TEXT), (rect.x + 16, rect.y + 12))
    last = trainer.history[-1] if trainer.history else None
    screen.blit(font.render(f"{trainer.cfg.evolution.population} cars  ·  train {trainer.cfg.difficulty}"
                            f"  ·  test {trainer.cfg.test_difficulty}", True, DIM),
                (rect.x + 16, rect.y + 42))

    rows = [
        ("generation", str(trainer.generation)),
        ("best ever", f"{trainer.evo.best_fitness:.3f}" if np.isfinite(trainer.evo.best_fitness) else "-"),
        ("last best", f"{last['best']:.3f}" if last else "-"),
        ("last mean", f"{last['mean']:.3f}" if last else "-"),
        ("unseen", f"{100 * last['val_finish_rate']:.0f}%"
         if last and last.get("val_finish_rate") is not None else "-"),
        ("alive", f"{int(alive.sum())} / {sim.net.size}"),
        ("track", f"{track_i + 1} / {trainer.cfg.tracks_per_generation}"),
        ("mutation", f"{trainer.evo.sigma:.3f}"),
        ("sim", ("paused" if paused else f"x{speed_mult}") + ("" if rendering else "  blind")),
    ]
    y = rect.y + 66
    for key, value in rows:
        screen.blit(font.render(key, True, DIM), (rect.x + 16, y))
        screen.blit(bold.render(value, True, TEXT), (rect.x + 116, y))
        y += 18


def run_live_training(
    trainer: Trainer,
    generations: int | None = None,
    size: tuple[int, int] = (1560, 900),
    panel_width: int = 560,
    camera: str = "fit",
    screenshot=None,
    screenshot_generation: int = 1,
) -> None:
    import pygame

    cfg = trainer.cfg
    pygame.init()
    pygame.display.set_caption("racing-AI · training")
    screen = pygame.display.set_mode(size)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo,dejavusansmono,consolas,monospace", 13)
    bold = pygame.font.SysFont("menlo,dejavusansmono,consolas,monospace", 13, bold=True)
    title = pygame.font.SysFont("menlo,dejavusansmono,consolas,monospace", 22, bold=True)
    fonts = {"font": font, "bold": bold, "title": title}

    viewport = pygame.Rect(0, 0, size[0] - panel_width, size[1])
    status_rect = pygame.Rect(viewport.width, 0, panel_width, 208)
    curve_rect = pygame.Rect(viewport.width, 212, panel_width, 174)
    net_rect = pygame.Rect(viewport.width, 390, panel_width, 310)
    ctrl_rect = pygame.Rect(viewport.width, 704, panel_width, size[1] - 704)

    offsets = ray_offsets(cfg.sensors)
    state = {"running": True, "paused": False, "speed": 4, "rendering": True,
             "skip": False, "camera": camera, "follow_leader": True, "focus": 0,
             "wrecks": True}

    def handle_events(sim):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                state["running"] = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    state["running"] = False
                elif event.key == pygame.K_SPACE:
                    state["paused"] = not state["paused"]
                elif event.key == pygame.K_f:
                    state["skip"] = True
                elif event.key == pygame.K_v:
                    state["rendering"] = not state["rendering"]
                elif event.key == pygame.K_c:
                    state["camera"] = "follow" if state["camera"] == "fit" else "fit"
                elif event.key == pygame.K_w:
                    state["wrecks"] = not state["wrecks"]
                elif event.key == pygame.K_l:
                    state["follow_leader"] = not state["follow_leader"]
                elif event.key == pygame.K_TAB and sim is not None:
                    state["follow_leader"] = False
                    state["focus"] = (state["focus"] + 1) % sim.net.size
                elif event.key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS):
                    state["speed"] = min(64, state["speed"] * 2)
                elif event.key in (pygame.K_DOWN, pygame.K_MINUS):
                    state["speed"] = max(1, state["speed"] // 2)

    def paint(sim, canvas, net, track_i, alive):
        focus = state["focus"]
        if state["follow_leader"] and sim.active.size:
            focus = int(sim.active[np.argmax(sim.progress[sim.active])])
            state["focus"] = focus

        screen.fill(BG)
        screen.set_clip(viewport)
        follow = sim.pos[focus] if state["camera"] == "follow" else None
        view = canvas.blit(screen, viewport, follow)
        alive = draw_field(pygame, screen, view, sim, cfg, focus, offsets,
                           state["wrecks"])
        screen.set_clip(None)

        draw_status(pygame, screen, status_rect, trainer, sim, track_i, fonts,
                    state["speed"], state["paused"], state["rendering"], alive)
        draw_curve(pygame, screen, curve_rect, trainer.history, font)
        trace = net.activations(sim.inputs[focus][None, :], np.array([focus]))
        draw_network(pygame, screen, net_rect, net, trace, cfg.layer_sizes, font, focus)
        draw_controls(pygame, screen, ctrl_rect, sim, focus, cfg, font, bold)

        help_text = ("SPACE pause · F skip generation · V blind mode · L leader · TAB car · "
                     "W hide wrecks · C camera · +/- speed · ESC stop")
        screen.blit(font.render(help_text, True, DIM), (16, size[1] - 24))
        pygame.display.flip()
        return alive

    def overlay(text):
        box = pygame.Rect(viewport.centerx - 190, viewport.centery - 26, 380, 52)
        pygame.draw.rect(screen, PANEL, box, border_radius=6)
        pygame.draw.rect(screen, PANEL_LINE, box, 1, border_radius=6)
        label = bold.render(text, True, KERB)
        screen.blit(label, (box.centerx - label.get_width() // 2,
                            box.centery - label.get_height() // 2))
        pygame.display.flip()

    target = None if generations is None else trainer.generation + generations

    while state["running"] and (target is None or trainer.generation < target):
        started = time.perf_counter()
        tracks = trainer.tracks()
        net = trainer.network()
        sim = Simulation(net, tracks[0], cfg)
        canvas = TrackCanvas(pygame, tracks[0], (viewport.width, viewport.height))
        state["skip"] = False
        alive = np.ones(net.size, dtype=bool)

        while state["running"] and not state["skip"] and state["rendering"] and not sim.done:
            handle_events(sim)
            if not state["paused"]:
                for _ in range(state["speed"]):
                    if sim.done:
                        break
                    sim.step()
            alive = paint(sim, canvas, net, 0, alive)
            clock.tick(60)

        if not state["running"]:
            break

        while not sim.done:
            sim.step()

        results = [sim.result()]
        for i, track in enumerate(tracks[1:], start=1):
            if state["rendering"]:
                overlay(f"scoring track {i + 1} / {len(tracks)}")
            handle_events(sim)
            results.append(run_episode(net, track, cfg))

        fitness = trainer.combine(results)
        trainer.record(fitness, results, time.perf_counter() - started)
        trainer.advance(fitness)

        if not state["rendering"]:
            handle_events(None)
            alive = np.zeros(net.size, dtype=bool)
            paint(sim, canvas, net, len(tracks) - 1, alive)

        if screenshot is not None and trainer.generation >= screenshot_generation:
            pygame.image.save(screen, str(screenshot))
            break

    pygame.quit()
    trainer.finish()
