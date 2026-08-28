import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest

from racing_ai.config import Config
from racing_ai.env import Simulation
from racing_ai.network import PopulationNetwork, random_genomes
from racing_ai.render import TrackCanvas, View, car_corners, run_viewer, signed_colour
from racing_ai.track import make_track

pygame = pytest.importorskip("pygame")


def test_view_maps_world_to_screen_with_flipped_y():
    view = View(scale=2.0, offset_x=100.0, offset_y=50.0)
    out = view.to_screen(np.array([[0.0, 0.0], [10.0, 10.0]]))
    assert np.allclose(out[0], [100.0, 50.0])
    assert np.allclose(out[1], [120.0, 30.0])


def test_car_corners_form_the_right_rectangle():
    corners = car_corners(np.zeros(2), 0.0, 10.0, 4.0)
    assert corners.shape == (4, 2)
    assert np.isclose(np.linalg.norm(corners[0] - corners[1]), 4.0)
    assert np.isclose(np.linalg.norm(corners[1] - corners[2]), 10.0)

    turned = car_corners(np.zeros(2), np.pi / 2, 10.0, 4.0)
    assert np.isclose(turned[0][1], 5.0, atol=1e-6)


def test_signed_colour_splits_positive_and_negative():
    assert signed_colour(1.0) != signed_colour(-1.0)
    assert signed_colour(0.0) == signed_colour(-0.0)


def test_track_canvas_renders_both_cameras():
    pygame.init()
    pygame.display.set_mode((320, 240))
    track = make_track(7)
    canvas = TrackCanvas(pygame, track, (600, 500))
    screen = pygame.Surface((900, 500))
    viewport = pygame.Rect(0, 0, 600, 500)

    fit = canvas.blit(screen, viewport, None)
    follow = canvas.blit(screen, viewport, track.start_pos)
    assert fit.scale < follow.scale

    # con la camera che insegue, la macchina seguita deve finire al centro del viewport
    centre = follow.to_screen(track.start_pos[None, :])[0]
    assert np.allclose(centre, [viewport.centerx, viewport.centery], atol=1.5)
    pygame.quit()


def test_activations_line_up_with_the_drawn_layers():
    cfg = Config()
    net = PopulationNetwork(random_genomes(np.random.default_rng(0), 4, cfg.layer_sizes),
                            cfg.layer_sizes)
    sim = Simulation(net, make_track(7, cfg.track), cfg)
    sim.step()
    trace = net.activations(sim.inputs[2][None, :], np.array([2]))
    assert [t.shape[1] for t in trace] == list(cfg.layer_sizes)


def test_viewer_runs_headless_and_writes_a_screenshot(tmp_path):
    cfg = Config()
    genomes = random_genomes(np.random.default_rng(1), 6, cfg.layer_sizes)
    shot = tmp_path / "frame.png"
    run_viewer(genomes, cfg, [1_000_000], screenshot=shot, screenshot_steps=20)
    assert shot.exists() and shot.stat().st_size > 0


def test_human_mode_runs_headless(tmp_path):
    cfg = Config()
    genomes = random_genomes(np.random.default_rng(2), 1, cfg.layer_sizes)
    shot = tmp_path / "human.png"
    run_viewer(genomes, cfg, [1_000_000], human=True, screenshot=shot, screenshot_steps=10)
    assert shot.exists()


def test_human_driver_reports_its_controls():
    from racing_ai.render import HumanDriver

    driver = HumanDriver()
    driver.steer, driver.throttle = -1.0, 0.5
    out = driver.forward(np.zeros((3, 8)))
    assert out.shape == (3, 2)
    assert np.allclose(out[:, 0], -1.0)
    assert np.allclose(out[:, 1], 0.5)


def test_the_track_never_bleeds_into_the_side_panel(tmp_path):
    from racing_ai.render import ASPHALT

    cfg = Config()
    genomes = random_genomes(np.random.default_rng(3), 4, cfg.layer_sizes)
    shot = tmp_path / "follow.png"
    run_viewer(genomes, cfg, [1_000_000], size=(1560, 900), camera="follow",
               screenshot=shot, screenshot_steps=30)

    surface = pygame.image.load(str(shot))
    panel = pygame.Rect(1000, 0, 560, 900)
    for x in range(panel.x, panel.right, 7):
        for y in range(0, panel.bottom, 7):
            assert surface.get_at((x, y))[:3] != ASPHALT


def test_wrecks_are_hidden_unless_asked_for():
    from racing_ai.render import CAR_DEAD, View, draw_field

    pygame.init()
    pygame.display.set_mode((320, 240))
    cfg = Config()
    net = PopulationNetwork(random_genomes(np.random.default_rng(5), 24, cfg.layer_sizes),
                            cfg.layer_sizes)
    track = make_track(1_000_000, cfg.track, "hard")
    sim = Simulation(net, track, cfg)
    while not sim.done:
        sim.step()
    assert sim.active.size == 0

    view = View(scale=1.0, offset_x=400.0, offset_y=400.0)
    offsets = np.zeros(cfg.sensors.n_rays)

    def wreck_pixels(show):
        surface = pygame.Surface((800, 800))
        surface.fill((0, 0, 0))
        draw_field(pygame, surface, view, sim, cfg, 0, offsets, show)
        return sum(surface.get_at((x, y))[:3] == CAR_DEAD
                   for x in range(0, 800, 2) for y in range(0, 800, 2))

    assert wreck_pixels(False) < wreck_pixels(True)
    pygame.quit()


def test_arrow_keys_steer_the_way_they_point_on_screen():
    from racing_ai.render import HumanDriver

    cfg = Config()
    track = make_track(1_000_000, cfg.track)
    view = View(scale=1.0, offset_x=0.0, offset_y=0.0)

    def screen_turn(left, right):
        driver = HumanDriver()
        driver.apply_keys(left, right, up=True, down=False)
        sim = Simulation(driver, track, cfg)
        start = view.to_screen(np.array([[np.cos(sim.heading[0]), np.sin(sim.heading[0])]]))[0]
        for _ in range(15):
            sim.step()
        end = view.to_screen(np.array([[np.cos(sim.heading[0]), np.sin(sim.heading[0])]]))[0]
        # in pixel la y cresce verso il basso, quindi il prodotto vettoriale è
        # positivo quando la macchina ruota in senso orario, cioè verso destra
        return start[0] * end[1] - start[1] * end[0]

    assert screen_turn(left=False, right=True) > 0
    assert screen_turn(left=True, right=False) < 0
    assert screen_turn(left=True, right=True) == 0


def test_human_keys_map_to_steer_and_throttle():
    from racing_ai.render import HumanDriver

    driver = HumanDriver()
    driver.apply_keys(left=False, right=False, up=True, down=False)
    assert (driver.steer, driver.throttle) == (0.0, 1.0)
    driver.apply_keys(left=False, right=False, up=False, down=True)
    assert (driver.steer, driver.throttle) == (0.0, -1.0)
