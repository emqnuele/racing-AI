import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from dataclasses import replace

import numpy as np
import pytest

from racing_ai.cli import USAGE, main
from racing_ai.config import Config, EpisodeConfig, EvolutionConfig
from racing_ai.trainer import Trainer


def test_bare_invocation_prints_the_menu(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    for command in ("train --live", "watch", "play", "demo", "eval"):
        assert command in out


def test_help_and_version(capsys):
    assert main(["--help"]) == 0
    assert main(["--version"]) == 0
    assert "racing-AI" in capsys.readouterr().out


def test_unknown_command_fails(capsys):
    assert main(["fly"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_every_advertised_command_is_routed():
    advertised = {line.strip().split()[1] for line in USAGE.splitlines()
                  if line.strip().startswith("racing ")}
    for command in advertised:
        with pytest.raises(SystemExit):
            main([command, "--help"])


def test_train_subcommand_reaches_the_trainer(capsys, tmp_path):
    assert main(["train", "--generations", "1", "--population", "12", "--tracks", "1",
                 "--validate-every", "0", "--save-every", "0", "--max-steps", "80",
                 "--no-plot", "--checkpoint-dir", str(tmp_path / "ckpt")]) == 0
    assert "topology 8 -> 16 -> 12 -> 8 -> 2" in capsys.readouterr().out


def _small_cfg():
    return replace(Config(), tracks_per_generation=2, validation_every=1, validation_tracks=2,
                   evolution=replace(EvolutionConfig(), population=16),
                   episode=replace(EpisodeConfig(), max_steps=80))


def test_live_training_advances_and_saves(tmp_path):
    from racing_ai.live import run_live_training

    trainer = Trainer.start(_small_cfg(), tmp_path / "ckpt", save_every=2)
    run_live_training(trainer, generations=3)
    assert trainer.generation == 3
    assert len(trainer.history) == 3
    assert (tmp_path / "ckpt" / "latest.npz").exists()
    assert (tmp_path / "ckpt" / "history.csv").exists()


def test_live_and_headless_training_agree(tmp_path):
    from racing_ai.live import run_live_training

    headless = Trainer.start(_small_cfg(), tmp_path / "a", save_every=0)
    for _ in range(3):
        headless.run_generation()

    live = Trainer.start(_small_cfg(), tmp_path / "b", save_every=0)
    run_live_training(live, generations=3)

    assert np.allclose(headless.evo.genomes, live.evo.genomes)
    assert headless.evo.best_fitness == live.evo.best_fitness
