"""Tests for scripts/play.py CLI argument parsing and visualizer configuration."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
from isaaclab_visualizers.viser import ViserVisualizerCfg
from play import _configure_visualizers, _parse_args


def test_play_defaults_to_viser_and_port_9080():
    args, _ = _parse_args([])
    assert args.visualizer == ["viser"]
    assert args.viser_port == 9080


def test_play_custom_viser_port():
    args, _ = _parse_args(["--viser-port", "9123"])
    assert args.visualizer == ["viser"]
    assert args.viser_port == 9123


def test_play_explicit_visualizer():
    args, _ = _parse_args(["--visualizer", "newton"])
    assert args.visualizer == ["newton"]


def test_play_headless_does_not_force_viser():
    args, _ = _parse_args(["--headless"])
    assert args.visualizer is None


def test_configure_visualizers_sets_viser_port():
    env_cfg = SimpleNamespace(sim=SimpleNamespace(visualizer_cfgs=[]))
    args = argparse.Namespace(visualizer=["viser"], viser_port=9080, collision_only=False)

    _configure_visualizers(env_cfg, args)

    assert len(env_cfg.sim.visualizer_cfgs) == 1
    viser_cfg = env_cfg.sim.visualizer_cfgs[0]
    assert isinstance(viser_cfg, ViserVisualizerCfg)
    assert viser_cfg.port == 9080


def test_configure_visualizers_preserves_custom_port():
    env_cfg = SimpleNamespace(sim=SimpleNamespace(visualizer_cfgs=[]))
    args = argparse.Namespace(visualizer=["viser"], viser_port=9555, collision_only=False)

    _configure_visualizers(env_cfg, args)

    viser_cfg = env_cfg.sim.visualizer_cfgs[0]
    assert viser_cfg.port == 9555


def test_configure_visualizers_with_collision_only():
    env_cfg = SimpleNamespace(sim=SimpleNamespace(visualizer_cfgs=[]))
    args = argparse.Namespace(visualizer=["viser"], viser_port=9080, collision_only=True)

    _configure_visualizers(env_cfg, args)

    types = {getattr(cfg, "visualizer_type", None) for cfg in env_cfg.sim.visualizer_cfgs}
    assert "viser" in types
    assert "newton" in types


def test_command_edit_defaults_to_port_9080():
    from command_edit import _parse_args as _parse_edit_args

    args, _ = _parse_edit_args([])
    assert args.viser_port == 9080
    assert args.port == 9080
    assert args.visualizer is None
    assert args.visualizer_disable_all is True


def test_command_edit_custom_viser_port():
    from command_edit import _parse_args as _parse_edit_args

    args, _ = _parse_edit_args(["--viser-port", "9123"])
    assert args.viser_port == 9123
    assert args.port == 9123


def test_command_edit_legacy_port_flag():
    from command_edit import _parse_args as _parse_edit_args

    args, _ = _parse_edit_args(["--port", "9200"])
    assert args.viser_port == 9200
    assert args.port == 9200


def test_command_edit_rejects_incompatible_visualizer():
    from command_edit import _parse_args as _parse_edit_args

    with pytest.raises(SystemExit):
        _parse_edit_args(["--visualizer", "newton"])


def test_command_edit_accepts_viser_visualizer():
    from command_edit import _parse_args as _parse_edit_args

    args, _ = _parse_edit_args(["--visualizer", "viser"])
    assert args.port == 9080


def test_play_num_demos_flag():
    args, _ = _parse_args(["--num-demos", "3"])
    assert args.num_demos == 3

    args_underscore, _ = _parse_args(["--num_demos", "2"])
    assert args_underscore.num_demos == 2


def test_play_num_demos_must_be_positive():
    with pytest.raises(SystemExit):
        _parse_args(["--num-demos", "0"])

    with pytest.raises(SystemExit):
        _parse_args(["--num-demos", "-1"])


