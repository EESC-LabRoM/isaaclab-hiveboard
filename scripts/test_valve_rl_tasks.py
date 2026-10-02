"""RL valve tasks: the ball- and small-valve configurations and the valve terms' conventions.

Config-only checks (no simulation)::

    uv run --with pytest python -m pytest scripts/test_valve_rl_tasks.py
"""

from __future__ import annotations

import importlib
import math

import gymnasium as gym
import isaaclab_hiveboard  # noqa: F401
import pytest
import torch

from isaaclab_hiveboard.tasks.anymal.ball_valve_rl import valve_dynamics

_TASKS = {
    "Isaac-HiveBoard-Anymal-BallValve-RL-v0": ("ball_valve", "Isaac-HiveBoard-Anymal-BallValve-v0"),
    "Isaac-HiveBoard-Anymal-SmallValve-RL-v0": ("small_valve", "Isaac-HiveBoard-Anymal-SmallValve-v0"),
}


def _cfg(task: str):
    module, name = gym.spec(task).kwargs["env_cfg_entry_point"].split(":")
    return importlib.import_module(module), getattr(importlib.import_module(module), name)()


@pytest.mark.parametrize("task", list(_TASKS))
@pytest.mark.parametrize("play", [False, True])
def test_every_valve_reference_names_the_task_valve(task, play):
    module, cfg = _cfg(task.replace("-RL-v0", "-RL-Play-v0") if play else task)
    asset, expert_task = _TASKS[task]
    assert cfg.valve_task.asset_name == asset
    assert getattr(cfg.scene, asset) is not None
    assert cfg.events.valve_gravcomp.params["asset_cfg"].name == asset
    assert cfg.events.valve_physics_material is None or cfg.events.valve_physics_material.params["asset_cfg"].name == asset
    assert cfg.actions.valve_load.asset_name == asset
    assert cfg.viewer.asset_name == asset
    assert cfg.events.reset_from_bank.params["expert_task"] == expert_task == module.EXPERT_TASK
    # The grasp frame exists in the scene and sits on a body of the valve.
    frames = {f.name: f for f in cfg.scene.target_frame.target_frames}
    assert cfg.valve_task.grasp_frame in frames
    assert frames[cfg.valve_task.grasp_frame].prim_path.startswith("{ENV_REGEX_NS}/Valve/")
    # The bank builder's settings.
    for name in ("VALVE_POSE_RANGE", "VALVE_ANGLE_RANGE", "ARM_POSTURES", "VALVE_DYNAMICS_RANGES",
                 "EXPERT_DIVERSITY_RANGES", "EXPERT_GRASP_AXES", "EXPERT_OVERSHOOT", "EXPERT_GRIP_S", "BANK_PREFIX"):
        assert hasattr(module, name), name
    assert cfg.events.reset_from_bank.params["mid_start_prob"] == (0.0 if play else 0.5)


def test_ball_valve_dynamics_unchanged():
    _, cfg = _cfg("Isaac-HiveBoard-Anymal-BallValve-RL-v0")
    params = cfg.events.valve_dynamics.params
    assert params["ranges"] == valve_dynamics.VALVE_DYNAMICS_RANGES
    scale = valve_dynamics.dynamics_scale(params["ranges"], params["stuck_breakaway"])
    assert scale == (2.0, 0.5, 1.0, 5.0, 0.02)


@pytest.mark.parametrize("closed, opening_sign", [(0.0, -1.0), (0.0, 1.0), (0.3, -1.0)])
def test_closing_torque_turns_toward_closed(closed, opening_sign):
    spring, breakaway = torch.tensor([1.0]), torch.tensor([2.0])
    seated = valve_dynamics.closing_torque(torch.tensor([closed]), spring, breakaway, closed, opening_sign)
    part_open = torch.tensor([closed + opening_sign * 0.5])
    opened = valve_dynamics.closing_torque(part_open, spring, breakaway, closed, opening_sign)
    # Full seat torque at closed, spring only past the seat, both pushing against opening.
    assert seated.item() == pytest.approx(-opening_sign * 2.0)
    assert opened.item() == pytest.approx(-opening_sign * 0.5)
    # Past closed (the wrong way) only the seat torque, still toward closed.
    beyond = valve_dynamics.closing_torque(torch.tensor([closed - opening_sign * 0.2]), spring, breakaway, closed, opening_sign)
    assert beyond.item() == pytest.approx(-opening_sign * 2.0)


def test_small_valve_turns_a_quarter_turn_open_negative():
    _, cfg = _cfg("Isaac-HiveBoard-Anymal-SmallValve-RL-v0")
    assert cfg.valve_task.open_rad == pytest.approx(-math.pi / 2) and cfg.valve_task.closed_rad == 0.0
