"""Phase-tagged skills and mechanism goals of the cuRobo expert sequences.

Config-only checks (no simulation): the skill factories, the phase vocabulary,
and the JSON round trip of the saved command setups.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    CuroboPlannedRotateFrameCfg,
    GripperCommand,
)
from isaaclab_hiveboard.mdp.commands.skills import PHASES, MechanismGoalCfg, SkillSet
from isaaclab_hiveboard.tasks.anymal.mechanism import ANYMAL_CUROBO
from isaaclab_hiveboard.utils.command_setup import decode_command, encode_command, load_setup, validate_setup

_REPO = Path(__file__).resolve().parents[1]
_BALL_VALVE_SETUP = _REPO / "configs/Isaac-HiveBoard-Anymal-BallValve-v0.json"


def test_skills_tag_their_phase_and_robot():
    anymal = SkillSet(**ANYMAL_CUROBO)
    commands = [
        anymal.approach("approaching"),
        anymal.engage("lever_pivot"),
        anymal.grip(),
        anymal.turn(angle_deg=-90.0),
        anymal.move("pressed"),
        anymal.release(),
        anymal.retreat("approaching"),
    ]
    assert [c.phase for c in commands] == ["approach", "engage", "grip", "actuate", "actuate", "release", "retreat"]
    assert set(c.phase for c in commands) == set(PHASES)
    for cmd in commands:
        if isinstance(cmd, (CuroboPlannedGoToFrameCfg, CuroboPlannedRotateFrameCfg)):
            assert cmd.robot_joint_names == ANYMAL_CUROBO["robot_joint_names"]
            assert cmd.frame_name == "target_frame"
    assert commands[0].gripper_open and not commands[3].gripper_open
    assert commands[2].open_gripper is False and commands[5].open_gripper is True


def test_skill_keywords_override_defaults():
    anymal = SkillSet(**ANYMAL_CUROBO)
    cmd = anymal.approach("approaching", velocity=0.4, chain_with_next=True)
    assert cmd.velocity == 0.4 and cmd.chain_with_next


def test_phase_and_until_survive_the_json_round_trip():
    anymal = SkillSet(**ANYMAL_CUROBO)
    goal = MechanismGoalCfg(goal_tolerance=0.05, settle_speed=0.1)
    cmd = anymal.turn(angle_deg=-90.0, until=goal)
    data = json.loads(json.dumps(encode_command(cmd)))
    assert data["parameters"]["phase"] == "actuate"
    assert data["parameters"]["until"]["goal_tolerance"] == 0.05
    back = decode_command(data)
    assert back.phase == "actuate"
    assert isinstance(back.until, MechanismGoalCfg)
    assert back.until.goal_tolerance == 0.05 and back.until.settle_speed == 0.1 and back.until.low is None


def test_untagged_setups_still_load():
    data = encode_command(GripperCommand(open_gripper=True, duration_s=0.3))
    del data["parameters"]["phase"], data["parameters"]["until"]
    cmd = decode_command(data)
    assert cmd.phase is None and cmd.until is None


@pytest.mark.parametrize(
    "bad",
    [
        {"phase": "wiggle"},
        {"until": {"low": 0.0, "colour": 1.0}},
        {"until": {}},
        {"until": {"settle_speed": -1.0}},
        {"until": {"low": float("nan")}},
    ],
)
def test_bad_phase_or_goal_is_rejected(bad):
    data = encode_command(GripperCommand(open_gripper=True, duration_s=0.3))
    data["parameters"].update(bad)
    with pytest.raises(ValueError):
        decode_command(data)


def test_ball_valve_setup_is_phase_tagged():
    commands, _ = validate_setup(load_setup(_BALL_VALVE_SETUP))
    phases = [c.phase for c in commands]
    assert phases == ["approach", "engage", "actuate", "release", "retreat"]
    # Phases never go backwards within one cycle.
    order = [PHASES.index(p) for p in phases]
    assert order == sorted(order)


@pytest.mark.parametrize("path", sorted((_REPO / "configs").glob("Isaac-HiveBoard-*.json")), ids=lambda p: p.stem)
def test_every_saved_setup_loads(path):
    validate_setup(load_setup(path))
