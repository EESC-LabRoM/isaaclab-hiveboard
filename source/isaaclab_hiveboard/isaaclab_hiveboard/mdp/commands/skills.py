# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Named skills for the cuRobo expert sequences.

Every HiveBoard manipulation sequence is a short grammar of six phases::

    approach -> engage -> grip -> actuate -> release -> retreat

* **approach**: free-space cuRobo move to a stand-off pose in front of the
  object, gripper preshaped.
* **engage**: short, slow move from the stand-off onto the contact or grasp
  pose.
* **grip**: close the gripper on the object.
* **actuate**: move the mechanism along its degree of freedom (turn a lever,
  pull a drawer, press a button), usually until a :class:`MechanismGoalCfg`
  holds.
* **release**: open the gripper.
* **retreat**: back off to a stand-off pose, clear of the object.

A phase can be skipped (the ball valve's turn closes the gripper itself, a push
needs no grip) or repeated (a thread is turned in several regrasp cycles). Each
segment carries its phase as :attr:`BaseCmd.phase`, so the expert bank, the RL
references and the datasets can find the phases by name rather than by
segment index. :class:`SkillSet` builds the segments for one robot::

    anymal = SkillSet(**ANYMAL_CUROBO)
    commands = [
        anymal.approach("approaching", chain_with_next=True),
        anymal.engage("lever_pivot", chain_with_next=True),
        anymal.turn(angle_deg=-90.0, until=MechanismGoalCfg(goal_tolerance=0.05)),
        anymal.release(),
        anymal.retreat("approaching", hold_current_orientation=True),
    ]
"""

from __future__ import annotations

from collections.abc import Sequence

from .sequential_pose_command import (
    PHASES,
    CuroboPlannedGoToFrameCfg,
    CuroboPlannedRotateFrameCfg,
    GripperCommand,
    MechanismGoalCfg,
)

__all__ = ["PHASES", "MechanismGoalCfg", "SkillSet"]


class SkillSet:
    """Builds phase-tagged cuRobo segments for one robot.

    Every keyword argument of a skill is passed on to its segment config, so a
    skill only sets the defaults that make it that skill (gripper state,
    speed, thresholds) and anything else can still be tuned per task.
    """

    def __init__(
        self,
        robot_joint_names: Sequence[str],
        robot_curobo_yaml: str | None = None,
        robot_urdf: str | None = None,
        frame_name: str = "target_frame",
    ):
        """
        Args:
            robot_joint_names: Arm joints cuRobo plans for, in action order.
            robot_curobo_yaml: The robot's cuRobo configuration.
            robot_urdf: The robot's URDF, for cuRobo's kinematics.
            frame_name: Frame-transformer sensor holding the object's target frames.
        """
        self.planner = {
            "robot_joint_names": list(robot_joint_names),
            "robot_curobo_yaml": robot_curobo_yaml,
            "robot_urdf": robot_urdf,
        }
        self.frame_name = frame_name

    def _goto(self, phase: str, target_frame_name: str, **kwargs) -> CuroboPlannedGoToFrameCfg:
        kwargs.setdefault("frame_name", self.frame_name)
        return CuroboPlannedGoToFrameCfg(phase=phase, target_frame_name=target_frame_name, **self.planner, **kwargs)

    def approach(self, target_frame_name: str, **kwargs) -> CuroboPlannedGoToFrameCfg:
        """Free-space move to the stand-off pose ``target_frame_name``, gripper open."""
        kwargs.setdefault("gripper_open", True)
        kwargs.setdefault("velocity", 0.25)
        kwargs.setdefault("distance_threshold", 0.03)
        return self._goto("approach", target_frame_name, **kwargs)

    def engage(self, target_frame_name: str, **kwargs) -> CuroboPlannedGoToFrameCfg:
        """Slow move from the stand-off onto the grasp or contact pose ``target_frame_name``."""
        kwargs.setdefault("gripper_open", True)
        kwargs.setdefault("velocity", 0.15)
        kwargs.setdefault("distance_threshold", 0.02)
        return self._goto("engage", target_frame_name, **kwargs)

    def grip(self, duration_s: float = 0.3, **kwargs) -> GripperCommand:
        """Close the gripper on the object."""
        return GripperCommand(phase="grip", open_gripper=False, duration_s=duration_s, **kwargs)

    def turn(self, target_frame_name: str = "rotate_frame", **kwargs) -> CuroboPlannedRotateFrameCfg:
        """Turn the gripped object about the axis of ``target_frame_name`` (a lever, key or handwheel)."""
        kwargs.setdefault("frame_name", self.frame_name)
        kwargs.setdefault("gripper_open", False)
        return CuroboPlannedRotateFrameCfg(
            phase="actuate", target_frame_name=target_frame_name, **self.planner, **kwargs
        )

    def move(self, target_frame_name: str, **kwargs) -> CuroboPlannedGoToFrameCfg:
        """Drive the mechanism in a straight move to ``target_frame_name`` (pull, push or press).

        Pass ``until`` to finish on the mechanism rather than on the TCP
        reaching the target: a press or a push stops short of it.
        """
        kwargs.setdefault("gripper_open", False)
        kwargs.setdefault("velocity", 0.05)
        kwargs.setdefault("distance_threshold", 0.01)
        return self._goto("actuate", target_frame_name, **kwargs)

    def hold(self, phase: str, duration_s: float, open_gripper: bool = False, **kwargs) -> GripperCommand:
        """Keep the pose for ``duration_s`` within ``phase``, e.g. to let the mechanism settle."""
        return GripperCommand(phase=phase, open_gripper=open_gripper, duration_s=duration_s, **kwargs)

    def release(self, duration_s: float = 0.3, **kwargs) -> GripperCommand:
        """Open the gripper."""
        return GripperCommand(phase="release", open_gripper=True, duration_s=duration_s, **kwargs)

    def retreat(self, target_frame_name: str, **kwargs) -> CuroboPlannedGoToFrameCfg:
        """Back off to the stand-off pose ``target_frame_name``, clear of the object."""
        kwargs.setdefault("gripper_open", True)
        kwargs.setdefault("velocity", 0.2)
        kwargs.setdefault("distance_threshold", 0.03)
        return self._goto("retreat", target_frame_name, **kwargs)
