# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Franka Research 3 hidden-button task, retargeted from the ANYmal task of the same scene."""

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    CuroboPlannedGoToFrameCfg,
    CuroboPlannedRotateFrameCfg,
)
from isaaclab_hiveboard.tasks.anymal.button.env import AnymalButtonEnvCfg, AnymalButtonEnvCfg_PLAY
from isaaclab_hiveboard.tasks.franka.common import FRANKA_CUROBO, use_franka
from isaaclab_hiveboard.tasks.scenes.button import LID_PINCH_QUAT

# Object y of the FR3's lid pinch. The shared -0.018 sits on the lid's rounded
# free end, where the FR3's independent fingers squeeze a taper and shove the
# compliant arm sideways; pinch where the lid is full width instead.
LID_PINCH_Y = -0.005

# Released, the FR3's fingers straddle the open lid from above, pointing
# down, at about object (0.05-0.08, 0.03-0.05) depending on where the lid
# arc stopped. The joint-space plan straight to the button drags the palm
# across the lid and slams it shut (the hinge is passive), so clear the lid
# first, keeping the wrist pointing down:
# - slide the fingers off the lid's free end (object x ~0.055, y ~0.05),
#   along its plane and slightly away from it. With the arm nearly stretched
#   there, cuRobo cannot plan a straight lift; a diagonal toward the robot
#   plans reliably;
# - back away from the board. Leveling the hand swings the fingertips about
#   the wrist ~0.15 m toward the board, which near the lid knocks it.
# Then level the hand, keeping the pinch roll (also rolling to the button's
# FACE_QUAT makes cuRobo detour through the lid), and let button_approaching
# roll it in front of the button.
LID_CLEAR_POS = (0.13, 0.09, 0.0)
LID_BACK_POS = (0.28, 0.13, 0.0)
BUTTON_CLEAR_POS = (0.20, 0.06, 0.0)


def _use_franka_button(env_cfg) -> None:
    use_franka(env_cfg, "button", "reset_object_root")
    target_frame = env_cfg.scene.target_frame
    for frame in target_frame.target_frames:
        if frame.name in ("lid_approaching", "lid_grasp"):
            x, _, z = frame.offset.pos
            frame.offset = OffsetCfg(pos=(x, LID_PINCH_Y, z), rot=frame.offset.rot)

    root = target_frame.prim_path
    target_frame.target_frames += [
        FrameTransformerCfg.FrameCfg(prim_path=root, name="lid_clear", offset=OffsetCfg(pos=LID_CLEAR_POS)),
        FrameTransformerCfg.FrameCfg(prim_path=root, name="lid_back", offset=OffsetCfg(pos=LID_BACK_POS)),
        FrameTransformerCfg.FrameCfg(
            prim_path=root, name="button_clear", offset=OffsetCfg(pos=BUTTON_CLEAR_POS, rot=LID_PINCH_QUAT)
        ),
    ]

    commands = list(env_cfg.commands.pose_command.commands)
    # The lid arc is followed by the release; clear the lid right after it.
    release = next(i for i, c in enumerate(commands) if isinstance(c, CuroboPlannedRotateFrameCfg)) + 2
    commands[release:release] = [
        CuroboPlannedGoToFrameCfg(
            frame_name="target_frame",
            gripper_open=True,
            distance_threshold=0.03,
            target_frame_name="lid_clear",
            velocity=0.1,
            hold_current_orientation=True,
            **FRANKA_CUROBO,
        ),
        CuroboPlannedGoToFrameCfg(
            frame_name="target_frame",
            gripper_open=True,
            distance_threshold=0.03,
            target_frame_name="lid_back",
            velocity=0.2,
            hold_current_orientation=True,
            **FRANKA_CUROBO,
        ),
        CuroboPlannedGoToFrameCfg(
            frame_name="target_frame",
            gripper_open=True,
            distance_threshold=0.03,
            target_frame_name="button_clear",
            velocity=0.2,
            **FRANKA_CUROBO,
        ),
    ]
    env_cfg.commands.pose_command.commands = commands
    # Time for the three extra legs.
    env_cfg.episode_length_s += 5.0


@configclass
class FrankaButtonEnvCfg(AnymalButtonEnvCfg):
    """Table-mounted FR3 on the shared HiveBoard hidden-button scene."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_button(self)


@configclass
class FrankaButtonEnvCfg_PLAY(AnymalButtonEnvCfg_PLAY):
    """Deterministic one-environment FR3 demonstration."""

    def __post_init__(self):
        super().__post_init__()
        _use_franka_button(self)
