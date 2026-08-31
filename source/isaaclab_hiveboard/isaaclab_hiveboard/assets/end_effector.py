# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-robot flange → TCP profiles.

Task ``target_frame`` poses are desired TCP poses in a single convention, so
HiveBoard scenes can be shared across robots. Tune ``tcp_offset`` once per
robot; commands, IK, ``ee_frame``, and reset IK all read it from here.

Canonical TCP (right-handed):
    +X  out of the palm (approach)
    +Z  jaw "up"
    +Y  across the jaws

Quaternions are Isaac Lab ``(w, x, y, z)``.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass


@configclass
class WorkspaceCfg:
    """Default HiveBoard placement relative to this robot's base."""

    warehouse_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """World-frame warehouse spawn. Spot sits at z=0 with a 0.60 m floor drop."""

    object_pos: tuple[float, float, float] = (1.0, 0.0, 0.0)
    """Default object root position in the env frame."""

    object_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Default object root quaternion ``(w, x, y, z)``."""


@configclass
class EndEffectorCfg:
    """Flange body and the offset from that body to the canonical TCP."""

    body_name: str = MISSING  # type: ignore
    """Articulation body used by pose commands and differential IK."""

    source_prim: str = MISSING  # type: ignore
    """Robot-relative prim for the ``ee_frame`` source (e.g. ``body``, ``fr3_link0``)."""

    body_prim: str = MISSING  # type: ignore
    """Robot-relative prim of the flange body (e.g. ``arm_link_wr1``, ``fr3_hand``)."""

    tcp_offset: OffsetCfg = MISSING  # type: ignore
    """Flange → canonical TCP. Tune this once per robot."""

    finger_frames: tuple[FrameTransformerCfg.FrameCfg, ...] = ()
    """Optional finger frames appended after the TCP on ``ee_frame``."""


def print_ee_offset_report(env, ee: EndEffectorCfg, *, env_id: int = 0) -> None:
    """Print gripper-body vs offset-TCP axes so the offset can be checked by eye.

    Canonical TCP: +X out of the fingers, +Y across the jaws, +Z jaw-up.
    The RGB gizmo on ``ee_frame`` is the same frame (red/green/blue = X/Y/Z).
    """
    import isaaclab.utils.math as math_utils

    robot = env.scene["robot"]
    body_ids, body_names = robot.find_bodies(ee.body_name)
    if not body_ids:
        print(f"[EE] body {ee.body_name!r} not found")
        return
    bid = body_ids[0]
    body_pos = robot.data.body_pos_w[env_id, bid]
    body_quat = robot.data.body_quat_w[env_id, bid]
    off_pos = body_pos.new_tensor(ee.tcp_offset.pos)
    off_rot = body_quat.new_tensor(ee.tcp_offset.rot)
    tcp_pos, tcp_quat = math_utils.combine_frame_transforms(
        body_pos.unsqueeze(0), body_quat.unsqueeze(0), off_pos.unsqueeze(0), off_rot.unsqueeze(0)
    )
    tcp_pos, tcp_quat = tcp_pos[0], tcp_quat[0]

    def _ax(quat, vec):
        v = math_utils.quat_apply(quat.unsqueeze(0), quat.new_tensor([vec]))[0]
        return tuple(round(float(v[i]), 3) for i in range(3))

    def _p(t):
        return tuple(round(float(t[i]), 3) for i in range(3))

    print("[EE] offset check  (canonical: +X approach, +Y across jaws, +Z up)")
    print(f"[EE] body={body_names[0]!r}  pos_w={_p(body_pos)}")
    print(f"[EE]   body +X={_ax(body_quat, [1, 0, 0])}  +Y={_ax(body_quat, [0, 1, 0])}  +Z={_ax(body_quat, [0, 0, 1])}")
    print(f"[EE] tcp_offset pos={tuple(ee.tcp_offset.pos)}  rot(wxyz)={tuple(round(float(x), 4) for x in ee.tcp_offset.rot)}")
    print(f"[EE] tcp  pos_w={_p(tcp_pos)}")
    print(f"[EE]   tcp  +X={_ax(tcp_quat, [1, 0, 0])}  <- red, out of fingers")
    print(f"[EE]   tcp  +Y={_ax(tcp_quat, [0, 1, 0])}  <- green, across jaws")
    print(f"[EE]   tcp  +Z={_ax(tcp_quat, [0, 0, 1])}  <- blue, jaw up")


def make_ee_frame(ee: EndEffectorCfg, *, debug_vis: bool = False) -> FrameTransformerCfg:
    """Build the scene ``ee_frame``. Target 0 is always the TCP."""
    from isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (  # isort: skip
        FRAME_MARKER_SMALL_CFG,
    )

    tcp = FrameTransformerCfg.FrameCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + ee.body_prim,
        name="ee_tcp",
        offset=OffsetCfg(pos=tuple(ee.tcp_offset.pos), rot=tuple(ee.tcp_offset.rot)),
    )
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + ee.source_prim,
        debug_vis=debug_vis,
        visualizer_cfg=FRAME_MARKER_SMALL_CFG.replace(prim_path="/Visuals/EndEffectorFrameTransformer"),
        target_frames=[tcp],  # , *ee.finger_frames],
    )


def as_command_offset(ee: EndEffectorCfg):
    """TCP offset for :class:`SequentialPoseCommandCfg`."""
    from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
        SequentialPoseCommandCfg,
    )

    return SequentialPoseCommandCfg.OffsetCfg(
        pos=tuple(ee.tcp_offset.pos),
        rot=tuple(ee.tcp_offset.rot),
    )


def as_ik_offset(ee: EndEffectorCfg):
    """TCP offset for differential IK."""
    from isaaclab.envs.mdp.actions.actions_cfg import (
        DifferentialInverseKinematicsActionCfg,
    )

    return DifferentialInverseKinematicsActionCfg.OffsetCfg(
        pos=tuple(ee.tcp_offset.pos),
        rot=tuple(ee.tcp_offset.rot),
    )


SPOT_EE = EndEffectorCfg(
    body_name="arm_link_wr1",
    source_prim="body",
    body_prim="arm_link_wr1",
    tcp_offset=OffsetCfg(pos=(0.21, 0.0, -0.03), rot=(1.0, 0.0, 0.0, 0.0)),
)
"""Spot arm: ``arm_link_wr1`` +X is already the canonical approach axis."""

SPOT_WORKSPACE = WorkspaceCfg(
    warehouse_pos=(0.0, 0.0, -0.60),
    object_pos=(1.0, 0.0, 0.0),
)

# Franka hand axes differ from the shared canonical TCP convention.  This
# fixed transform aligns the fingertip approach direction with canonical +X
# and the thin/top hand axis with canonical +Z, so scene target frames can
# stay identical for every robot.
FRANKA_EE = EndEffectorCfg(
    body_name="fr3_hand",
    source_prim="fr3_link0",
    body_prim="fr3_hand",
    tcp_offset=OffsetCfg(
        pos=(0.0, 0.0, 0.1034),
        # A 180 deg roll about TCP X keeps the red X axis pointing forward
        # through the fingertips and flips the blue Z axis from down to up.
        rot=(0.0, 0.7071068, 0.0, 0.7071068),
    ),
    finger_frames=(
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/fr3_leftfinger",
            name="tool_leftfinger",
            offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
        ),
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/fr3_rightfinger",
            name="tool_rightfinger",
            offset=OffsetCfg(pos=(0.0, 0.0, 0.046)),
        ),
    ),
)
"""Franka FR3 hand transformed into the shared canonical TCP frame."""

FRANKA_WORKSPACE = WorkspaceCfg(
    warehouse_pos=(0.0, 0.0, 0.0),
    object_pos=(0.55, 0.0, 0.40),
    object_rot=(0.0, 0.0, 0.0, 1.0),
)

# Isaac Lab 2F-140 (UR10e Robotiq_2f_140 payload): palm +Z is finger approach,
# +Y across the jaws. Rotation -90 deg about Y: TCP +X = body +Z, TCP +Y =
# body +Y, TCP +Z = body -X. Translation 0.20 m along body +Z puts the triad
# on the pads (measured on the 2F-140 palm).
ANYMAL_EE = EndEffectorCfg(
    body_name="robotiq_base_link",
    source_prim="base",
    body_prim="robotiq_2f_140/robotiq_base_link",
    tcp_offset=OffsetCfg(
        pos=(0.0, 0.0, 0.20),
        rot=(0.7071068, 0.0, -0.7071068, 0.0),
    ),
)
"""ANYmal-D DynaArm + Isaac Lab 2F-140 transformed into the shared canonical TCP frame."""

ANYMAL_WORKSPACE = WorkspaceCfg(
    warehouse_pos=(0.0, 0.0, 0.0),
    object_pos=(0.80, 0.0, 0.90),
    # 180 deg yaw so the HiveBoard lever faces the robot, same as Franka.
    object_rot=(0.0, 0.0, 0.0, 1.0),
)
