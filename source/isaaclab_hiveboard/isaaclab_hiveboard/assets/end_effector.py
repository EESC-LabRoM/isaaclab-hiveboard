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
