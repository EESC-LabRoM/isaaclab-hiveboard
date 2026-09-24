# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared builder for HiveBoard screw scenes (M8 / M30 thread, threaded peg).

Each asset chains ``RevoluteJoint`` (continuous, about object +X) and
``PrismaticJoint`` (along object +X, 0 = seated) from its fixed root link, so
the part screws in toward the plate as the revolute turns negative about +X
(right-hand thread). The command term couples the two joints at the thread
pitch; see ``tasks/anymal/screw.py``.
"""

from dataclasses import dataclass

from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from isaaclab_hiveboard.tasks.scenes.mechanism import (
    FACE_QUAT,
    MECHANISM_SPAWN_POS,
    face_roll_quat,
    mechanism_spawn,
    target_frames,
)


@dataclass
class ScrewSpec:
    """Geometry and thread of one screw attachment, in its root-link frame."""

    pitch: float
    """Axial travel per revolution [m]."""
    start: float
    """PrismaticJoint at reset [m]; 0 is seated."""
    travel: float
    """PrismaticJoint upper limit [m]."""
    grasp_x: float
    """Mid-pad TCP height above the plate for ANYmal/Spot [m]."""
    franka_tip_x: float
    """FR3 fingertip TCP height [m]; its pads sit behind the tip."""
    jaw_roll_deg: float
    """Jaw direction, see :func:`face_roll_quat` (flats of a hex nut)."""
    seated_tolerance: float
    """PrismaticJoint at or below this counts as seated [m]."""
    friction: float
    """Thread friction on the RevoluteJoint [N m]."""

    @property
    def turns(self) -> float:
        return self.start / self.pitch


def screw_articulation(prim_name: str, usd_path: str, semantic_class: str, spec: ScrewSpec) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/" + prim_name,
        spawn=mechanism_spawn(usd_path, semantic_class),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=MECHANISM_SPAWN_POS,
            rot=FACE_QUAT,
            joint_pos={"RevoluteJoint": 0.0, "PrismaticJoint": spec.start},
            joint_vel={".*": 0.0},
        ),
        actuators={
            # Thread friction as solver joint friction. An explicit resistance
            # torque (ScrewJointCouplingCfg friction terms) is unstable on
            # links this light.
            # Armature and damping keep a finger release from flicking the
            # light part round several turns.
            "spin": ImplicitActuatorCfg(
                joint_names_expr=["RevoluteJoint"],
                stiffness=0.0,
                damping=0.05,
                armature=0.001,
                friction=spec.friction,
                effort_limit_sim=5.0,
            ),
            # Follows the pitch-coupled position target written every step,
            # stiff enough that pushing on the part does not slide it along
            # the thread.
            "advance": ImplicitActuatorCfg(
                joint_names_expr=["PrismaticJoint"],
                stiffness=1.0e5,
                damping=200.0,
                effort_limit_sim=500.0,
            ),
        },
    )


SCREW_PINCH_FRAMES = ("approaching", "grasp", "clear")
"""Frames that close the jaws on the part (rolled for Spot by ``use_spot``)."""
SCREW_GRASP_FRAMES = ("grasp", "clear", "rotate_frame")
"""Frames that move with the TCP-to-pad offset (shifted for the FR3)."""


def screw_frames(root_link_prim: str, marker: str, spec: ScrewSpec) -> FrameTransformerCfg:
    pinch = face_roll_quat(spec.jaw_roll_deg)
    return target_frames(
        root_link_prim,
        marker,
        {
            "approaching": OffsetCfg(pos=(0.15, 0.0, 0.0), rot=pinch),
            "grasp": OffsetCfg(pos=(spec.grasp_x, 0.0, 0.0), rot=pinch),
            # Straight back off the part after each turn, before regrasping.
            "clear": OffsetCfg(pos=(spec.grasp_x + 0.03, 0.0, 0.0), rot=pinch),
            # On the thread axis; the turn is about object +X (frame -X).
            "rotate_frame": OffsetCfg(pos=(spec.grasp_x, 0.0, 0.0), rot=FACE_QUAT),
        },
    )
