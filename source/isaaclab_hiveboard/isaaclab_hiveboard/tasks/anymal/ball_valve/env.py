# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .configs.actions import AnymalIKAbsActionCfg
from .configs.commands import FramePoseCommandsCfg
from .configs.events import AnymalBallValveEventCfg
from .configs.observations import ObservationsCfg
from .configs.scene import AnymalBallValveSceneCfg
from .configs.terminations import TerminationsCfg


@configclass
class AnymalBallValveEnvCfg(ManagerBasedRLEnvCfg):
    """ANYmal-D + DynaArm + 2F-140 opening the HiveBoard ball valve."""

    scene: AnymalBallValveSceneCfg = AnymalBallValveSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: ObservationsCfg = ObservationsCfg()  # type: ignore
    actions: AnymalIKAbsActionCfg = AnymalIKAbsActionCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    events: AnymalBallValveEventCfg = AnymalBallValveEventCfg()  # type: ignore
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()  # type: ignore
    rewards = None

    def __post_init__(self):
        # Roll the TCP +90 deg about approach (+X) so jaw-across (green) sits
        # where jaw-up (blue) was — 2F-140 fingers stack vertically on the lever.
        half = math.sqrt(0.5)
        tcp_roll_x_plus_90 = (half, half, 0.0, 0.0)
        # At the valve grasp, palm +Y is world-down. ANYMAL_EE.tcp_offset is only
        # 0.20 m along palm +Z (approach, slightly down), so the TCP sits ~3 cm
        # under the lever. A +Y shift puts the TCP below the palm; IK then lifts
        # the arm onto the handle. Keep this here so only-gripper vis stays centered.
        tcp_lift_along_jaws = 0.04
        tcp_offsets = (
            self.scene.ee_frame.target_frames[0].offset,
            self.commands.pose_command.body_offset,
            self.actions.arm_action.body_offset,
        )
        for offset in tcp_offsets:
            if offset is None:
                raise ValueError("ANYmal ball-valve TCP offsets must be configured")
            px, py, pz = offset.pos
            offset.pos = (px, py + tcp_lift_along_jaws, pz)
            w1, x1, y1, z1 = offset.rot
            w2, x2, y2, z2 = tcp_roll_x_plus_90
            offset.rot = (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )

        self.decimation = 10
        # Position-only DLS spends most of the episode walking ~1.5 m to the lever.
        self.episode_length_s = 20.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "ball_valve"
        self.viewer.body_name = "alavanca_pivot"
        self.viewer.env_index = 0
        self.viewer.eye = (-1.5, 1.5, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.resolution = (2560, 1440)
        self.sim.dt = 1 / 200
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
