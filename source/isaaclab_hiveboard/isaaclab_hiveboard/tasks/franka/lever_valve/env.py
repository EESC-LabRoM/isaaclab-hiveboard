# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.tasks.franka.circuit_breaker.configs.actions import (
    FrankaIKAbsActionCfg,
)
from isaaclab_hiveboard.tasks.franka.circuit_breaker.configs.observations import (
    ObservationsCfg,
)
from isaaclab_hiveboard.tasks.franka.lever_valve.configs.terminations import (
    TerminationsCfg,
)
from isaaclab_hiveboard.tasks.franka.lever_valve.configs.commands import (
    FramePoseCommandsCfg,
)
from isaaclab_hiveboard.tasks.franka.lever_valve.configs.events import (
    FrankaLeverValveEventCfg,
)
from isaaclab_hiveboard.tasks.franka.lever_valve.configs.scene import (
    FrankaLeverValveSceneCfg,
)


@configclass
class FrankaLeverValveEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Franka Research 3 lever / ball valve manipulation environment."""

    scene: FrankaLeverValveSceneCfg = FrankaLeverValveSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: ObservationsCfg = ObservationsCfg()  # type: ignore
    actions: FrankaIKAbsActionCfg = FrankaIKAbsActionCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    events: FrankaLeverValveEventCfg = FrankaLeverValveEventCfg()  # type: ignore
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()  # type: ignore
    rewards = None

    def __post_init__(self):
        # Give this environment its own TCP orientation while leaving the
        # valve target and rotation frames unchanged. Rotating the TCP frame
        # -90 deg makes the physical hand roll +90 deg about TCP +X, preserving
        # the successful grasp orientation from the previous target-frame fix.
        half_sqrt = math.sqrt(0.5)
        tcp_roll_x_minus_90 = (
            half_sqrt,
            -half_sqrt,
            0.0,
            0.0,
        )  # quaternion (w, x, y, z)
        tcp_offsets = (
            self.scene.ee_frame.target_frames[0].offset,
            self.commands.pose_command.body_offset,
            self.actions.arm_action.body_offset,
        )
        for offset in tcp_offsets:
            if offset is None:
                raise ValueError("Franka lever-valve TCP offsets must be configured")
            w1, x1, y1, z1 = offset.rot
            w2, x2, y2, z2 = tcp_roll_x_minus_90
            offset.rot = (
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            )

        self.decimation = 5
        # 11 s = 440 environment steps. The scripted sequence needs 365 steps.
        self.episode_length_s = 11.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "ball_valve"
        self.viewer.body_name = "alavanca_pivot"
        self.viewer.env_index = 0
        self.viewer.eye = (-1.0, 1.2, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.resolution = (2560, 1440)
        self.sim.dt = 1 / 200
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
