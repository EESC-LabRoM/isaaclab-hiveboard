# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.tasks.franka.circuit_breaker.configs.actions import (
    FrankaIKAbsActionCfg,
)
from isaaclab_hiveboard.tasks.franka.circuit_breaker.configs.commands import (
    FramePoseCommandsCfg,
)
from isaaclab_hiveboard.tasks.franka.circuit_breaker.configs.events import (
    FrankaCircuitBreakerEventCfg,
)
from isaaclab_hiveboard.tasks.franka.circuit_breaker.configs.observations import (
    ObservationsCfg,
)
from isaaclab_hiveboard.tasks.franka.circuit_breaker.configs.scene import (
    FrankaCircuitBreakerSceneCfg,
)
from isaaclab_hiveboard.tasks.franka.circuit_breaker.configs.terminations import (
    TerminationsCfg,
)


@configclass
class FrankaCircuitBreakerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Franka Emika Panda circuit breaker manipulation environment."""

    scene: FrankaCircuitBreakerSceneCfg = FrankaCircuitBreakerSceneCfg(
        num_envs=1, env_spacing=3.0
    )  # type: ignore
    observations: ObservationsCfg = ObservationsCfg()  # type: ignore
    actions: FrankaIKAbsActionCfg = FrankaIKAbsActionCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    events: FrankaCircuitBreakerEventCfg = FrankaCircuitBreakerEventCfg()  # type: ignore
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()  # type: ignore
    rewards = None

    def __post_init__(self):
        self.decimation = 5
        # 11 s = 440 environment steps. The scripted sequence needs 397 steps.
        self.episode_length_s = 11.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "circuit_breaker"
        self.viewer.body_name = "lever_pivot"
        self.viewer.env_index = 0
        self.viewer.eye = (-1.0, 1.2, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.resolution = (2560, 1440)
        self.sim.dt = 1 / 200
        self.sim.render_interval = 10
        self.sim.render.antialiasing_mode = "DLAA"
        self.sim.render.dlss_mode = 2
        self.sim.render.enable_reflections = True
        self.sim.render.enable_shadows = True
        self.sim.render.enable_direct_lighting = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_global_illumination = True
        self.sim.render.enable_dl_denoiser = True
        self.sim.render.samples_per_pixel = 8
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
