# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hold one known TCP position with the bench-valve command/action pipeline."""

import torch
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_tasks.utils import PresetCfg

from isaaclab_hiveboard.assets.spot.bench import DECIMATION, PHYSICS_DT, TCP_SITE_POS
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    GoToFrameCfg,
    SequentialPoseCommandCfg,
    _GoToFrameHandler,
)
from isaaclab_hiveboard.tasks.spot.bench_valve.configs.actions import SpotBenchPositionActionCfg
from isaaclab_hiveboard.tasks.spot.bench_valve.configs.events import BenchValveEventCfg
from isaaclab_hiveboard.tasks.spot.gains.configs.scene import SpotGainsSceneCfg

# Metres relative to each environment's origin; this is the bench approach bead.
TARGET_POSITION_ENV = (0.9182, -0.0727, 0.8261)


class _HoldPositionHandler(_GoToFrameHandler):
    def is_done(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Keep tracking the target, including after reaching it."""
        return torch.zeros_like(env_ids, dtype=torch.bool)


@configclass
class CommandsCfg:
    pose_command = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name="arm_link_wr1",
        body_offset=SequentialPoseCommandCfg.OffsetCfg(pos=TCP_SITE_POS),
        resampling_time_range=(1e9, 1e9),
        debug_vis=True,
        commands=[
            GoToFrameCfg(
                class_type=_HoldPositionHandler,
                frame_name="",
                target_frame_name="",
                target_position_env=TARGET_POSITION_ENV,
                gripper_open=True,
                velocity=0.15,
                canonicalize_upward=False,
            )
        ],
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        command = ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "pose_command"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy = PolicyCfg()


@configclass
class TerminationsCfg:
    """No automatic success or timeout: hold the target until stopped."""


@configclass
class ValidationPhysicsCfg(PresetCfg):
    # Startup gravity-compensation changes must be evaluated by the solver.
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=600,
            nconmax=400,
            iterations=100,
            ls_iterations=20,
            impratio=10.0,
            ccd_iterations=50,
            use_mujoco_contacts=False,
        ),
        num_substeps=1,
        use_cuda_graph=False,
    )
    default = newton_mjwarp


@configclass
class ValidateCommandSpotEnvCfg(ManagerBasedRLEnvCfg):
    scene = SpotGainsSceneCfg(num_envs=1, env_spacing=3.0)
    commands = CommandsCfg()
    actions = SpotBenchPositionActionCfg()
    observations = ObservationsCfg()
    events = BenchValveEventCfg()
    terminations = TerminationsCfg()
    rewards = None
    recorders = None
    sim = SimulationCfg(
        dt=PHYSICS_DT,
        render_interval=DECIMATION,
        physics=ValidationPhysicsCfg(),
    )

    def __post_init__(self):
        self.decimation = DECIMATION
        self.episode_length_s = 1e9
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (1.8, -2.0, 1.2)
        self.viewer.lookat = (0.7, 0.0, 0.8)
