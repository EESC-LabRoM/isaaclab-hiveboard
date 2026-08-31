# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Minimal stationary ANYmal-D + DynaArm + 2F-140 scene for inspecting the TCP."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import (
    ANYMAL_D_DYNAARM_ROBOTIQ_HIGH_PD_CFG,
    ANYMAL_EE,
    ROBOTIQ_INIT_JOINT_POS,
    make_ee_frame,
)


@configclass
class AnymalOnlyRobotSceneCfg(InteractiveSceneCfg):
    """ANYmal-D with DynaArm + 2F-140 and a TCP marker."""

    robot: ArticulationCfg = ANYMAL_D_DYNAARM_ROBOTIQ_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=ANYMAL_D_DYNAARM_ROBOTIQ_HIGH_PD_CFG.spawn.replace(
            articulation_props=ANYMAL_D_DYNAARM_ROBOTIQ_HIGH_PD_CFG.spawn.articulation_props.replace(
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
            joint_pos={
                ".*HAA": 0.0,
                ".*F_HFE": 0.4,
                ".*H_HFE": -0.4,
                ".*F_KFE": -0.8,
                ".*H_KFE": 0.8,
                # Mount yaw is 180 deg, so +pi shoulder rotation puts the
                # gripper in front of the trunk for TCP inspection.
                "dynaarm_shoulder_rotation": 3.14159,
                "dynaarm_shoulder_flexion": 0.4,
                "dynaarm_elbow_flexion": 1.2,
                "dynaarm_forearm_rotation": 0.0,
                "dynaarm_wrist_flexion": 0.0,
                "dynaarm_wrist_rotation": 1.5708,
                **ROBOTIQ_INIT_JOINT_POS,
            },
            joint_vel={".*": 0.0},
        ),
    )
    ee_frame: FrameTransformerCfg = make_ee_frame(ANYMAL_EE, debug_vis=True)

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Intentionally empty: this diagnostic robot cannot receive commands."""


@configclass
class ObservationsCfg:
    """Small observation set required by the manager-based environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    """Keep the diagnostic viewport running until the user closes it."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class AnymalOnlyRobotEnvCfg(ManagerBasedRLEnvCfg):
    """Fixed-base ANYmal-D + DynaArm TCP-frame inspection environment."""

    scene: AnymalOnlyRobotSceneCfg = AnymalOnlyRobotSceneCfg(num_envs=1, env_spacing=2.0)  # type: ignore
    observations: ObservationsCfg = ObservationsCfg()  # type: ignore
    actions: ActionsCfg = ActionsCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    commands = None
    rewards = None

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 1.0e6
        self.sim.dt = 1 / 120
        self.sim.render.antialiasing_mode = "DLSS"
        self.sim.render.dlss_mode = 0
        self.sim.render.enable_reflections = False
        self.sim.render.enable_global_illumination = False
        self.sim.render.enable_ambient_occlusion = False
        self.sim.render.enable_dl_denoiser = False
        self.sim.render.samples_per_pixel = 1
        # self.viewer.origin_type = "asset_body"
        # self.viewer.asset_name = "robot"
        # self.viewer.body_name = ANYMAL_EE.body_name
        # self.viewer.env_index = 0
        # 3/4 view of the 2F-140 palm; look slightly along +Z (finger approach).
        # self.viewer.eye = (-0.45, 0.40, 0.22)
        # self.viewer.lookat = (0.08, 0.0, 0.12)
        # self.viewer.resolution = (1280, 720)
