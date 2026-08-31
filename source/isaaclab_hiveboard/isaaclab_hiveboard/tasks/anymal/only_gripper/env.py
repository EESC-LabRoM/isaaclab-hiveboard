# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fixed-base Isaac Lab Robotiq 2F-140. Play.py sweeps finger_joint 0 → 0.7."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import ROBOTIQ_2F140_CFG, ROBOTIQ_JOINT_GEAR, make_ee_frame
from isaaclab_hiveboard.assets.end_effector import ANYMAL_EE, EndEffectorCfg

# Same TCP offset as ANYmal, but prims live at the gripper root (no DynaArm prefix).
ROBOTIQ_DEBUG_EE = EndEffectorCfg(
    body_name=ANYMAL_EE.body_name,
    source_prim=ANYMAL_EE.body_name,
    body_prim=ANYMAL_EE.body_name,
    tcp_offset=ANYMAL_EE.tcp_offset,
)


@configclass
class AnymalOnlyGripperSceneCfg(InteractiveSceneCfg):
    """Just the 2F-140, a TCP marker, and a light."""

    robot: ArticulationCfg = ROBOTIQ_2F140_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame: FrameTransformerCfg = make_ee_frame(ROBOTIQ_DEBUG_EE, debug_vis=True)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Drive ``finger_joint`` and inner fingers at ``-q`` (parallel jaw)."""

    gripper_action = JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ROBOTIQ_JOINT_GEAR),
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class AnymalOnlyGripperEnvCfg(ManagerBasedRLEnvCfg):
    """Watch mimic signs on the 2F-140 without ANYmal, the DynaArm, or a valve."""

    scene: AnymalOnlyGripperSceneCfg = AnymalOnlyGripperSceneCfg(num_envs=1, env_spacing=2.0)  # type: ignore
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
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "robot"
        self.viewer.body_name = ROBOTIQ_DEBUG_EE.body_name
        self.viewer.env_index = 0
        self.viewer.eye = (-0.35, 0.28, 0.18)
        self.viewer.lookat = (0.0, 0.0, 0.08)
        self.viewer.resolution = (1280, 720)
