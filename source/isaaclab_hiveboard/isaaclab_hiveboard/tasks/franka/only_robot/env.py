"""Minimal stationary Franka scene for inspecting the TCP axes."""

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

from isaaclab_hiveboard.assets import FRANKA_EE, FRANKA_FR3_HIGH_PD_CFG, make_ee_frame


@configclass
class FrankaOnlyRobotSceneCfg(InteractiveSceneCfg):
    """FR3 and its TCP marker, without any warehouse or task assets."""

    robot: ArticulationCfg = FRANKA_FR3_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "fr3_joint1": 0.0,
                "fr3_joint2": -0.569,
                "fr3_joint3": 0.0,
                "fr3_joint4": -2.810,
                "fr3_joint5": 0.0,
                "fr3_joint6": 3.037,
                "fr3_joint7": 0.741,
                "fr3_finger_joint.*": 0.04,
            },
        ),
    )
    # Red, green, and blue arrows show the physical TCP axes directly.
    ee_frame: FrameTransformerCfg = make_ee_frame(FRANKA_EE, debug_vis=True)

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
class FrankaOnlyRobotEnvCfg(ManagerBasedRLEnvCfg):
    """Fast-starting, fixed-base FR3 TCP-frame inspection environment."""

    scene: FrankaOnlyRobotSceneCfg = FrankaOnlyRobotSceneCfg(num_envs=1, env_spacing=2.0)  # type: ignore
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
        self.viewer.body_name = FRANKA_EE.body_name
        self.viewer.env_index = 0
        self.viewer.eye = (-0.65, 0.65, 0.35)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.resolution = (1280, 720)
