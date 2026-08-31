from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaaclab_hiveboard.tasks.franka.circuit_breaker.configs.actions import FrankaIKAbsActionCfg
from .configs.scene import FrankaLampSceneCfg
from .configs.commands import FramePoseCommandsCfg
from .configs.observations import ObservationsCfg
from .configs.terminations import TerminationsCfg
from .configs.events import FrankaLampEventCfg

@configclass
class FrankaLampEnvCfg(ManagerBasedRLEnvCfg):
    scene: FrankaLampSceneCfg = FrankaLampSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: ObservationsCfg = ObservationsCfg()  # type: ignore
    actions: FrankaIKAbsActionCfg = FrankaIKAbsActionCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    events: FrankaLampEventCfg = FrankaLampEventCfg()  # type: ignore
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()  # type: ignore
    rewards = None

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 20.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "lamp"
        self.viewer.body_name = "lamp_pivot"
        self.viewer.eye = (-1.0, 1.2, 0.8)
        self.viewer.lookat = (0.55, 0.0, 0.4)
        self.sim.dt = 1 / 200
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.enable_ccd = False
        # Tip diameter is ~40 mm (gap 0.020 m per finger). Command slightly
        # through that surface instead of driving the fingers to 0.0.
        self.actions.gripper_action.close_command_expr = {"fr3_finger_joint.*": 0.018}
        self.scene.robot.actuators["fr3_hand"].effort_limit_sim = 20.0
        self.scene.robot.actuators["fr3_hand"].stiffness = 200.0
        self.scene.robot.actuators["fr3_hand"].damping = 20.0
