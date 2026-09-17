from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.tasks.franka.circuit_breaker.configs.observations import ObservationsCfg
from isaaclab_hiveboard.tasks.franka.gate_valve_small.configs.actions import FrankaGateValveSmallActionsCfg
from isaaclab_hiveboard.tasks.franka.gate_valve_small.configs.commands import FramePoseCommandsCfg
from isaaclab_hiveboard.tasks.franka.gate_valve_small.configs.events import FrankaGateValveSmallEventCfg
from isaaclab_hiveboard.tasks.franka.gate_valve_small.configs.scene import FrankaGateValveSmallSceneCfg
from isaaclab_hiveboard.tasks.franka.gate_valve_small.configs.terminations import TerminationsCfg


@configclass
class FrankaGateValveSmallEnvCfg(ManagerBasedRLEnvCfg):
    """Franka opening the small gate valve one full turn through regrasping."""

    scene = FrankaGateValveSmallSceneCfg(num_envs=1, env_spacing=3.0)
    observations = ObservationsCfg()
    actions = FrankaGateValveSmallActionsCfg()
    terminations = TerminationsCfg()
    events = FrankaGateValveSmallEventCfg()
    commands = FramePoseCommandsCfg()
    rewards = None

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 60.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "small_valve"
        self.viewer.body_name = "valvula_gaveta"
        self.viewer.env_index = 0
        self.viewer.eye = (1.0, 1.0, 0.8)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        self.viewer.resolution = (2560, 1440)
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
