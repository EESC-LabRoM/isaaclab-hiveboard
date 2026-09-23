from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from .configs.actions import FrankaButtonActionsCfg
from .configs.commands import FramePoseCommandsCfg
from .configs.events import FrankaButtonEventCfg
from .configs.observations import ObservationsCfg
from .configs.scene import FrankaButtonSceneCfg
from .configs.terminations import TerminationsCfg


@configclass
class FrankaButtonEnvCfg(ManagerBasedRLEnvCfg):
    """Franka opens the protective cover and fully depresses the hidden button."""

    scene: FrankaButtonSceneCfg = FrankaButtonSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: ObservationsCfg = ObservationsCfg()  # type: ignore
    actions: FrankaButtonActionsCfg = FrankaButtonActionsCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    events: FrankaButtonEventCfg = FrankaButtonEventCfg()  # type: ignore
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()  # type: ignore
    rewards = None

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 45.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "button"
        self.viewer.body_name = "button_pivot"
        self.viewer.env_index = 0
        self.viewer.eye = (-0.8, 1.0, 0.8)
        self.viewer.lookat = (0.55, 0.0, 0.10)
        self.sim.dt = 1 / 200
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.00625
