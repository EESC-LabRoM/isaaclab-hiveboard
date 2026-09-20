"""Kitless Isaac Lab 3 lamp task using Newton MJWarp."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from .configs.actions import SpotLampActionCfg
from .configs.commands import FramePoseCommandsCfg
from .configs.events import LampEventCfg
from .configs.observations import ObservationsCfg
from .configs.scene import LampSceneCfg
from .configs.terminations import TerminationsCfg


@configclass
class LampPhysicsCfg(PresetCfg):
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            iterations=100,
            ls_iterations=20,
            njmax=600,
            nconmax=4000,
        ),
        num_substeps=2,
        use_cuda_graph=False,
    )
    default = newton_mjwarp


@configclass
class SpotLampEnvCfg(ManagerBasedRLEnvCfg):
    """Seat the lamp with sixteen quarter turns and wrist unwinds."""

    scene: LampSceneCfg = LampSceneCfg(num_envs=1, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: SpotLampActionCfg = SpotLampActionCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: LampEventCfg = LampEventCfg()
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=15, physics=LampPhysicsCfg())
    rewards = None
    recorders = None

    def __post_init__(self):
        self.decimation = 15
        self.episode_length_s = 85.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "lamp"
        self.viewer.body_name = "lamp_pivot"
        self.viewer.env_index = 0
        self.viewer.eye = (-0.5, 0.5, 0.2)
        self.viewer.lookat = (0.0, 0.0, 0.0)
