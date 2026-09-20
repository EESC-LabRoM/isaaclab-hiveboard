"""Franka lamp environment for Isaac Lab 3 and Newton."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.spot.lamp.env import SpotLampEnvCfg

from .configs.actions import FrankaLampActionCfg
from .configs.commands import FramePoseCommandsCfg
from .configs.events import FrankaLampEventCfg
from .configs.observations import ObservationsCfg
from .configs.scene import FrankaLampSceneCfg
from .configs.terminations import TerminationsCfg


@configclass
class FrankaLampEnvCfg(SpotLampEnvCfg):
    scene: FrankaLampSceneCfg = FrankaLampSceneCfg(num_envs=1, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: FrankaLampActionCfg = FrankaLampActionCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: FrankaLampEventCfg = FrankaLampEventCfg()
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()
