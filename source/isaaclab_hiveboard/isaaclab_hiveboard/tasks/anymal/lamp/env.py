# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""ANYmal + DynaArm lamp environment for Isaac Lab 3 and Newton."""

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.spot.lamp.env import SpotLampEnvCfg

from .configs.actions import AnymalLampActionCfg
from .configs.commands import FramePoseCommandsCfg
from .configs.events import AnymalLampEventCfg
from .configs.observations import ObservationsCfg
from .configs.scene import AnymalLampSceneCfg
from .configs.terminations import TerminationsCfg


@configclass
class AnymalLampEnvCfg(SpotLampEnvCfg):
    scene: AnymalLampSceneCfg = AnymalLampSceneCfg(num_envs=1, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: AnymalLampActionCfg = AnymalLampActionCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: AnymalLampEventCfg = AnymalLampEventCfg()
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()
