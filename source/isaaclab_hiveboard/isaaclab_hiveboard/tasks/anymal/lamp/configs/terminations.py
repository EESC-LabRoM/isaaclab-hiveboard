# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.tasks.spot.lamp.configs.terminations import TerminationsCfg as LampTerminationsCfg


@configclass
class TerminationsCfg(LampTerminationsCfg):
    """Shared lamp terminations for ANYmal + DynaArm."""
