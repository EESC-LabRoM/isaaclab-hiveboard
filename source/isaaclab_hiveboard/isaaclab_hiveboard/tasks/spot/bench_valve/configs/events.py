# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp


@configclass
class BenchValveEventCfg:
    """Deterministic reset to the website home pose. No domain randomization."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
