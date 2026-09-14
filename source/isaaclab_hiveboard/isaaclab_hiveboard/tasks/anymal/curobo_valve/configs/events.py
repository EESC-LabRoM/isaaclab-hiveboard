# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.mdp.events import apply_articulation_gravcomp


@configclass
class AnymalCuroboValveEventCfg:
    """Deterministic reset to the bench home pose. No domain randomization."""

    robot_gravcomp = EventTerm(
        func=apply_articulation_gravcomp,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "gravcomp": 1.0},
    )
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
