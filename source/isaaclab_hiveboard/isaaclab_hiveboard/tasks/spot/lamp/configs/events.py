"""Deterministic lamp resets and Newton gravity compensation."""

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.mdp.events import apply_articulation_gravcomp


@configclass
class LampEventCfg:
    robot_gravcomp = EventTerm(
        func=apply_articulation_gravcomp,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "gravcomp": 1.0},
    )
    lamp_gravcomp = EventTerm(
        func=apply_articulation_gravcomp,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("lamp"), "gravcomp": 1.0},
    )
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
