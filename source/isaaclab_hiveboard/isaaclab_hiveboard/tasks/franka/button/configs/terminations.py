from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from .scene import BUTTON_PRESSED_POSITION, MINIMUM_COVER_SUCCESS_POSITION

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def button_task_success(
    env: ManagerBasedRLEnv,
    command_name: str,
    button_cfg: SceneEntityCfg,
    cover_cfg: SceneEntityCfg,
    button_tolerance: float,
    minimum_cover_angle: float,
) -> torch.Tensor:
    """Require a 90-degree-open cover and the complete 10 mm button stroke."""
    command = env.command_manager.get_term(command_name)
    asset = env.scene[button_cfg.name]
    button_position = asset.data.joint_pos[:, button_cfg.joint_ids][:, 0]
    cover_position = asset.data.joint_pos[:, cover_cfg.joint_ids][:, 0]
    button_pressed = torch.abs(button_position - BUTTON_PRESSED_POSITION) <= button_tolerance
    cover_open = cover_position >= minimum_cover_angle
    return command.is_done() & cover_open & button_pressed


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=button_task_success,
        params={
            "command_name": "pose_command",
            "button_cfg": SceneEntityCfg("button", joint_names=["PrismaticJoint"]),
            "cover_cfg": SceneEntityCfg("button", joint_names=["RevoluteJoint"]),
            "button_tolerance": 0.001,
            "minimum_cover_angle": MINIMUM_COVER_SUCCESS_POSITION,
        },
    )
