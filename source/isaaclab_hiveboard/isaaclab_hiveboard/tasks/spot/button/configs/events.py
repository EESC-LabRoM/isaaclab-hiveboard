# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.assets import Articulation
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.assets import BUTTON_URDF
from isaaclab_hiveboard.mdp.events import (
    RandomizeValveHandlePoseEvent,
    reset_joint_position,
)


def latch_open_lid(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    open_threshold: float = -0.75,
    hold_stiffness: float = 16.0,
    hold_damping: float = 4.0,
) -> None:
    """Hold the cover once the TCP has driven it past ``open_threshold``.

    Zero-stiffness hinge otherwise, so the closed gripper still has to open it.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = asset.find_joints(asset_cfg.joint_names[0], preserve_order=True)
    jid = joint_ids[0]
    lid = asset.data.joint_pos[:, jid]
    is_open = lid > open_threshold
    stiffness = torch.zeros(asset.num_instances, device=asset.device)
    damping = torch.zeros_like(stiffness)
    stiffness[is_open] = hold_stiffness
    damping[is_open] = hold_damping
    asset.write_joint_stiffness_to_sim(stiffness, joint_ids=[jid])
    asset.write_joint_damping_to_sim(damping, joint_ids=[jid])
    if is_open.any():
        opened = torch.zeros(asset.num_instances, 1, device=asset.device)
        asset.set_joint_position_target(opened, joint_ids=[jid])


@configclass
class ButtonEventCfg:
    """Reset Spot onto the approach frame with the cover closed and button up."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["arm_link_fngr", "arm_link_wr1"]
            ),
            "static_friction_range": (0.3, 0.3),
            "dynamic_friction_range": (0.3, 0.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=RandomizeValveHandlePoseEvent,  # type: ignore
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["arm.*"]),
            "valve_cfg": SceneEntityCfg("button", body_names=["lid_pivot"]),
            "valve_urdf": BUTTON_URDF,
            "valve_root_link": "World",
            "valve_ee_link": "lid_pivot",
            "frame_name": "target_frame",
            "target_frame_name": "approaching",
            "n_x": 1,
            "n_y": 1,
            "n_z": 1,
            "n_yaw": 1,
            "max_x": 0.0,
            "max_y": 0.0,
            "max_z": 0.0,
            "max_yaw": 0.0,
            "valve_joint_range": (-1.56, -1.56),
            "n_valve_states": 1,
            "align_rotation": False,
        },
    )
    reset_lid_closed = EventTerm(
        func=reset_joint_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("button", joint_names=["RevoluteJoint"]),
            "position": -1.56,
        },
    )
    reset_button_unpressed = EventTerm(
        func=reset_joint_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("button", joint_names=["PrismaticJoint"]),
            "position": 0.0,
        },
    )
    latch_lid = EventTerm(
        func=latch_open_lid,
        mode="interval",
        interval_range_s=(0.025, 0.025),
        params={
            "asset_cfg": SceneEntityCfg("button", joint_names=["RevoluteJoint"]),
            "open_threshold": -0.75,
            "hold_stiffness": 16.0,
            "hold_damping": 4.0,
        },
    )
