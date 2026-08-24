from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.assets import ASSET_DIR, CIRCUIT_BREAKER_URDF, FRANKA_EE
from isaaclab_hiveboard.mdp.events import (
    RandomizeValveHandlePoseEvent,
    reset_joint_position,
)


@configclass
class FrankaCircuitBreakerEventCfg:
    """Configuration for Franka events."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["fr3_leftfinger", "fr3_rightfinger"]
            ),
            "static_friction_range": (0.4, 0.4),
            "dynamic_friction_range": (0.4, 0.4),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # Shared reset implementation: the TCP offset comes from pose_command's
    # body_offset, exactly as it does for Spot and runtime control.
    reset_robot_joints = EventTerm(
        func=RandomizeValveHandlePoseEvent,  # type: ignore
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["fr3_joint[1-7]"]),
            "valve_cfg": SceneEntityCfg(
                "circuit_breaker", body_names=["lever_pivot"]
            ),
            "valve_urdf": CIRCUIT_BREAKER_URDF,
            "valve_root_link": "World",
            "valve_ee_link": "lever_pivot",
            "robot_ee_body_name": FRANKA_EE.body_name,
            "robot_curobo_yaml": f"{ASSET_DIR}/franka/cumotion/fr3.yaml",
            "robot_urdf": f"{ASSET_DIR}/franka/cumotion/fr3.urdf",
            # This kinematic model has no collision meshes; reset IK only
            # needs the joint chain and limits.
            "self_collision_check": False,
            "frame_name": "target_frame",
            "target_frame_name": "approaching",
            "n_x": 1,
            "n_y": 1,
            "n_z": 1,
            "n_roll": 1,
            "n_pitch": 1,
            "n_yaw": 1,
            "max_x": 0.0,
            "max_y": 0.0,
            "max_z": 0.0,
            "max_roll": 0.0,
            "max_pitch": 0.0,
            "max_yaw": 0.0,
            "valve_joint_range": (0.0, 0.0),
            "align_rotation": False,
        },
    )
    # URDF default is +30 deg (UP). Force DOWN so below→above can drive the paddle.
    reset_breaker_down = EventTerm(
        func=reset_joint_position,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "circuit_breaker", joint_names=["RevoluteJoint"]
            ),
            "position": 0.0,
        },
    )
