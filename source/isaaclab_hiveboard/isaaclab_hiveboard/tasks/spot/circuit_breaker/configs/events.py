from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.assets import CIRCUIT_BREAKER_URDF
from isaaclab_hiveboard.mdp.events import (
    RandomizeValveHandlePoseEvent,
)


@configclass
class CircuitBreakerEventCfg:
    """Configuration for events."""

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
            "valve_cfg": SceneEntityCfg("circuit_breaker", body_names=["lever_pivot"]),
            "valve_urdf": CIRCUIT_BREAKER_URDF,
            "valve_root_link": "World",
            "valve_ee_link": "lever_pivot",
            # TCP offset comes from pose_command.body_offset. Spawn on the
            # approaching FrameCfg.
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
            "valve_joint_range": (0.5235988, 0.5235988),
            "align_rotation": False,  # Keeps asset base horizontal/unrotated, only applies translation
        },
    )
