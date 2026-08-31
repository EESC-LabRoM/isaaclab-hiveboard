from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.mdp.events import RandomizeValveHandlePoseEvent
from isaaclab_hiveboard.tasks.spot.lamp.configs.scene import (
    LAMP_SPAWN_POS,
    LAMP_SPAWN_QUAT,
    LAMP_ROOT_POS_IN_ROBOT_BASE,
    LAMP_RESET_APPROACH_OFFSET,
    LAMP_URDF,
)


@configclass
class LampEventCfg:
    """Reset the lamp and place Spot's TCP at the approach frame."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["arm_link_fngr", "arm_link_wr1"]
            ),
            "static_friction_range": (0.6, 0.6),
            "dynamic_friction_range": (0.5, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    lamp_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("lamp", body_names=["lamp_pivot"]),
            "static_friction_range": (0.6, 0.6),
            "dynamic_friction_range": (0.5, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=RandomizeValveHandlePoseEvent,  # type: ignore
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["arm.*"]),
            "valve_cfg": SceneEntityCfg("lamp", body_names=["rotation_pivot"]),
            "valve_urdf": LAMP_URDF,
            "valve_root_link": "World",
            "valve_ee_link": "rotation_pivot",
            "valve_root_pose": OffsetCfg(
                pos=LAMP_ROOT_POS_IN_ROBOT_BASE, rot=LAMP_SPAWN_QUAT
            ),
            "valve_offset": LAMP_RESET_APPROACH_OFFSET,
            "n_x": 1,
            "n_y": 1,
            "n_z": 1,
            "n_yaw": 1,
            "max_x": 0.0,
            "max_y": 0.0,
            "max_z": 0.0,
            "max_yaw": 0.0,
        },
    )
