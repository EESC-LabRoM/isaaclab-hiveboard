from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.mdp.events import (
    RandomizeValveHandlePoseEvent,
)
from isaaclab_hiveboard.tasks.spot.small_valve.configs.scene import (
    SMALL_VALVE_URDF,
    VALVE_SPAWN_POS,
    VALVE_SPAWN_QUAT,
    VALVE_Y90_QUAT,
)


@configclass
class ValveEventCfg:
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
            "valve_cfg": SceneEntityCfg("small_valve", body_names=["eixo_trans"]),
            "valve_urdf": SMALL_VALVE_URDF,
            "valve_root_link": "valvula_gaveta",
            "valve_ee_link": "eixo_trans",
            # Valve stays at its scene pose; IK the arm to it. Without this
            # the event teleports the valve onto the retract TCP (the robot).
            "valve_root_pose": OffsetCfg(
                pos=VALVE_SPAWN_POS,
                rot=VALVE_SPAWN_QUAT,
            ),
            # TCP offset comes from pose_command.body_offset. Spawn offset is
            # still explicit: it is closer than target_frame/approaching.
            "valve_offset": OffsetCfg(
                pos=(0.0, 0.0, 0.10),
                rot=VALVE_Y90_QUAT,
            ),
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
