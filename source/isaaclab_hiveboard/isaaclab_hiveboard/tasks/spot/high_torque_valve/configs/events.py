from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.tasks.spot.high_torque_valve.configs.scene import (
    HIGH_TORQUE_VALVE_URDF,
    VALVE_Y90_QUAT,
)
from isaaclab_hiveboard.mdp.events import (
    RandomizeValveHandlePoseEvent,
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
            "valve_cfg": SceneEntityCfg("high_torque_valve", body_names=["nut"]),
            "valve_urdf": HIGH_TORQUE_VALVE_URDF,
            "valve_root_link": "World",
            "valve_ee_link": "nut",
            "ee_offset": OffsetCfg(
                pos=(0.21, 0.0, -0.03),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            # Align TCP with the approach frame so the arm only moves forward.
            "valve_offset": OffsetCfg(
                pos=(0.0, 0.0, 0.26),
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
