import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.tasks.spot.ball_valve.configs.scene import (
    BALL_VALVE_URDF,
)
from isaaclab_hiveboard.mdp.events import (
    RandomizeValveHandlePoseEvent,
)

PI = 355 / 113


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

    # Valve actuator and joint-domain randomization. These are startup terms,
    # matching ALMA-D's collection setup: each simulator instance receives a
    # fixed physical model while episodes still vary in pose below.
    valve_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball_valve", joint_names=["RevoluteJoint"]),
            "stiffness_distribution_params": (5.0e-5, 2.0e-4),
            "damping_distribution_params": (1.0e-5, 5.0e-5),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    valve_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball_valve", joint_names=["RevoluteJoint"]),
            "friction_distribution_params": (0.01, 0.10),
            "armature_distribution_params": (0.001, 0.01),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    valve_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball_valve"),
            "static_friction_range": (0.2, 1.0),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "make_consistent": True,
        },
    )
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joints = EventTerm(
        func=RandomizeValveHandlePoseEvent,  # type: ignore
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["arm.*"]),
            "valve_cfg": SceneEntityCfg("ball_valve", body_names=["alavanca_pivot"]),
            "valve_urdf": BALL_VALVE_URDF,
            "valve_root_link": "valvula_esfera",
            "valve_ee_link": "alavanca_pivot",
            # Valve-first reset pose in the fixed robot-base frame.  At zero
            # valve angle, the approach TCP is (1.02, 0.0, 0.154) with identity
            # orientation, matching Spot's nominal arm pose.  The pi yaw makes
            # the front of the HiveBoard valve face the robot.
            "valve_root_pose": OffsetCfg(
                pos=(1.14, 0.06, 0.154),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
            "ee_offset": OffsetCfg(
                pos=(0.21, 0.0, -0.03),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            # Align TCP with the *approach* frame, not the grasp. The command
            # sequence then only moves forward onto the lever.
            "valve_offset": OffsetCfg(
                pos=(0.12, 0.06, 0.0),
                rot=(0.0, 0.0, 0.0, 1.0),
            ),
            # Set on-the-fly to True to generate reachable reset states dynamically on each reset
            "on_the_fly": True,
            # Keep perturbations inside the tested Spot reachability margin.
            "max_x": 0.20,
            "max_y": 0.30,
            "max_z": 0.30,
            "max_roll": PI / 6,
            "max_pitch": PI / 6,
            "max_yaw": PI / 5,
            # Keep starts at least 0.35 rad from either endpoint so both open
            # and close tasks remain feasible. Each state has a matching IK pose.
            "valve_joint_range": (-torch.pi / 2 + 0.35, -0.35),
        },
    )
