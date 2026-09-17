import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.mdp.events import apply_articulation_gravcomp

PI = math.pi


@configclass
class CircuitBreakerEventCfg:
    """Configuration for events."""

    robot_gravcomp = EventTerm(
        func=apply_articulation_gravcomp,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "gravcomp": 1.0},
    )
    breaker_gravcomp = EventTerm(
        func=apply_articulation_gravcomp,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("circuit_breaker"), "gravcomp": 1.0},
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["arm_link_fngr", "arm_link_wr1"]),
            "static_friction_range": (0.3, 0.3),
            "dynamic_friction_range": (0.3, 0.3),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )

    breaker_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("circuit_breaker", joint_names=["RevoluteJoint"]),
            "stiffness_distribution_params": (5.0e-5, 2.0e-4),
            "damping_distribution_params": (1.0e-5, 5.0e-5),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    breaker_joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("circuit_breaker", joint_names=["RevoluteJoint"]),
            "friction_distribution_params": (0.01, 0.10),
            "armature_distribution_params": (0.001, 0.01),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    breaker_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("circuit_breaker"),
            "static_friction_range": (0.2, 1.0),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
            "make_consistent": True,
        },
    )
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    reset_breaker_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.20, 0.20),
                "y": (-0.30, 0.30),
                "z": (-0.30, 0.30),
                "roll": (-PI / 6, PI / 6),
                "pitch": (-PI / 6, PI / 6),
                "yaw": (-PI / 5, PI / 5),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("circuit_breaker"),
        },
    )
    reset_breaker_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("circuit_breaker", joint_names=["RevoluteJoint"]),
        },
    )


@configclass
class CircuitBreakerPlayEventCfg(CircuitBreakerEventCfg):
    """One fixed, reachable reset with no domain randomization."""

    robot_physics_material = None
    breaker_actuator_gains = None
    breaker_joint_parameters = None
    breaker_physics_material = None

    def __post_init__(self):
        self.reset_breaker_root.params["pose_range"] = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.reset_breaker_joint.params["position_range"] = (0.0, 0.0)
