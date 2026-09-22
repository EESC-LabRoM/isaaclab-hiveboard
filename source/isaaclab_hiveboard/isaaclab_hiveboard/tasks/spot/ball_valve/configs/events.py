import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from isaaclab_hiveboard.mdp.events import apply_articulation_gravcomp

PI = math.pi


@configclass
class ValveEventCfg:
    """Configuration for events."""

    robot_gravcomp = EventTerm(
        func=apply_articulation_gravcomp,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot"), "gravcomp": 1.0},
    )
    valve_gravcomp = EventTerm(
        func=apply_articulation_gravcomp,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("ball_valve"), "gravcomp": 1.0},
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
    # WARNING: this term currently makes the task unsolvable by the scripted
    # cuRobo expert. Holding every other setting fixed and toggling only this
    # one, the expert seats the valve in 3/3 episodes with it disabled and 0/8
    # with it enabled - the sampled joint friction resists the gripper for the
    # whole range, not just its upper end. Until the range is retuned against
    # the gripper's achievable torque, demonstration collection has to disable
    # it (see the imitation-learning section of the README).
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
    # Reset the complete scene first, then apply the valve-specific pose and
    # angle below. Event terms run in declaration order.
    reset_all = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    # Isaac Lab's reset terms operate on the asset's scene ``init_state``.  The
    # pose ranges below are offsets from that configured root pose (see
    # ``LeverValveSceneCfg.ball_valve.init_state``), in metres and radians.
    # Set a range to ``(v, v)`` for a fixed custom value.
    reset_valve_root = EventTerm(
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
            "asset_cfg": SceneEntityCfg("ball_valve"),
        },
    )
    reset_valve_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("ball_valve", joint_names=["RevoluteJoint"]),
        },
    )


@configclass
class ValvePlayEventCfg(ValveEventCfg):
    """One fixed, reachable reset with no domain randomization."""

    robot_physics_material = None
    valve_actuator_gains = None
    valve_joint_parameters = None
    valve_physics_material = None

    def __post_init__(self):
        # Keep the play task deterministic while retaining the same standard
        # Isaac Lab reset functions used by the randomized task.
        self.reset_valve_root.params["pose_range"] = {
            key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
        }
        self.reset_valve_joint.params["position_range"] = (0.0, 0.0)
