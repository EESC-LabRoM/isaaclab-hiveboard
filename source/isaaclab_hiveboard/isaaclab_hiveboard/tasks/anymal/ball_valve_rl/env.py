# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RL (teacher/student) configuration for the fixed-base ANYmal + DynaArm ball valve.

The scene, robot, valve and domain randomization match the scripted-expert
task in ``tasks/anymal/ball_valve``; what changes is the MDP:

* Action: integrated arm joint-position increments (7 = 6 arm + 1 gripper).
  The gripper is binary (negative closes), ramped like the expert's.
* ``teacher`` observations: privileged simulator state, used by PPO for both
  actor and critic.
* ``policy`` observations: what the robot has at deployment - arm encoders,
  gripper angle, forward-kinematics TCP pose, previous action, and the
  registered valve pose with a per-episode registration error - stacked over
  a short history. The student is distilled from the teacher on these.
"""

import os

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_tasks.core.cabinet import mdp as base_mdp
from isaaclab_tasks.utils import PresetCfg

from isaaclab_hiveboard.assets.anymal.bench import (
    ANYMAL_ARM_JOINT_NAMES,
    ANYMAL_NEWTON_GRIPPER_CLOSE,
    ANYMAL_NEWTON_GRIPPER_OPEN,
    NEWTON_GRIPPER_JOINT_NAMES,
)
from isaaclab_hiveboard.mdp.actions import RateLimitedBinaryJointPositionActionCfg
from isaaclab_hiveboard.mdp.events import apply_articulation_gravcomp
from isaaclab_hiveboard.tasks.anymal.ball_valve.configs.scene import BallValveSceneCfg
from isaaclab_hiveboard.tasks.anymal.ball_valve_rl import expert_bank, mdp

ARM = SceneEntityCfg("robot", joint_names=list(ANYMAL_ARM_JOINT_NAMES), preserve_order=True)
GRIPPER = SceneEntityCfg("robot", joint_names=list(NEWTON_GRIPPER_JOINT_NAMES), preserve_order=True)
VALVE_JOINT = SceneEntityCfg("ball_valve", joint_names=[mdp.VALVE_JOINT])

# Same tolerance as the scripted-expert task's success term.
SUCCESS_TOLERANCE_RAD = 0.035
# The lever counts as held when the gripper is closed within these TCP errors
# [m, rad] of the expert grasp pose; the same gate scores grasp, turn and open.
HOLD = {"dist_threshold": 0.03, "ang_threshold": 0.35}

# Reset distribution, shared with scripts/rl/build_expert_bank.py so the bank's
# trajectories start from the same states the uniform resets would draw.
VALVE_POSE_RANGE = {"x": (-0.03, 0.03), "y": (-0.04, 0.04), "z": (-0.03, 0.03), "yaw": (-0.1, 0.1)}
"""Valve root pose offset [m, rad]: board placement beyond the student's registration error."""
VALVE_ANGLE_RANGE = (-0.4, 0.0)
"""Initial valve angle offset [rad]: some episodes start part-open."""
ARM_RANGE = (-0.1, 0.1)
"""Initial arm joint offset [rad]."""

# Precomputed cuRobo expert trajectories (scripts/rl/build_expert_bank.py).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 6))
EXPERT_BANK_PATH = os.path.join(REPO_ROOT, "logs", "expert_bank", "anymal_ball_valve_bank_5000.pt")


@configclass
class BallValveRLPhysicsCfg(PresetCfg):
    """Newton MJWarp, sized for thousands of parallel worlds.

    ``nconmax``/``njmax`` are per world; the scripted task's 4000 contacts per
    world only fits a handful of environments.
    """

    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            njmax=600,
            nconmax=300,
            iterations=100,
            ls_iterations=20,
            impratio=10.0,
            ccd_iterations=50,
            use_mujoco_contacts=False,
        ),
        num_substeps=2,
        debug_mode=False,
        # Startup gravity-compensation changes must be evaluated by the solver.
        use_cuda_graph=False,
    )
    default = newton_mjwarp


@configclass
class ActionsCfg:
    """Integrated arm joint increments plus a ramped binary gripper."""

    arm_action = mdp.IntegratedJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ANYMAL_ARM_JOINT_NAMES),
        preserve_order=True,
        # 0.1 rad per 20 Hz step = 2 rad/s at |a| = 1 (clip_actions = 1). The
        # cuRobo expert's shoulder and wrist flexion reach 1.7-1.8 rad/s.
        scale=0.1,
    )
    gripper_action = RateLimitedBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(NEWTON_GRIPPER_JOINT_NAMES),
        open_command_expr=dict(zip(NEWTON_GRIPPER_JOINT_NAMES, ANYMAL_NEWTON_GRIPPER_OPEN)),
        close_command_expr=dict(zip(NEWTON_GRIPPER_JOINT_NAMES, ANYMAL_NEWTON_GRIPPER_CLOSE)),
        close_speed=2.0,
    )


@configclass
class ObservationsCfg:
    """Deployable ``policy`` (student) and privileged ``teacher`` groups."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Proprioception + registered target pose; available on the real robot."""

        arm_joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel, params={"asset_cfg": ARM}, noise=Unoise(n_min=-0.002, n_max=0.002)
        )
        gripper_pos = ObsTerm(
            func=base_mdp.joint_pos, params={"asset_cfg": GRIPPER}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        tcp_pose = ObsTerm(func=mdp.tcp_pose_b)
        registered_valve = ObsTerm(
            func=mdp.registered_valve_b,
            # ~1 cm / ~2 deg registration error and ~2 deg on the lever's
            # initial angle, latched per episode; 2 mm jitter per step.
            params={"bias_pos": 0.01, "bias_rot": 0.035, "jitter_pos": 0.002, "angle_noise": 0.035},
        )
        last_action = ObsTerm(func=base_mdp.last_action)
        # Operator input on the robot: how fast to turn the valve.
        turn_rate = ObsTerm(func=mdp.turn_rate_command, params={"command_name": "valve_turn"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5

    @configclass
    class TeacherCfg(ObsGroup):
        """Privileged simulator state for the PPO teacher's actor and critic."""

        arm_joint_pos = ObsTerm(func=base_mdp.joint_pos_rel, params={"asset_cfg": ARM})
        arm_joint_vel = ObsTerm(func=base_mdp.joint_vel_rel, params={"asset_cfg": ARM}, scale=0.1)
        gripper_pos = ObsTerm(func=base_mdp.joint_pos, params={"asset_cfg": GRIPPER})
        tcp_pose = ObsTerm(func=mdp.tcp_pose_b)
        valve_pose = ObsTerm(func=mdp.valve_pose_b)
        grasp_target = ObsTerm(func=mdp.grasp_target_b)
        tcp_to_grasp = ObsTerm(func=mdp.tcp_to_grasp_b, scale=5.0)
        valve_state = ObsTerm(func=mdp.valve_state)
        pad_force = ObsTerm(func=mdp.pad_valve_force, scale=0.02, clip=(0.0, 10.0))
        last_action = ObsTerm(func=base_mdp.last_action)
        # Appended last so a pre-command checkpoint can be warm-started by
        # zero-padding its input layer (scripts/rl/expand_checkpoint_inputs.py).
        turn_rate = ObsTerm(func=mdp.turn_rate_command, params={"command_name": "valve_turn"})
        valve_reference = ObsTerm(func=mdp.valve_reference_state, params={"command_name": "valve_turn"})
        # Distance of each arm joint from the expert's (reach by time, turn by
        # reference angle). Appended last, like the command terms above.
        expert_joint_error = ObsTerm(func=expert_bank.expert_joint_error, params={"command_name": "valve_turn"})
        expert_gripper = ObsTerm(func=expert_bank.expert_gripper_reference, params={"command_name": "valve_turn"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()


@configclass
class EventCfg:
    """Gravity compensation, the scripted task's valve DR, and reset randomization."""

    quiet_solver = EventTerm(func=mdp.silence_solver_overflow_warnings, mode="startup")
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
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_inner_finger", "right_inner_finger"]),
            "static_friction_range": (0.3, 0.6),
            "dynamic_friction_range": (0.3, 0.5),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )
    valve_joint_parameters = EventTerm(
        func=base_mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": VALVE_JOINT,
            "friction_distribution_params": (0.01, 0.10),
            "armature_distribution_params": (0.001, 0.01),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    valve_physics_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
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

    # Reset order matters: defaults first, then the randomized offsets. The
    # integrated arm action then starts from the resulting joints (it resets
    # after the events).
    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})
    reset_valve_root = EventTerm(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Board placement relative to the robot, beyond the registration
            # error the student sees: the policy must use the registered pose.
            "pose_range": dict(VALVE_POSE_RANGE),
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("ball_valve"),
        },
    )
    reset_valve_joint = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        # Start some episodes part-open so the turning phase gets explored
        # early (a sampling-based curriculum in the IndustReal sense).
        params={"position_range": VALVE_ANGLE_RANGE, "velocity_range": (0.0, 0.0), "asset_cfg": VALVE_JOINT},
    )
    reset_arm = EventTerm(
        func=base_mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": ARM_RANGE, "velocity_range": (0.0, 0.0), "asset_cfg": ARM},
    )
    # Replaces the three uniform resets above when an expert bank is used
    # (see AnymalBallValveRLEnvCfg.__post_init__): same distribution, but each
    # episode starts exactly where a successful expert trajectory started.
    reset_from_bank = expert_bank.reset_from_expert_bank_cfg(EXPERT_BANK_PATH)


@configclass
class CommandsCfg:
    """Operator-set turning speed; its reference angle is what the turn is scored against."""

    valve_turn = mdp.ValveTurnRateCommandCfg(rate_range=(0.25, 0.8), hold_dist=0.03, hold_ang=0.35)


@configclass
class RewardsCfg:
    """Expert-tracked reach and grasp, then tracking of the commanded turn.

    Once the lever is held, the reward is for following the reference angle of
    the ``valve_turn`` command, and the per-step "open" bonus only pays once the
    reference itself has reached open. Turning faster than commanded therefore
    earns nothing; the first teachers, rewarded for progress, flicked the lever
    open at 6-7 rad/s.

    Episodes do not end on success: a terminal bonus would be outweighed by the
    dense reward forfeited for the rest of the episode, teaching the policy to
    stop just short of open. Holding the valve open instead pays every step.
    """

    grasp = RewTerm(func=mdp.grasp_lever, weight=2.0, params=dict(HOLD))
    align_held = RewTerm(func=mdp.align_held, weight=2.0, params={"std": 0.1, **HOLD})
    track_turn = RewTerm(func=mdp.track_valve_reference, weight=5.0, params={"std": 0.1, **HOLD})
    # Follow the expert joint by joint, not only at the TCP: near the wrist
    # singularity a small TCP error can hide large forearm/wrist excursions.
    # The reach is shaped only by these: Cartesian reach terms paid for being
    # at the lever as early as possible, which pulled teacher v11 off the
    # expert's paced joint path onto a fast approach that arrived 0.5-0.8 rad
    # off in orientation. The coarse kernel keeps a gradient back to the
    # expert when the arm is far from it; the fine one asks for precision.
    track_expert_coarse = RewTerm(func=expert_bank.track_expert_joints, weight=2.0, params={"std": 0.5})
    track_expert = RewTerm(func=expert_bank.track_expert_joints, weight=3.0, params={"std": 0.15})
    # Close the gripper when the expert does (teacher v10, rewarded on joints
    # only, reached the lever every time but never closed it in 600 iterations).
    track_expert_gripper = RewTerm(func=expert_bank.track_expert_gripper, weight=1.0)
    success = RewTerm(
        func=mdp.valve_opened_on_schedule, weight=10.0, params={"threshold_rad": SUCCESS_TOLERANCE_RAD, **HOLD}
    )
    # Strong enough that the raw actions are smooth themselves, not only after
    # the action filter: teacher v7 (-0.05, no magnitude term) dithered between
    # +-1 every step behind the filter (lag-1 autocorrelation -0.85), which the
    # student could not imitate and a real drive should not receive.
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.5)
    action_magnitude = RewTerm(func=base_mdp.action_l2, weight=-0.02)
    arm_joint_vel = RewTerm(func=base_mdp.joint_vel_l2, weight=-1.0e-3, params={"asset_cfg": ARM})
    pad_force = RewTerm(func=mdp.pad_force_excess, weight=-1.0e-3, params={"max_force": 60.0})
    valve_overspeed = RewTerm(func=mdp.valve_rate_excess, weight=-1.0, params={"factor": 1.5})
    valve_rate_deviation = RewTerm(func=mdp.valve_rate_deviation, weight=-2.0)
    valve_unheld_motion = RewTerm(func=mdp.valve_unheld_motion, weight=-2.0, params=dict(HOLD))


@configclass
class TerminationsCfg:
    """Timeout and solver blow-up (success is scored, not terminated; see RewardsCfg)."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    invalid = DoneTerm(func=mdp.invalid_state, params={"asset_cfg": ARM})


@configclass
class AnymalBallValveRLEnvCfg(ManagerBasedRLEnvCfg):
    """ANYmal + DynaArm opens the HiveBoard ball valve (closed -> -90 deg) with RL."""

    scene: BallValveSceneCfg = BallValveSceneCfg(num_envs=1024, env_spacing=3.0)  # type: ignore
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=10, physics=BallValveRLPhysicsCfg())  # type: ignore

    def __post_init__(self):
        # 20 Hz policy. The slowest commanded turn (0.25 rad/s) takes 6.3 s after
        # a ~1.5 s reach, leaving ~4 s of held-open time within 12 s.
        self.decimation = 10
        self.episode_length_s = 12.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "ball_valve"
        self.viewer.body_name = "alavanca_pivot"
        self.viewer.eye = (-1.5, 1.5, 0.5)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # Bank resets replace the uniform ones; they come from the same ranges.
        self.events.reset_valve_root = None
        self.events.reset_valve_joint = None
        self.events.reset_arm = None


@configclass
class AnymalBallValveRLEnvCfg_PLAY(AnymalBallValveRLEnvCfg):
    """Evaluation: nominal physics, bank starts (randomized like training), registration error kept."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.events.robot_physics_material = None
        self.events.valve_joint_parameters = None
        self.events.valve_physics_material = None
        self.observations.policy.enable_corruption = False
        # The cuRobo expert's turning speed; sweep with
        # env.commands.valve_turn.rate_range=[r,r].
        self.commands.valve_turn.rate_range = (0.3, 0.3)
