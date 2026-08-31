from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.tasks.spot.ball_valve.configs.actions import SpotIKAbsActionCfg
from isaaclab_hiveboard.tasks.spot.lamp.configs.commands import FramePoseCommandsCfg
from isaaclab_hiveboard.tasks.spot.lamp.configs.events import LampEventCfg
from isaaclab_hiveboard.tasks.spot.lamp.configs.observations import ObservationsCfg
from isaaclab_hiveboard.tasks.spot.lamp.configs.scene import (
    SPOT_FORWARD_OFFSET,
    LampSceneCfg,
)
from isaaclab_hiveboard.tasks.spot.lamp.configs.terminations import TerminationsCfg


@configclass
class SpotLampEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for Spot screwing a lamp into its HiveBoard socket."""

    scene: LampSceneCfg = LampSceneCfg(num_envs=1, env_spacing=3.0)  # type: ignore
    observations: ObservationsCfg = ObservationsCfg()  # type: ignore
    actions: SpotIKAbsActionCfg = SpotIKAbsActionCfg()  # type: ignore
    terminations: TerminationsCfg = TerminationsCfg()  # type: ignore
    events: LampEventCfg = LampEventCfg()  # type: ignore
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()  # type: ignore
    rewards = None

    def __post_init__(self):
        self.decimation = 5 * 5
        self.episode_length_s = 85.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "lamp"
        self.viewer.body_name = "lamp_pivot"
        self.viewer.env_index = 0
        self.viewer.eye = (-0.5, 0.5, 0.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        # self.viewer.resolution = (1080, 720)
        self.sim.dt = 1 / 1000
        # self.sim.physx.bounce_threshold_velocity = 0.2
        # self.sim.physx.friction_correlation_distance = 0.00625
        self.scene.robot.spawn.joint_drive.gains.stiffness = None
        self.scene.robot.spawn.fix_base = True
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count = 16
        self.scene.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
        self.scene.robot.spawn.rigid_props.max_depenetration_velocity = 0.2
        self.scene.robot.actuators["spot_arm_f1x"].effort_limit = 15.32
        self.scene.robot.init_state.pos = (SPOT_FORWARD_OFFSET, 0.0, 0.0)
        # Keep the fingers wide around the thick bulb body.  The lowered TCP
        # lets the lower finger clear the widest section without using the
        # -1.57 rad hard-open limit, whose abrupt expansion inside the bulb
        # was producing a large contact impulse during each regrasp.
        self.actions.gripper_action.open_command_expr = {"arm_f1x": -1.3}
        # As in Forge, command past the expected object surface and let contact
        # stop the gripper instead of prescribing a visually plausible gap.
        self.actions.gripper_action.close_command_expr = {"arm_f1x": -0.6}

        self.sim.physx.enable_ccd = False
        self.sim.physx.solver_type = 1  # TGS
        self.sim.physx.bounce_threshold_velocity = 0.1
        self.sim.physx.friction_correlation_distance = 0.005
        self.sim.physx.enable_enhanced_determinism = True
        self.sim.physx.enable_stabilization = True
