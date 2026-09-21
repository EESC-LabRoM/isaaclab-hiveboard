"""Kitless Isaac Lab 3 lamp task using Newton MJWarp."""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import register_screw_joint_mimic
from isaaclab_tasks.utils import PresetCfg

from .configs.actions import SpotLampActionCfg
from .configs.commands import FramePoseCommandsCfg
from .configs.events import LampEventCfg
from .configs.observations import ObservationsCfg
from .configs.scene import LampSceneCfg
from .configs.terminations import TerminationsCfg


@configclass
class LampPhysicsCfg(PresetCfg):
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            iterations=100,
            ls_iterations=20,
            njmax=600,
            nconmax=4000,
        ),
        num_substeps=2,
        use_cuda_graph=False,
    )
    default = newton_mjwarp


@configclass
class SpotLampEnvCfg(ManagerBasedRLEnvCfg):
    """Seat the lamp with sixteen quarter turns and wrist unwinds."""

    scene: LampSceneCfg = LampSceneCfg(num_envs=1, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: SpotLampActionCfg = SpotLampActionCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: LampEventCfg = LampEventCfg()
    commands: FramePoseCommandsCfg = FramePoseCommandsCfg()
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=15, physics=LampPhysicsCfg())
    rewards = None
    recorders = None

    def __post_init__(self):
        self.decimation = 15
        self.episode_length_s = 10.0
        self.viewer.origin_type = "asset_body"
        self.viewer.asset_name = "lamp"
        self.viewer.body_name = "lamp_pivot"
        self.viewer.env_index = 0
        self.viewer.eye = (-0.5, 0.5, 0.2)
        self.viewer.lookat = (0.0, 0.0, 0.0)

        # Register the lamp's screw coupling for a native Newton mimic
        # constraint. Must happen here (not in SequentialPoseCommand
        # .__init__) since MODEL_INIT fires during sim.reset(), before
        # CommandTerms exist — see ScrewJointCouplingCfg's docstring. By the
        # time this runs, ``self.commands`` is a fully-built (and, for
        # FrankaLampEnvCfg, already-overridden) field, so the registered cfg
        # reflects whatever the concrete task actually uses.
        coupling = self.commands.pose_command.screw_coupling
        if coupling is not None and coupling.use_native_mimic_constraint:
            lamp_reset_joint_pos = getattr(self.scene, coupling.asset_name).init_state.joint_pos
            register_screw_joint_mimic(
                coupling,
                revolute_pos_at_reset=lamp_reset_joint_pos.get(coupling.revolute_joint_name, 0.0),
                prismatic_pos_at_reset=lamp_reset_joint_pos.get(coupling.prismatic_joint_name, 0.0),
            )
