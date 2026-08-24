from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import (
    FRANKA_EE,
    FRANKA_FR3_HIGH_PD_CFG,
    FRANKA_WORKSPACE,
    make_ee_frame,
)
from isaaclab_hiveboard.tasks.scenes.circuit_breaker import (
    CircuitBreakerSceneCfg as CircuitBreakerSceneBase,
)


@configclass
class FrankaCircuitBreakerSceneCfg(CircuitBreakerSceneBase):
    """Franka Research 3 + shared HiveBoard circuit breaker scene."""

    robot: ArticulationCfg = FRANKA_FR3_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "fr3_joint1": 0.0,
                "fr3_joint2": -0.569,
                "fr3_joint3": 0.0,
                "fr3_joint4": -2.810,
                "fr3_joint5": 0.0,
                "fr3_joint6": 3.037,
                "fr3_joint7": 0.741,
                "fr3_finger_joint.*": 0.04,
            },
        ),
    )
    ee_frame: FrameTransformerCfg = make_ee_frame(FRANKA_EE)

    def __post_init__(self):
        self.warehouse.init_state.pos = FRANKA_WORKSPACE.warehouse_pos
        self.circuit_breaker.init_state.pos = FRANKA_WORKSPACE.object_pos
        self.circuit_breaker.init_state.rot = FRANKA_WORKSPACE.object_rot
        # Start DOWN so below → above actually drives the paddle up.
        self.circuit_breaker.init_state.joint_pos["RevoluteJoint"] = 0.0
        # Keep the canonical target-frame offsets and orientation from the
        # shared scene.  Its identity rotation makes +X point toward the
        # circuit-breaker panel, matching Spot and preserving identical
        # scenario goal poses for every robot.
