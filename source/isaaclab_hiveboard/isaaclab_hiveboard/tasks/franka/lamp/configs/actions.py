from isaaclab.utils import configclass

from isaaclab_hiveboard.tasks.franka.circuit_breaker.configs.actions import (
    FrankaBaseFrameIKAction,
    FrankaIKAbsActionCfg,
)


@configclass
class FrankaLampActionsCfg(FrankaIKAbsActionCfg):
    """Franka IK using a base-frame Jacobian for the offset canonical TCP."""

    arm_action = FrankaIKAbsActionCfg().arm_action.replace(
        class_type=FrankaBaseFrameIKAction,
    )
