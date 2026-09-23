from isaaclab.utils import configclass

from isaaclab_hiveboard.tasks.franka.circuit_breaker.configs.actions import (
    FrankaBaseFrameIKAction,
    FrankaIKAbsActionCfg,
)


@configclass
class FrankaButtonActionsCfg(FrankaIKAbsActionCfg):
    """Absolute differential-IK control for the Franka canonical TCP."""

    arm_action = FrankaIKAbsActionCfg().arm_action.replace(
        class_type=FrankaBaseFrameIKAction,
    )
