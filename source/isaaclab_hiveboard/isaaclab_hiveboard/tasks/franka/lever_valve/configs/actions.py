"""Franka lever-valve IK with a base-frame TCP Jacobian."""

from isaaclab.utils import configclass

from isaaclab_hiveboard.tasks.franka.circuit_breaker.configs.actions import (
    FrankaBaseFrameIKAction,
    FrankaIKAbsActionCfg,
)


@configclass
class FrankaLeverValveActionsCfg(FrankaIKAbsActionCfg):
    """Use the TCP Jacobian correction only for this environment."""

    arm_action = FrankaIKAbsActionCfg().arm_action.replace(
        class_type=FrankaBaseFrameIKAction,
    )
