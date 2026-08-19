# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

from __future__ import annotations

from isaaclab.actuators.actuator_cfg import RemotizedPDActuatorCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.assets.spot.actuators.actuator_spot import SpotKneeActuator
from isaaclab_hiveboard.assets.spot.constants import (
    NEG_TORQUE_SPEED_LIMIT,
    POS_TORQUE_SPEED_LIMIT,
)


@configclass
class SpotKneeActuatorCfg(RemotizedPDActuatorCfg):
    """Configuration for the Spot knee actuator."""

    class_type: type = SpotKneeActuator

    enable_torque_speed_limit: bool = False

    pos_torque_speed_limit = POS_TORQUE_SPEED_LIMIT
    neg_torque_speed_limit = NEG_TORQUE_SPEED_LIMIT
