# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets import ANYMAL_EE, make_ee_frame
from isaaclab_hiveboard.assets.anymal.anymal import ROBOTIQ_LEFT_PAD_PRIM, ROBOTIQ_RIGHT_PAD_PRIM
from isaaclab_hiveboard.assets.anymal.bench import ANYMAL_ARM_NEWTON_CFG
from isaaclab_hiveboard.tasks.scenes.circuit_breaker import (
    CircuitBreakerSceneCfg as CircuitBreakerSceneBase,
)

_BREAKER_CONTACT_FILTER = ["{ENV_REGEX_NS}/CircuitBreaker/.*"]


@configclass
class CircuitBreakerSceneCfg(CircuitBreakerSceneBase):
    """ANYmal + DynaArm + shared HiveBoard circuit breaker scene."""

    robot: ArticulationCfg = ANYMAL_ARM_NEWTON_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    ee_frame: FrameTransformerCfg = make_ee_frame(ANYMAL_EE)
    finger_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + ROBOTIQ_LEFT_PAD_PRIM,
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=_BREAKER_CONTACT_FILTER,
    )
    jaw_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + ROBOTIQ_RIGHT_PAD_PRIM,
        update_period=0.0,
        history_length=1,
        filter_prim_paths_expr=_BREAKER_CONTACT_FILTER,
    )
