# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import (
    ANYMAL_D_DYNAARM_ROBOTIQ_HIGH_PD_CFG,
    ANYMAL_EE,
    ANYMAL_WORKSPACE,
    ROBOTIQ_INIT_JOINT_POS,
    make_ee_frame,
)
from isaaclab_hiveboard.tasks.scenes.lever_valve import (
    LeverValveSceneCfg as LeverValveSceneBase,
)


@configclass
class AnymalBallValveSceneCfg(LeverValveSceneBase):
    """ANYmal-D + DynaArm + 2F-140 with the shared HiveBoard lever valve."""

    robot: ArticulationCfg = ANYMAL_D_DYNAARM_ROBOTIQ_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=ANYMAL_D_DYNAARM_ROBOTIQ_HIGH_PD_CFG.spawn.replace(
            articulation_props=ANYMAL_D_DYNAARM_ROBOTIQ_HIGH_PD_CFG.spawn.articulation_props.replace(
                fix_root_link=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
            joint_pos={
                ".*HAA": 0.0,
                ".*F_HFE": 0.4,
                ".*H_HFE": -0.4,
                ".*F_KFE": -0.8,
                ".*H_KFE": 0.8,
                # Forward-reach (ALMA standing-manipulation) instead of folded.
                "dynaarm_shoulder_rotation": 0.0,
                "dynaarm_shoulder_flexion": 0.0,
                "dynaarm_elbow_flexion": 1.7,
                "dynaarm_forearm_rotation": 0.0,
                "dynaarm_wrist_flexion": 0.0,
                "dynaarm_wrist_rotation": 1.5708,
                **ROBOTIQ_INIT_JOINT_POS,
            },
            joint_vel={".*": 0.0},
        ),
    )
    ee_frame: FrameTransformerCfg = make_ee_frame(ANYMAL_EE)

    def __post_init__(self):
        self.warehouse.init_state.pos = ANYMAL_WORKSPACE.warehouse_pos
        self.ball_valve.init_state.pos = ANYMAL_WORKSPACE.object_pos
        self.ball_valve.init_state.rot = ANYMAL_WORKSPACE.object_rot
