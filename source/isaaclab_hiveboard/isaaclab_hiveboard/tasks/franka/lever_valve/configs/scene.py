from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from isaaclab_hiveboard.assets import (
    FRANKA_EE,
    FRANKA_FR3_HIGH_PD_CFG,
    FRANKA_WORKSPACE,
    make_ee_frame,
)
from isaaclab_hiveboard.tasks.scenes.lever_valve import (
    LeverValveSceneCfg as LeverValveSceneBase,
)


@configclass
class FrankaLeverValveSceneCfg(LeverValveSceneBase):
    """Franka Research 3 + shared HiveBoard lever (ball) valve scene."""

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
        """Place the shared valve scene within the Franka workspace.

        Target-frame offsets intentionally remain inherited from
        :class:`LeverValveSceneCfg`: they are canonical TCP poses shared with
        Spot (+X approach, +Z up).  The FR3-to-canonical conversion belongs in
        ``FRANKA_EE.tcp_offset``, not in task targets.
        """
        self.warehouse.init_state.pos = FRANKA_WORKSPACE.warehouse_pos
        self.ball_valve.init_state.pos = FRANKA_WORKSPACE.object_pos
        self.ball_valve.init_state.rot = FRANKA_WORKSPACE.object_rot
        # A fixed retreat pose above the lowered handle. Its orientation is
        # the pre-grasp orientation, allowing the wrist to unwind after release.
        # Attaching it to the housing prevents lever motion from moving the goal.
        self.target_frame.target_frames.append(
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Valve/valvula_esfera",
                name="retreat",
                offset=OffsetCfg(pos=(0.12, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
            )
        )
