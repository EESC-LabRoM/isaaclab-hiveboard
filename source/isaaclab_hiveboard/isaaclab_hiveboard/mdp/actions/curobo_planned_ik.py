"""Action term that preserves cuRobo's validated redundant-joint branch."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.mdp.pose_actions import OffsetDifferentialIKAction


class CuroboPlannedDifferentialInverseKinematicsAction(OffsetDifferentialIKAction):
    """Use direct joint waypoints only while a cuRobo command is active.

    Unplanned portions of the scripted task use offset-aware absolute-pose
    differential IK, including the lamp's coupled screw motion.
    """

    cfg: "CuroboPlannedDifferentialInverseKinematicsActionCfg"

    def __init__(self, cfg: "CuroboPlannedDifferentialInverseKinematicsActionCfg", env):
        super().__init__(cfg, env)
        self._command_term = env.command_manager.get_term(cfg.command_name)

    def apply_actions(self):
        active, joint_targets = self._command_term.get_curobo_joint_targets()
        all_planned = bool(torch.all(active))
        if not all_planned:
            super().apply_actions()
        if not torch.any(active):
            return
        # Preserve the offset-aware IK targets for unplanned environments.
        targets = joint_targets
        if not all_planned:
            ik_targets = self._asset.data.joint_pos_target.torch[:, self._joint_ids]
            targets = torch.where(active[:, None], joint_targets, ik_targets)
        self._asset.set_joint_position_target_index(target=targets, joint_ids=self._joint_ids)


@configclass
class CuroboPlannedDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """Differential IK with direct execution for cuRobo-planned command terms."""

    class_type: type = CuroboPlannedDifferentialInverseKinematicsAction
    command_name: str = "pose_command"
