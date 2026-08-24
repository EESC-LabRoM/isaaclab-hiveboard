"""Action term that preserves cuRobo's validated redundant-joint branch."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass


class CuroboPlannedDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Use direct joint waypoints only while a cuRobo command is active.

    All unplanned portions of the scripted task retain Isaac Lab's regular
    absolute-pose differential IK behavior.
    """

    cfg: "CuroboPlannedDifferentialInverseKinematicsActionCfg"

    def __init__(self, cfg: "CuroboPlannedDifferentialInverseKinematicsActionCfg", env):
        super().__init__(cfg, env)
        self._command_term = env.command_manager.get_term(cfg.command_name)

    def apply_actions(self):
        super().apply_actions()
        active, joint_targets = self._command_term.get_curobo_joint_targets()
        if not torch.any(active):
            return
        # Match Isaac Lab's own DifferentialInverseKinematicsAction call
        # shape.  The legacy articulation backend used by this workspace has
        # a faulty CUDA path for the partial-environment overload.
        self._asset.set_joint_position_target(
            joint_targets, joint_ids=self._joint_ids
        )


@configclass
class CuroboPlannedDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """Differential IK with direct execution for cuRobo-planned command terms."""

    class_type: type = CuroboPlannedDifferentialInverseKinematicsAction
    command_name: str = "pose_command"
