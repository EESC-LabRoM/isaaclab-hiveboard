"""Absolute joint-position action driven by cuRobo command waypoints."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils.configclass import configclass


class CuroboJointPositionAction(JointPositionAction):
    """Apply the cuRobo command's joint waypoints as absolute position targets.

    The :class:`SequentialPoseCommand` term already publishes validated joint
    waypoints for cuRobo-planned segments, so no differential IK is needed and
    policy outputs are ignored. Where the command has no joint plan (gripper
    holds, fallback servo), the last targets are held.
    """

    cfg: "CuroboJointPositionActionCfg"

    def __init__(self, cfg: "CuroboJointPositionActionCfg", env):
        super().__init__(cfg, env)
        self._command_term = env.command_manager.get_term(cfg.command_name)
        self._held = self._asset.data.default_joint_pos.torch[:, self._joint_ids].clone()

    def apply_actions(self):
        active, joint_targets = self._command_term.get_curobo_joint_targets()
        if torch.any(active):
            self._held[active] = joint_targets[active].to(self._held.dtype)
        # Match the full-call pattern used by the cuRobo IK action: the
        # legacy articulation backend used by this workspace has a faulty
        # CUDA path for the partial-environment overload.
        self._asset.set_joint_position_target(self._held, joint_ids=self._joint_ids)


@configclass
class CuroboJointPositionActionCfg(JointPositionActionCfg):
    """Absolute joint positions sourced from the cuRobo command term."""

    class_type: type = CuroboJointPositionAction
    command_name: str = "pose_command"
