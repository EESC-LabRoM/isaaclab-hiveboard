"""Franka lever-valve IK with a base-frame TCP Jacobian."""

import torch

import isaaclab.utils.math as math_utils
from isaaclab.utils import configclass

from isaaclab_hiveboard.mdp.actions.curobo_planned_ik import (
    CuroboPlannedDifferentialInverseKinematicsAction,
)
from isaaclab_hiveboard.tasks.franka.circuit_breaker.configs.actions import (
    FrankaIKAbsActionCfg,
)


class FrankaLeverValveIKAction(CuroboPlannedDifferentialInverseKinematicsAction):
    """Keep TCP pose errors and their geometric Jacobian in the same frame.

    Isaac Lab 2.3.2 applies the body-local TCP offset directly to a base-frame
    Jacobian and rotates its angular rows by the fixed TCP rotation. This
    produces incorrect DLS corrections when the lever sequence leaves cuRobo
    joint tracking and starts its Cartesian retreat.
    """

    def _compute_frame_jacobian(self):
        jacobian = self.jacobian_b.clone()
        if self.cfg.body_offset is not None:
            body_quat_b = math_utils.quat_mul(
                math_utils.quat_inv(self._asset.data.root_quat_w),
                self._asset.data.body_quat_w[:, self._body_idx],
            )
            offset_pos_b = math_utils.quat_apply(body_quat_b, self._offset_pos)
            jacobian[:, :3, :] -= torch.bmm(
                math_utils.skew_symmetric_matrix(offset_pos_b), jacobian[:, 3:, :]
            )
            # The rigidly attached TCP and hand have the same angular velocity
            # expressed in the base frame, regardless of their fixed rotation.
        return jacobian


@configclass
class FrankaLeverValveActionsCfg(FrankaIKAbsActionCfg):
    """Use the TCP Jacobian correction only for this environment."""

    arm_action = FrankaIKAbsActionCfg().arm_action.replace(
        class_type=FrankaLeverValveIKAction,
    )
