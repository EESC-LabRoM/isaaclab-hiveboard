# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Differential IK for a TCP offset from its rigid-body origin."""

import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction


class OffsetDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Pose IK with the wrist-to-TCP displacement expressed in the Jacobian frame."""

    def _compute_frame_jacobian(self):
        jacobian = self.jacobian_w.clone()
        if self._offset_pos is not None:
            offset_w = math_utils.quat_apply(self._asset.data.body_quat_w.torch[:, self._body_idx], self._offset_pos)
            jacobian[:, :3, :] += torch.bmm(-math_utils.skew_symmetric_matrix(offset_w), jacobian[:, 3:, :])
        world_to_base = math_utils.matrix_from_quat(math_utils.quat_inv(self._asset.data.root_quat_w.torch))
        jacobian[:, :3, :] = torch.bmm(world_to_base, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(world_to_base, jacobian[:, 3:, :])
        return jacobian

