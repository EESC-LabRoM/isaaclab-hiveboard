# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Read-only path sampling for the active sequential command's debug markers."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils

from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    ScrewFrameCfg,
    _CuroboPlannedGoToFrameHandler,
    _GoToFrameHandler,
    _RotateFrameHandler,
)


def active_command_path(handler, command: torch.Tensor, env_index: int = 0, samples: int = 33):
    """Return base-frame positions, xyzw orientations and the next waypoint index.

    Keep cuRobo's actual plan when available. Direct GoTo commands show their
    remaining translation/orientation motion; Rotate/Screw show the complete
    signed arc captured when that command started. Never reset or advance a
    handler here: drawing must not affect execution or start a planner.
    """
    if isinstance(handler, _CuroboPlannedGoToFrameHandler) and not handler._fallback[env_index]:
        count = int(handler._waypoint_count[env_index])
        if handler._planned[env_index] and count > 0:
            index = min(int(handler._waypoint_index[env_index]), count - 1)
            return (
                handler._waypoint_pos_b[env_index, :count],
                handler._waypoint_quat_b[env_index, :count],
                index,
            )

    fraction = torch.linspace(0.0, 1.0, samples, device=command.device, dtype=command.dtype)
    if isinstance(handler, _RotateFrameHandler):
        angle = handler.angle_rad_tensor[env_index] * fraction
        axis = handler.rot_axis_b[env_index : env_index + 1].expand(samples, -1)
        radius = handler.radius_vec[env_index : env_index + 1].expand(samples, -1)
        pos = (
            handler.axis_pos_b[env_index]
            + handler.axial_vec[env_index]
            + handler._rodrigues_rotate(radius, axis, angle)
        )
        if isinstance(handler.cfg, ScrewFrameCfg):
            pos = pos + axis * (fraction * handler.cfg.axial_distance)[:, None]
        quat = math_utils.quat_mul(
            math_utils.quat_from_angle_axis(angle, axis),
            handler.initial_quat_b[env_index : env_index + 1].expand(samples, -1),
        )
        index = int(torch.argmin(torch.abs(torch.abs(angle) - handler._progress_abs[env_index])))
        return pos, quat, index

    if isinstance(handler, _GoToFrameHandler):
        ids = torch.tensor([env_index], device=command.device)
        target_pos, target_quat = handler.get_target_in_base_frame(ids)
        start_pos, start_quat = command[env_index, 1:4], command[env_index : env_index + 1, 4:8]
        delta = target_pos[0] - start_pos
        distance = torch.linalg.vector_norm(delta).clamp_min(1e-8)
        rotation = math_utils.quat_box_minus(target_quat, start_quat)
        angle = torch.linalg.vector_norm(rotation).clamp_min(1e-8)
        duration = torch.maximum(distance / handler.cfg.velocity, angle / handler.cfg.angular_velocity)
        elapsed = fraction * duration
        pos_fraction = (elapsed * handler.cfg.velocity / distance).clamp(max=1.0)
        rot_fraction = (elapsed * handler.cfg.angular_velocity / angle).clamp(max=1.0)
        pos = start_pos + pos_fraction[:, None] * delta
        quat = math_utils.quat_box_plus(start_quat.expand(samples, -1), rot_fraction[:, None] * rotation)
        return pos, quat, 0

    # A gripper hold has no new Cartesian path.
    return None
