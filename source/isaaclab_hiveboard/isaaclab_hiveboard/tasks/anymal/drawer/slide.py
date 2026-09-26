# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forward travel of the free drawer box in the housing frame.

Open is this distance, still inside the slot. The box has no joint.
"""

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from isaaclab_hiveboard.tasks.anymal.drawer.configs.commands import DRAWER_OPEN


def _as_tensor(value) -> torch.Tensor:
    return value.torch if hasattr(value, "torch") else value


def drawer_displacement(env, box_name: str, housing_name: str) -> torch.Tensor:
    """Box origin in the housing frame, shape ``(num_envs, 3)``.

    ``x`` is the pull, ``z`` is the rise. Quaternions are ``xyzw``.
    """
    box = env.scene[box_name]
    housing = env.scene[housing_name]
    position, _orientation = math_utils.subtract_frame_transforms(
        _as_tensor(housing.data.root_pos_w),
        _as_tensor(housing.data.root_quat_w),
        _as_tensor(box.data.root_pos_w),
        _as_tensor(box.data.root_quat_w),
    )
    return position


def drawer_forward(env, box_name: str, housing_name: str) -> torch.Tensor:
    """Pull distance in metres, shape ``(num_envs,)``."""
    return drawer_displacement(env, box_name, housing_name)[:, 0]


def drawer_slide_observation(env, box_cfg: SceneEntityCfg, housing_cfg: SceneEntityCfg) -> torch.Tensor:
    """Pull distance as a column, the observation that used to be a joint."""
    return drawer_forward(env, box_cfg.name, housing_cfg.name).unsqueeze(-1)


def travel_in_range(forward: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """True where forward travel sits in ``[low, high]``. Lift is not consulted."""
    return (forward >= float(low)) & (forward <= float(high))


def done_when_travel_holds(env, env_ids: torch.Tensor, spec: tuple[str, str, float, float]) -> torch.Tensor:
    """Pull-completion for ``(box, housing, low, high)`` on ``env_ids``."""
    box_name, housing_name, low, high = spec
    forward = drawer_forward(env, box_name, housing_name)[env_ids]
    return travel_in_range(forward, low, high)


def drawer_slide_success(
    env,
    command_name: str,
    box_name: str,
    housing_name: str,
    open_distance: float = DRAWER_OPEN,
) -> torch.Tensor:
    """Sequence finished and the box pulled at least ``open_distance`` metres.

    A rise with no forward travel is not open. Removal is not required.
    """
    command = env.command_manager.get_term(command_name)
    if not hasattr(command, "is_done"):
        raise AttributeError(f"The command term '{command_name}' does not have the method 'is_done'.")
    forward = drawer_forward(env, box_name, housing_name)
    return command.is_done() & travel_in_range(forward, open_distance, 1.0)
