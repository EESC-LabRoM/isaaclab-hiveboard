# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forward travel and complete removal of the contact-guided free drawer."""

from __future__ import annotations

import itertools
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from isaaclab_hiveboard.assets import ASSET_DIR
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


@lru_cache(maxsize=1)
def _shaft_corners_and_guide_end():
    """Conservative shaft bounds and guide extent from the collision URDF."""
    root = ET.parse(Path(ASSET_DIR) / "hiveboard/drawer/Drawer_Assembly.urdf").getroot()
    corners = []
    ends = []
    for collision in root.iter("collision"):
        name = collision.attrib.get("name", "")
        xyz = tuple(float(v) for v in collision.find("origin").attrib["xyz"].split())
        geometry = collision.find("geometry")
        cylinder = geometry.find("cylinder")
        if name.startswith("shaft_"):
            rpy = tuple(float(v) for v in collision.find("origin").attrib.get("rpy", "0 0 0").split())
            if not math.isclose(abs(rpy[0]), math.pi / 2, abs_tol=1e-6) or any(abs(v) > 1e-7 for v in rpy[1:]):
                raise ValueError("Drawer removal bounds require Y-axis shaft cylinders")
            radius = float(cylinder.attrib["radius"])
            half_length = float(cylinder.attrib["length"]) / 2
            corners.extend(
                tuple(c + sign * half for c, sign, half in zip(xyz, signs, (radius, half_length, radius)))
                for signs in itertools.product((-1, 1), repeat=3)
            )
        elif name.startswith("cut_"):
            half_x = (
                float(cylinder.attrib["radius"]) if cylinder is not None
                else float(geometry.find("box").attrib["size"].split()[0]) / 2
            )
            ends.append(xyz[0] + half_x)
    if len(corners) != 16 or not ends:
        raise ValueError("Drawer removal needs two Y-axis shafts and the housing cuts")
    return tuple(corners), max(ends)


def drawer_clear_of_guides(env, box_name: str, housing_name: str) -> torch.Tensor:
    """Both shaft bounds lie forward of every housing guide, including rotation."""
    box, housing = env.scene[box_name], env.scene[housing_name]
    position, orientation = math_utils.subtract_frame_transforms(
        _as_tensor(housing.data.root_pos_w), _as_tensor(housing.data.root_quat_w),
        _as_tensor(box.data.root_pos_w), _as_tensor(box.data.root_quat_w),
    )
    corners, guide_end = _shaft_corners_and_guide_end()
    points = position.new_tensor(corners).expand(position.shape[0], -1, -1)
    rotated = math_utils.quat_apply(orientation[:, None, :].expand(-1, points.shape[1], -1), points)
    minimum_x = (rotated + position[:, None, :])[..., 0].amin(dim=1)
    return minimum_x > guide_end + 0.001


def drawer_removal_success(env, command_name: str, box_name: str, housing_name: str) -> torch.Tensor:
    """Completed extraction: both shafts remain clear after release.

    Height is deliberately free after release: gravity can drop the removed
    drawer without making it part of the housing again.
    """
    displacement = drawer_displacement(env, box_name, housing_name)
    return (
        env.command_manager.get_term(command_name).is_done()
        & (displacement[:, 0] >= 0.05)
        & drawer_clear_of_guides(env, box_name, housing_name)
    )
