"""Helpers for loading Spot robot configs into cuRobo v2."""

from __future__ import annotations

import copy
import os
from typing import Any

from curobo.config_io import load_yaml

# Keys accepted by cuRobo v0 robot YAMLs that are invalid in v2.
_LEGACY_KINEMATICS_KEYS = (
    "use_usd_kinematics",
    "isaac_usd_path",
    "usd_path",
    "usd_robot_root",
    "usd_flip_joints",
    "usd_flip_joint_limits",
    "ee_link",
)


def load_spot_robot_cfg(
    yaml_path: str,
    urdf_path: str,
) -> dict[str, Any]:
    """Load a v0 Spot YAML and rewrite it for ``InverseKinematicsCfg.create``.

    The source ``spot.yaml`` still uses the v0 schema (``ee_link``,
    ``retract_config``, USD kinematics fields). cuRobo v2 requires
    ``tool_frames``, ``default_joint_position``, and only the kinematics
    fields that ``KinematicsLoaderCfg`` accepts.

    Args:
        yaml_path: Path to the existing Spot cuRobo YAML.
        urdf_path: Absolute path to the Spot URDF used in this repo.

    Returns:
        A ``{"robot_cfg": ...}`` dict that can be passed as the ``robot``
        argument to ``InverseKinematicsCfg.create``.
    """
    if not os.path.isabs(urdf_path) or not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF must be a local file: {urdf_path}")

    data = copy.deepcopy(load_yaml(yaml_path))
    kinematics = data["robot_cfg"]["kinematics"]

    kinematics["urdf_path"] = urdf_path
    kinematics["asset_root_path"] = os.path.dirname(urdf_path)

    ee_link = kinematics.get("ee_link")
    if kinematics.get("tool_frames") is None:
        if ee_link is None:
            raise ValueError(
                f"Robot YAML '{yaml_path}' must define tool_frames or ee_link."
            )
        kinematics["tool_frames"] = [ee_link]

    cspace = kinematics.get("cspace")
    if isinstance(cspace, dict):
        if "default_joint_position" not in cspace and "retract_config" in cspace:
            cspace["default_joint_position"] = cspace["retract_config"]
        cspace.pop("retract_config", None)

        # v2 requires every cspace joint to be either active in the configured
        # kinematic tree or explicitly locked. Spot's YAML lists the legs even
        # though only the arm chain is connected to the tool frame.
        lock_joints = dict(kinematics.get("lock_joints") or {})
        default_q = cspace.get("default_joint_position") or []
        for name, q in zip(cspace.get("joint_names") or [], default_q):
            if name.startswith(("fl_", "fr_", "hl_", "hr_")) and name not in lock_joints:
                lock_joints[name] = float(q)
        kinematics["lock_joints"] = lock_joints

    for key in _LEGACY_KINEMATICS_KEYS:
        kinematics.pop(key, None)

    return data
