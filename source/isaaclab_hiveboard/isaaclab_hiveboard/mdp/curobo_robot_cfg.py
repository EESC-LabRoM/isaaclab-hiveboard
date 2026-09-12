"""Load cuRobo robot descriptions with repo-absolute asset paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_curobo_robot_cfg(
    yaml_path: str | Path,
    urdf_path: str | Path | None = None,
    lock_joints: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Load a cuRobo robot yaml, rebasing relative asset paths onto this repo.

    cuRobo resolves ``urdf_path``/``asset_root_path`` against its own content
    tree. Our yaml lives in the repo, so rewrite those entries to absolute
    paths. The returned dict is passed straight to
    ``InverseKinematicsCfg.create`` (a dict passes ``resolve_config`` through
    unchanged) and must keep the top-level ``robot_cfg`` key.
    """
    payload = yaml.safe_load(Path(yaml_path).read_text())
    if not isinstance(payload, dict) or "robot_cfg" not in payload:
        raise ValueError(f"Expected a cuRobo robot yaml with a 'robot_cfg' key: {yaml_path}")
    kinematics = payload["robot_cfg"].setdefault("kinematics", {})
    # Drop cuRobo-v1/USD keys the v2 KinematicsLoaderCfg rejects. We solve
    # from the URDF, so none of these apply.
    for legacy_key in (
        "use_usd_kinematics",
        "isaac_usd_path",
        "usd_path",
        "usd_robot_root",
        "usd_flip_joints",
        "usd_flip_joint_limits",
        "use_global_cumul",
    ):
        kinematics.pop(legacy_key, None)
    # v2 names the tool frame list `tool_frames`; v1 yamls carry `ee_link`.
    if "tool_frames" not in kinematics and "ee_link" in kinematics:
        kinematics["tool_frames"] = [kinematics.pop("ee_link")]
    # v2 CSpaceParams calls the seed configuration `default_joint_position`.
    cspace = kinematics.get("cspace")
    if isinstance(cspace, dict) and "default_joint_position" not in cspace and "retract_config" in cspace:
        cspace["default_joint_position"] = cspace.pop("retract_config")
    if urdf_path is not None:
        urdf_path = Path(urdf_path)
        if not urdf_path.is_file():
            raise FileNotFoundError(f"cuRobo URDF not found: {urdf_path}")
        kinematics["urdf_path"] = str(urdf_path)
        kinematics["asset_root_path"] = str(urdf_path.parent)
    # Joints in cspace must be active or explicitly locked. Our sim welds the
    # body and holds the legs fixed, so lock them at their sim values here.
    if lock_joints:
        locks = kinematics.setdefault("lock_joints", {})
        locks.update(lock_joints)
    return payload
