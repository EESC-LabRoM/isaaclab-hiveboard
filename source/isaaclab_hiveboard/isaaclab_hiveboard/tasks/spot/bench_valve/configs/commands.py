# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json
import math

import torch
import isaaclab.utils.math as math_utils

from isaaclab.utils.configclass import configclass

from isaaclab_hiveboard.assets.spot.bench import (
    TCP_SITE_POS,
    TCP_SITE_QUAT_XYZW,
    HOME_ARM,
    TRAJECTORY_JSON,
    VALVE_JOINT_CLOSED,
    VALVE_JOINT_OPEN,
)
from isaaclab_hiveboard.mdp.commands.sequential_pose_command import (
    GoToFrameCfg,
    GripperCommand,
    SequentialPoseCommandCfg,
)

from isaaclab_hiveboard.utils.spot_traj import expand_keys


def _key_commands():
    """Visit enabled traj_edit beads in authored order, including gripper holds."""
    payload = json.loads(TRAJECTORY_JSON.read_text())
    keys = [
        key for key in payload.get("keys", [])
        if not key.get("off") and key.get("pos") is not None
    ]
    if not keys:
        raise ValueError(f"Trajectory {TRAJECTORY_JSON} has no enabled position keys")
    rate = float(payload.get("rate", 50))
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Trajectory rate must be positive and finite")
    # The editor authors orientations in its TCP site axes. The action uses
    # wrist axes at the same position, so remove the fixed site rotation.
    pitch = HOME_ARM[1] + HOME_ARM[2] + HOME_ARM[4]
    home_wrist = torch.tensor([0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2)])
    site_rotation = torch.tensor(TCP_SITE_QUAT_XYZW)
    home_site = math_utils.quat_mul(home_wrist, site_rotation).numpy()
    _, site_quats, _, _ = expand_keys(keys, max(int(k["sample"]) for k in keys) + 1, home_site)
    commands = []
    previous = None
    for key in keys:
        pos = tuple(float(v) for v in key["pos"])
        if len(pos) != 3 or not all(math.isfinite(v) for v in pos):
            raise ValueError(f"Invalid key position: {key!r}")
        gripper_open = float(key["grip"]) < -0.9
        commands.append(GoToFrameCfg(
            frame_name="",
            target_frame_name="",
            target_position_env=pos,
            target_orientation_env=tuple(math_utils.quat_mul(
                torch.tensor(site_quats[int(key["sample"])], dtype=torch.float32),
                math_utils.quat_inv(site_rotation),
            ).tolist()),
            gripper_open=gripper_open,
            velocity=0.15,
            distance_threshold=0.005,
            orientation_threshold_deg=2.0,
            canonicalize_upward=False,
        ))
        if previous is not None and math.dist(pos, previous["pos"]) < 0.001:
            duration = (int(key["sample"]) - int(previous["sample"])) / rate
            if duration > 0:
                commands.append(GripperCommand(open_gripper=gripper_open, duration_s=duration))
        previous = key
    return commands


@configclass
class BenchValveCommandsCfg:
    """Sequential Cartesian targets from the trajectory editor's key beads."""

    pose_command: SequentialPoseCommandCfg = SequentialPoseCommandCfg(
        asset_name="robot",
        body_name="arm_link_wr1",
        body_offset=SequentialPoseCommandCfg.OffsetCfg(pos=TCP_SITE_POS),
        resampling_time_range=(1e6, 1e6),
        commands=_key_commands(),
        valve_asset_name="ball_valve",
        valve_joint_name="RevoluteJoint",
        open_task_prob=1.0,
        valve_joint_closed=VALVE_JOINT_CLOSED,
        valve_joint_open=VALVE_JOINT_OPEN,
        debug_vis=True,
    )
