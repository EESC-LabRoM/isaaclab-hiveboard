# Copyright (c) 2024-2026 EESC-LabRoM & The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Refresh reference poses after reset without advancing simulation time."""


def refresh_frame_sensors(env):
    """Evaluate kinematics and sample frame sensors before commands read them.

    Newton's Isaac Lab frame sensors copy native sensor buffers. Forward
    kinematics alone leaves those buffers at zero or at the previous episode.
    """
    frames = [sensor for sensor in env.scene.sensors.values() if hasattr(sensor.cfg, "target_frames")]
    if not frames:
        return
    physics = env.sim.physics_manager
    physics.forward()
    for sensor in getattr(physics, "_newton_frame_transform_sensors", ()):
        sensor.update(physics.get_state())
    for sensor in frames:
        sensor.update(0.0, force_recompute=True)
